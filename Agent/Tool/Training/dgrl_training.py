from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite, prod
from random import Random

import torch
from torch.nn import functional

from Agent.Proposed.dgrl import (
    LexicographicValue,
    clipped_double_q,
    dbu_loss,
    dbu_target,
    sample_gaussian_dbu_indices,
)


@dataclass(frozen=True)
class LexicographicRewardConfig:
    primary_gain_bounds: tuple[float, float]
    secondary_gain_bounds: tuple[float, float]
    failure_cost_bounds: tuple[float, float]
    runtime_cost_bounds: tuple[float, float]
    levels: tuple[int, int, int, int]

    def __post_init__(self):
        bounds = (
            ("primary_gain", self.primary_gain_bounds),
            ("secondary_gain", self.secondary_gain_bounds),
            ("failure_cost", self.failure_cost_bounds),
            ("runtime_cost", self.runtime_cost_bounds),
        )
        for name, (lower, upper) in bounds:
            if not isfinite(float(lower)) or not isfinite(float(upper)):
                raise ValueError(f"{name} bounds must be finite")
            if float(lower) >= float(upper):
                raise ValueError(f"{name} lower bound must be below upper bound")
        if len(self.levels) != 4 or any(int(level) < 2 for level in self.levels):
            raise ValueError("reward levels must contain four integers of at least two")


def _reward_digit(value, bounds, levels, name, *, higher_is_better):
    lower, upper = (float(item) for item in bounds)
    value = float(value)
    if value < lower or value > upper:
        raise ValueError(f"{name} lies outside frozen reward bounds")
    digit = int(round((value - lower) / (upper - lower) * (levels - 1)))
    return digit if higher_is_better else levels - 1 - digit


def encode_lexicographic_reward(reward, config):
    values = (
        (reward.primary_gain, config.primary_gain_bounds, True, "primary_gain"),
        (
            reward.secondary_gain,
            config.secondary_gain_bounds,
            True,
            "secondary_gain",
        ),
        (reward.failure_cost, config.failure_cost_bounds, False, "failure_cost"),
        (reward.runtime_cost, config.runtime_cost_bounds, False, "runtime_cost"),
    )
    digits = [
        _reward_digit(
            value,
            bounds,
            levels,
            name,
            higher_is_better=higher_is_better,
        )
        for (value, bounds, higher_is_better, name), levels in zip(
            values,
            config.levels,
        )
    ]
    code = digits[0]
    for digit, radix in zip(digits[1:], config.levels[1:]):
        code = code * radix + digit
    return 2.0 * float(code) / float(prod(config.levels) - 1) - 1.0


@dataclass(frozen=True)
class DGRLTransition:
    graph: object
    context: torch.Tensor
    action: object
    reward: LexicographicValue
    dbu_action_pool: tuple
    next_graph: object
    next_context: torch.Tensor
    next_actions: tuple
    done: bool


class DGRLReplayBuffer:
    def __init__(self, capacity, seed):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items = deque(maxlen=int(capacity))
        self._rng = Random(seed)

    def __len__(self):
        return len(self._items)

    def append(self, transition):
        if not isinstance(transition, DGRLTransition):
            raise TypeError("transition must be a DGRLTransition")
        self._items.append(transition)

    def sample(self, batch_size):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._items):
            raise ValueError("batch_size exceeds replay buffer size")
        return tuple(self._rng.sample(tuple(self._items), batch_size))


@dataclass(frozen=True)
class DGRLTrainingStats:
    batch_size: int
    actor_loss: float
    critic_loss: float


def _reward_tensor(reward, reference, config):
    return reference.new_tensor(encode_lexicographic_reward(reward, config))


class DGRLTrainer:
    def __init__(
        self,
        model,
        *,
        seed,
        actor_learning_rate,
        critic_learning_rate,
        gamma,
        tau,
        dbu_temperature,
        dbu_noise_std,
        dbu_sample_count,
        reward_config,
    ):
        if actor_learning_rate <= 0 or critic_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 < tau <= 1:
            raise ValueError("tau must be in (0, 1]")
        if dbu_temperature <= 0:
            raise ValueError("dbu_temperature must be positive")
        if dbu_noise_std <= 0:
            raise ValueError("dbu_noise_std must be positive")
        if dbu_sample_count <= 0:
            raise ValueError("dbu_sample_count must be positive")
        if not isinstance(reward_config, LexicographicRewardConfig):
            raise TypeError("reward_config must be a LexicographicRewardConfig")
        self.model = model
        self.target_model = deepcopy(model)
        self.target_model.eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.dbu_temperature = float(dbu_temperature)
        self.dbu_noise_std = float(dbu_noise_std)
        self.dbu_sample_count = int(dbu_sample_count)
        self.reward_config = reward_config
        self.dbu_generator = torch.Generator().manual_seed(int(seed))
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(),
            lr=float(actor_learning_rate),
        )
        critic_parameters = (
            tuple(self.model.encoder.parameters())
            + tuple(self.model.context_projection.parameters())
            + tuple(self.model.critic1.parameters())
            + tuple(self.model.critic2.parameters())
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters,
            lr=float(critic_learning_rate),
        )

    def _target_value(self, transition, reference):
        reward = _reward_tensor(
            transition.reward,
            reference,
            self.reward_config,
        )
        if transition.done or not transition.next_actions:
            return reward
        q1, q2 = self.target_model.q_values(
            transition.next_graph,
            transition.next_context,
            transition.next_actions,
        )
        values = clipped_double_q(q1, q2)
        return reward + self.gamma * values.max()

    def _critic_loss(self, batch):
        losses = []
        for transition in batch:
            q1, q2 = self.model.q_values(
                transition.graph,
                transition.context,
                (transition.action,),
            )
            with torch.no_grad():
                target = self._target_value(transition, q1)
            losses.append(functional.mse_loss(q1[0], target))
            losses.append(functional.mse_loss(q2[0], target))
        return torch.stack(losses).mean()

    def _actor_loss(self, batch):
        losses = []
        for transition in batch:
            candidates = tuple(transition.dbu_action_pool)
            if not candidates:
                raise ValueError("DBU action pool must not be empty")
            with torch.no_grad():
                node_state, global_state = self.model._encode_state(
                    transition.graph,
                    transition.context,
                )
                embeddings = self.model._action_embeddings_from_nodes(
                    transition.graph,
                    node_state,
                    candidates,
                )
            proto = self.model._actor_vector(global_state.detach())
            with torch.no_grad():
                indices = sample_gaussian_dbu_indices(
                    proto.detach(),
                    embeddings,
                    sample_count=self.dbu_sample_count,
                    noise_std=self.dbu_noise_std,
                    generator=self.dbu_generator,
                )
                selected_actions = tuple(candidates[int(index)] for index in indices)
                selected_embeddings = embeddings[indices]
                q1, q2 = self.model.q_values_from_state(
                    transition.graph,
                    node_state,
                    global_state,
                    selected_actions,
                )
                q_values = clipped_double_q(q1, q2)
                target = dbu_target(
                    selected_embeddings,
                    q_values,
                    self.dbu_temperature,
                )
            losses.append(dbu_loss(proto, target))
        return torch.stack(losses).mean()

    def _soft_update_target(self):
        with torch.no_grad():
            for target, source in zip(
                self.target_model.parameters(),
                self.model.parameters(),
            ):
                target.mul_(1.0 - self.tau).add_(source, alpha=self.tau)

    def train_step(self, batch):
        batch = tuple(batch)
        if not batch:
            raise ValueError("training batch must not be empty")
        self.model.train()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss = self._critic_loss(batch)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss = self._actor_loss(batch)
        actor_loss.backward()
        self.actor_optimizer.step()
        self._soft_update_target()
        return DGRLTrainingStats(
            len(batch),
            float(actor_loss.detach()),
            float(critic_loss.detach()),
        )


__all__ = [
    "DGRLReplayBuffer",
    "DGRLTrainer",
    "DGRLTrainingStats",
    "DGRLTransition",
    "LexicographicRewardConfig",
    "encode_lexicographic_reward",
]
