from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from .dgrl_actions import NEIGHBORHOOD_TYPES
from .hgnn import RoleAwareHGNN


@dataclass(frozen=True)
class LexicographicValue:
    primary_gain: float
    secondary_gain: float
    runtime_cost: float
    failure_cost: float


@dataclass(frozen=True)
class ProtoAction:
    vector: torch.Tensor


def lexicographically_better(left, right, epsilon=1e-9):
    if left.primary_gain > right.primary_gain + epsilon:
        return True
    if right.primary_gain > left.primary_gain + epsilon:
        return False
    if left.secondary_gain > right.secondary_gain + epsilon:
        return True
    if right.secondary_gain > left.secondary_gain + epsilon:
        return False
    if left.failure_cost < right.failure_cost - epsilon:
        return True
    if right.failure_cost < left.failure_cost - epsilon:
        return False
    return left.runtime_cost < right.runtime_cost - epsilon


def clipped_double_q(q1, q2):
    if q1.shape != q2.shape:
        raise ValueError("twin Q values must have equal shapes")
    if q1.ndim != 1:
        raise ValueError("twin Q values must contain one scalar per candidate")
    return torch.minimum(q1, q2)


def dbu_target(action_embeddings, q_values, temperature):
    if action_embeddings.ndim != 2:
        raise ValueError("action_embeddings must be a matrix")
    if q_values.ndim != 1:
        raise ValueError("q_values must contain one scalar per candidate")
    if action_embeddings.shape[0] != q_values.shape[0]:
        raise ValueError("candidate embeddings and Q values must have equal rows")
    if action_embeddings.shape[0] == 0:
        raise ValueError("at least one DBU candidate is required")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    weights = torch.softmax(q_values / float(temperature), dim=0)
    return (weights.unsqueeze(1) * action_embeddings).sum(dim=0)


def dbu_loss(proto_action, target_action):
    if proto_action.shape != target_action.shape:
        raise ValueError("proto and target actions must have equal shapes")
    return functional.smooth_l1_loss(proto_action, target_action)


def sample_gaussian_dbu_indices(
    proto_action,
    action_embeddings,
    *,
    sample_count,
    noise_std,
    generator,
):
    if proto_action.ndim != 1:
        raise ValueError("proto_action must be a vector")
    if action_embeddings.ndim != 2:
        raise ValueError("action_embeddings must be a matrix")
    if action_embeddings.shape[0] == 0:
        raise ValueError("DBU action pool must not be empty")
    if action_embeddings.shape[1] != proto_action.shape[0]:
        raise ValueError("proto action and action embeddings must share a dimension")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if noise_std <= 0:
        raise ValueError("noise_std must be positive")
    noise = torch.randn(
        (int(sample_count), proto_action.shape[0]),
        dtype=proto_action.dtype,
        device=proto_action.device,
        generator=generator,
    )
    perturbed = proto_action.unsqueeze(0) + float(noise_std) * noise
    nearest = torch.cdist(perturbed, action_embeddings).argmin(dim=1)
    unique = []
    seen = set()
    for index in nearest.tolist():
        if index not in seen:
            unique.append(index)
            seen.add(index)
    return torch.tensor(unique, dtype=torch.long, device=proto_action.device)


def rank_action_probabilities(q_values, temperature):
    if q_values.ndim != 1 or q_values.shape[0] == 0:
        raise ValueError("q_values must contain at least one scalar candidate value")
    if not 0 < temperature <= 1:
        raise ValueError("rank temperature must be in (0, 1]")
    detached = q_values.detach().cpu()
    order = sorted(
        range(q_values.shape[0]),
        key=lambda index: (-float(detached[index]), index),
    )
    ranks = torch.empty_like(q_values)
    for rank, index in enumerate(order):
        ranks[index] = float(rank)
    weights = torch.pow(q_values.new_tensor(float(temperature)), ranks)
    return weights / weights.sum()


def select_scalar_action_index(q_values, *, training, temperature, rng):
    if q_values.ndim != 1 or q_values.shape[0] == 0:
        raise ValueError("q_values must contain at least one scalar candidate value")
    if not training:
        return int(torch.argmax(q_values).item())
    probabilities = rank_action_probabilities(q_values, temperature)
    return int(
        rng.choices(
            range(q_values.shape[0]),
            weights=probabilities.detach().cpu().tolist(),
            k=1,
        )[0]
    )


class HypergraphDGRL(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        layers,
        context_dim,
        release_levels,
        time_levels,
    ):
        super().__init__()
        if release_levels <= 0 or time_levels <= 0:
            raise ValueError("action levels must be positive")
        self.hidden_dim = int(hidden_dim)
        self.release_levels = int(release_levels)
        self.time_levels = int(time_levels)
        self.action_dim = len(NEIGHBORHOOD_TYPES) + self.hidden_dim + 2
        self.encoder = RoleAwareHGNN(feature_dim, hidden_dim, layers)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )
        critic_input_dim = hidden_dim + self.action_dim
        self.critic1 = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic2 = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_state(self, graph, context):
        node_state = self.encoder.encode(graph)
        context = torch.as_tensor(
            context,
            dtype=node_state.dtype,
            device=node_state.device,
        )
        if context.ndim != 1:
            raise ValueError("context must be a one-dimensional tensor")
        global_state = node_state.mean(dim=0) + self.context_projection(context)
        return node_state, global_state

    def _actor_vector(self, global_state):
        raw = self.actor(global_state)
        type_end = len(NEIGHBORHOOD_TYPES)
        seed_end = type_end + self.hidden_dim
        type_proto = torch.softmax(raw[:type_end], dim=0)
        seed_proto = functional.normalize(
            torch.tanh(raw[type_end:seed_end]),
            dim=0,
        )
        ordinal_proto = torch.sigmoid(raw[seed_end:])
        return torch.cat((type_proto, seed_proto, ordinal_proto), dim=0)

    def proto_action(self, graph, context):
        _, global_state = self._encode_state(graph, context)
        return self.proto_action_from_state(global_state)

    def encode_state(self, graph, context):
        return self._encode_state(graph, context)

    def proto_action_from_state(self, global_state):
        return ProtoAction(self._actor_vector(global_state))

    def _action_embeddings_from_nodes(self, graph, node_state, actions):
        node_index = {node_id: index for index, node_id in enumerate(graph.node_ids)}
        embeddings = []
        for action in actions:
            if action.seed_id not in node_index:
                raise ValueError("action seed is not present in the tensor hypergraph")
            type_vector = node_state.new_zeros(len(NEIGHBORHOOD_TYPES))
            type_vector[NEIGHBORHOOD_TYPES.index(action.neighborhood_type)] = 1.0
            seed_vector = functional.normalize(
                node_state[node_index[action.seed_id]],
                dim=0,
            )
            release = node_state.new_tensor(
                [action.release_profile / max(1, self.release_levels - 1)]
            )
            time_budget = node_state.new_tensor(
                [action.time_budget / max(1, self.time_levels - 1)]
            )
            embeddings.append(
                torch.cat((type_vector, seed_vector, release, time_budget), dim=0)
            )
        if not embeddings:
            return node_state.new_empty((0, self.action_dim))
        return torch.stack(embeddings)

    def action_embeddings(self, graph, actions):
        node_state = self.encoder.encode(graph)
        return self._action_embeddings_from_nodes(graph, node_state, actions)

    def q_values(self, graph, context, actions):
        node_state, global_state = self._encode_state(graph, context)
        return self.q_values_from_state(
            graph,
            node_state,
            global_state,
            actions,
        )

    def q_values_from_state(
        self,
        graph,
        node_state,
        global_state,
        actions,
    ):
        action_embeddings = self._action_embeddings_from_nodes(
            graph,
            node_state,
            actions,
        )
        critic_input = torch.cat(
            (
                global_state.expand(action_embeddings.shape[0], -1),
                action_embeddings,
            ),
            dim=1,
        )
        return (
            self.critic1(critic_input).squeeze(-1),
            self.critic2(critic_input).squeeze(-1),
        )


__all__ = [
    "HypergraphDGRL",
    "LexicographicValue",
    "ProtoAction",
    "clipped_double_q",
    "dbu_loss",
    "dbu_target",
    "rank_action_probabilities",
    "sample_gaussian_dbu_indices",
    "select_scalar_action_index",
    "lexicographically_better",
]
