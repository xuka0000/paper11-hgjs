from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from Agent.Tool.Training.dgrl_training import (
    DGRLTrainer,
    LexicographicRewardConfig,
)


DGRL_CORE_VERSION = "dgrl-sdn-dbu-scalar-q-v2"


@dataclass(frozen=True)
class DGRLTrainingConfig:
    seed: int
    actor_learning_rate: float
    critic_learning_rate: float
    gamma: float
    tau: float
    dbu_temperature: float
    dbu_noise_std: float
    dbu_sample_count: int
    sdn_exploration_temperature: float
    reward_config: LexicographicRewardConfig

    def __post_init__(self):
        if self.actor_learning_rate <= 0 or self.critic_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0, 1]")
        if self.dbu_temperature <= 0:
            raise ValueError("dbu_temperature must be positive")
        if self.dbu_noise_std <= 0:
            raise ValueError("dbu_noise_std must be positive")
        if self.dbu_sample_count <= 0:
            raise ValueError("dbu_sample_count must be positive")
        if not 0 < self.sdn_exploration_temperature <= 1:
            raise ValueError("sdn_exploration_temperature must be in (0, 1]")
        if not isinstance(self.reward_config, LexicographicRewardConfig):
            raise TypeError("reward_config must be a LexicographicRewardConfig")


def build_dgrl_trainer(model, config):
    if not isinstance(config, DGRLTrainingConfig):
        raise TypeError("config must be a DGRLTrainingConfig")
    return DGRLTrainer(
        model,
        seed=config.seed,
        actor_learning_rate=config.actor_learning_rate,
        critic_learning_rate=config.critic_learning_rate,
        gamma=config.gamma,
        tau=config.tau,
        dbu_temperature=config.dbu_temperature,
        dbu_noise_std=config.dbu_noise_std,
        dbu_sample_count=config.dbu_sample_count,
        reward_config=config.reward_config,
    )


def save_frozen_dgrl_checkpoint(
    path,
    model,
    config,
    *,
    manifest_hash,
    train_instance_ids,
    validation_instance_ids,
    test_instances_used,
    source_sha256="",
    initialization=None,
):
    if int(test_instances_used) != 0:
        raise ValueError("test instances must not be used before model freezing")
    train_ids = tuple(sorted(set(train_instance_ids)))
    validation_ids = tuple(sorted(set(validation_instance_ids)))
    if set(train_ids) & set(validation_ids):
        raise ValueError("training and validation instance overlap")
    if not str(manifest_hash).strip():
        raise ValueError("manifest_hash must not be empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "CHDGRL-LNS",
        "base_algorithm": "DGRL",
        "algorithm_core_version": DGRL_CORE_VERSION,
        "critic_output": "scalar_twin_q",
        "dbu_candidate_source": "independent_gaussian_projection",
        "reward_encoding": "bounded_mixed_radix_lexicographic",
        "status": "frozen",
        "training": asdict(config),
        "manifest_hash": str(manifest_hash),
        "train_instance_ids": train_ids,
        "validation_instance_ids": validation_ids,
        "test_instances_used": 0,
        "source_sha256": str(source_sha256),
        "initialization": dict(initialization or {}),
        "feature_dim": model.encoder.input_projection.in_features,
        "hidden_dim": model.hidden_dim,
        "layers": len(model.encoder.layers),
        "context_dim": model.context_projection.in_features,
        "release_levels": model.release_levels,
        "time_levels": model.time_levels,
        "model_state": model.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_frozen_dgrl_checkpoint(path, model):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("algorithm") != "CHDGRL-LNS":
        raise ValueError("checkpoint algorithm mismatch")
    if payload.get("algorithm_core_version") != DGRL_CORE_VERSION:
        raise ValueError("checkpoint DGRL core version mismatch")
    if payload.get("status") != "frozen":
        raise ValueError("checkpoint is not frozen")
    if int(payload.get("test_instances_used", -1)) != 0:
        raise ValueError("checkpoint used test instances")
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {key: value for key, value in payload.items() if key != "model_state"}


__all__ = [
    "DGRL_CORE_VERSION",
    "DGRLTrainingConfig",
    "build_dgrl_trainer",
    "load_frozen_dgrl_checkpoint",
    "save_frozen_dgrl_checkpoint",
]
