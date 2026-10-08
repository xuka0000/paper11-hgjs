from dataclasses import dataclass
import json
from pathlib import Path

from Agent.Proposed.closure import ClosureBudget
from Agent.Proposed.dgrl_actions import RelativeClosureBudget, ReleaseProfile
from Agent.Proposed.regions import RegionBudget
from Agent.Proposed.search import SearchConfig
from Agent.Tool.Training.dgrl_training import LexicographicRewardConfig
from Agent.Tool.Training.dgrl_training_runner import DGRLTrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "Data" / "Config" / "chdgrl_lns" / "formal_v1.json"
)
SCALE_NORMALIZED_CONFIG_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Config"
    / "chdgrl_lns"
    / "scale_normalized_v2.json"
)


@dataclass(frozen=True)
class CHDGRLModelConfig:
    hidden_dim: int
    layers: int
    context_dim: int


@dataclass(frozen=True)
class CHDGRLLoopConfig:
    epochs: int
    patience: int
    replay_capacity: int
    batch_size: int
    updates_per_epoch: int
    independent_pool_size: int
    trajectories_per_instance: int


@dataclass(frozen=True)
class CHDGRLSearchConfig:
    release_profiles: tuple[ReleaseProfile, ...]
    time_budgets: tuple[float, ...]
    controller_sample_count: int
    controller_radius: float
    max_iterations: int
    no_improve_limit: int
    total_time_limit_s: float
    objective_weights: dict[str, float]
    controller_guard_count: int = 8

    def __post_init__(self):
        if self.controller_guard_count < 0:
            raise ValueError("controller_guard_count must be nonnegative")

    def to_search_config(self):
        return SearchConfig(
            max_iterations=self.max_iterations,
            no_improve_limit=self.no_improve_limit,
            top_k=1,
            region_budgets=tuple(
                profile.region_budget for profile in self.release_profiles
            ),
            closure_budget=max(
                (profile.closure_budget for profile in self.release_profiles),
                key=lambda item: (
                    item.max_tasks,
                    item.max_tuples,
                    item.max_vehicles,
                    item.max_span,
                ),
            ),
            repair_time_limit_s=max(self.time_budgets),
            acceptance_mode="first_improvement",
            total_time_limit_s=self.total_time_limit_s,
            **self.objective_weights,
        )


@dataclass(frozen=True)
class CHDGRLExperimentConfig:
    config_version: str
    behavior_seed: int
    training_seeds: tuple[int, ...]
    development_base_seed: int
    test_base_seed: int
    model: CHDGRLModelConfig
    training: CHDGRLLoopConfig
    trainer: DGRLTrainingConfig
    reward: LexicographicRewardConfig
    search: CHDGRLSearchConfig


def _region_budget(payload):
    return RegionBudget(
        int(payload["max_tasks"]),
        int(payload["max_tuples"]),
        int(payload["max_vehicles"]),
        int(payload["max_time_span"]),
    )


def _closure_budget(payload):
    return ClosureBudget(
        int(payload["max_tasks"]),
        int(payload["max_tuples"]),
        int(payload["max_vehicles"]),
        int(payload["max_time_span"]),
    )


def _relative_closure_budget(payload):
    if payload is None:
        return None
    return RelativeClosureBudget(
        float(payload["task_fraction"]),
        float(payload["tuple_fraction"]),
        float(payload["vehicle_fraction"]),
        float(payload["span_fraction"]),
    )


def load_chdgrl_experiment_config(path=DEFAULT_CONFIG_PATH):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    reward_payload = payload["reward"]
    reward = LexicographicRewardConfig(
        tuple(float(value) for value in reward_payload["primary_gain_bounds"]),
        tuple(float(value) for value in reward_payload["secondary_gain_bounds"]),
        tuple(float(value) for value in reward_payload["failure_cost_bounds"]),
        tuple(float(value) for value in reward_payload["runtime_cost_bounds"]),
        tuple(int(value) for value in reward_payload["levels"]),
    )
    trainer_payload = payload["trainer"]
    trainer = DGRLTrainingConfig(
        seed=int(payload["training_seeds"][0]),
        actor_learning_rate=float(trainer_payload["actor_learning_rate"]),
        critic_learning_rate=float(trainer_payload["critic_learning_rate"]),
        gamma=float(trainer_payload["gamma"]),
        tau=float(trainer_payload["tau"]),
        dbu_temperature=float(trainer_payload["dbu_temperature"]),
        dbu_noise_std=float(trainer_payload["dbu_noise_std"]),
        dbu_sample_count=int(trainer_payload["dbu_sample_count"]),
        sdn_exploration_temperature=float(
            trainer_payload["sdn_exploration_temperature"]
        ),
        reward_config=reward,
    )
    search_payload = payload["search"]
    profiles = tuple(
        ReleaseProfile(
            _region_budget(profile["region"]),
            _closure_budget(profile["closure"]),
            _relative_closure_budget(profile.get("relative_closure")),
        )
        for profile in search_payload["release_profiles"]
    )
    return CHDGRLExperimentConfig(
        str(payload["config_version"]),
        int(payload["behavior_seed"]),
        tuple(int(seed) for seed in payload["training_seeds"]),
        int(payload["development_base_seed"]),
        int(payload["test_base_seed"]),
        CHDGRLModelConfig(**payload["model"]),
        CHDGRLLoopConfig(**payload["training"]),
        trainer,
        reward,
        CHDGRLSearchConfig(
            profiles,
            tuple(float(value) for value in search_payload["time_budgets"]),
            int(search_payload["controller_sample_count"]),
            float(search_payload["controller_radius"]),
            int(search_payload["max_iterations"]),
            int(search_payload["no_improve_limit"]),
            float(search_payload["total_time_limit_s"]),
            {
                key: float(value)
                for key, value in search_payload["objective_weights"].items()
            },
            int(search_payload.get("controller_guard_count", 8)),
        ),
    )


__all__ = [
    "CHDGRLExperimentConfig",
    "CHDGRLLoopConfig",
    "CHDGRLModelConfig",
    "CHDGRLSearchConfig",
    "DEFAULT_CONFIG_PATH",
    "SCALE_NORMALIZED_CONFIG_PATH",
    "load_chdgrl_experiment_config",
]
