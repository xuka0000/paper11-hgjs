from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import torch

from Agent.Baseline.exact_milp import ExactMilpBaseline
from Agent.Baseline.manual_alns import ManualAlnsBaseline
from Agent.Baseline.risk_greedy import RiskPriorityGreedyBaseline
from Agent.Proposed.dgrl import HypergraphDGRL
from Agent.Proposed.dgrl_actions import JointAction, project_action
from Agent.Proposed.dgrl_controller import DGRLController, DGRLDecision, _neighborhood_type
from Agent.Proposed.search import run_search
from Agent.Tool.Evaluation.method_runner import MethodSpec
from Agent.Tool.Training.chdgrl_encoder_ablation import (
    ENCODER_ABLATION_VERSION,
    OrdinaryGraphDGRL,
)
from Agent.Tool.Training.dgrl_training_runner import DGRL_CORE_VERSION


EVALUATION_SOURCE_FILES = (
    "Agent/Baseline/common.py",
    "Agent/Baseline/exact_milp.py",
    "Agent/Baseline/manual_alns.py",
    "Agent/Baseline/risk_greedy.py",
    "Agent/Proposed/closure.py",
    "Agent/Proposed/dgrl.py",
    "Agent/Proposed/dgrl_actions.py",
    "Agent/Proposed/dgrl_controller.py",
    "Agent/Proposed/gnn.py",
    "Agent/Proposed/hgnn.py",
    "Agent/Proposed/hypergraph.py",
    "Agent/Proposed/milp.py",
    "Agent/Proposed/regions.py",
    "Agent/Proposed/repair.py",
    "Agent/Proposed/search.py",
    "Agent/Proposed/validator.py",
    "Agent/Tool/Benchmark/benchmark_v3.py",
    "Agent/Tool/Benchmark/formal_v3_loader.py",
    "Agent/Tool/Evaluation/chdgrl_evaluation.py",
    "Agent/Tool/Evaluation/chdgrl_methods.py",
    "Agent/Tool/Evaluation/method_runner.py",
    "Agent/Tool/Evaluation/dynamic_evaluation.py",
    "Agent/Tool/Evaluation/evaluation_revision.py",
    "Agent/Tool/Evaluation/result_io.py",
    "Agent/Tool/Training/chdgrl_config.py",
    "Agent/Tool/Training/chdgrl_encoder_ablation.py",
    "Agent/Tool/Training/dgrl_training_runner.py",
    "Env/Execution.py",
    "Env/State.py",
    "Env/Runtime.py",
    "Env/Trigger.py",
    "Mask/ServiceTuple.py",
    "Mask/TupleFactory.py",
    "Mask/Vehicle.py",
    "Mask/UAV.py",
)


def _file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def evaluation_source_hashes(project_root):
    project_root = Path(project_root)
    hashes = {}
    for relative_path in EVALUATION_SOURCE_FILES:
        path = project_root / relative_path
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[relative_path] = _file_hash(path)
    return hashes


def stable_evaluation_seed(checkpoint_seed, instance_id, method_name):
    payload = (
        f"{int(checkpoint_seed)}|{instance_id}|{method_name}".encode("utf-8")
    )
    return int.from_bytes(sha256(payload).digest()[:4], "big")


def load_frozen_policy(
    checkpoint_path,
    config,
    development_manifest_sha256,
    *,
    expected_source_sha256=None,
):
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "algorithm",
        "algorithm_core_version",
        "status",
        "training",
        "manifest_hash",
        "test_instances_used",
        "source_sha256",
        "feature_dim",
        "hidden_dim",
        "layers",
        "context_dim",
        "release_levels",
        "time_levels",
        "model_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"frozen checkpoint fields missing: {missing}")
    if payload["algorithm"] != "CHDGRL-LNS":
        raise ValueError("checkpoint algorithm mismatch")
    if payload["algorithm_core_version"] != DGRL_CORE_VERSION:
        raise ValueError("checkpoint DGRL core version mismatch")
    if payload["status"] != "frozen":
        raise ValueError("checkpoint is not frozen")
    if int(payload["test_instances_used"]) != 0:
        raise ValueError("checkpoint used test instances")
    if payload["manifest_hash"] != str(development_manifest_sha256):
        raise ValueError("checkpoint development manifest hash mismatch")
    if not str(payload["source_sha256"]).strip():
        raise ValueError("checkpoint source fingerprint is empty")
    if (
        expected_source_sha256 is not None
        and payload["source_sha256"] != str(expected_source_sha256)
    ):
        raise ValueError("checkpoint source fingerprint mismatch")
    expected_dimensions = {
        "hidden_dim": config.model.hidden_dim,
        "layers": config.model.layers,
        "context_dim": config.model.context_dim,
        "release_levels": len(config.search.release_profiles),
        "time_levels": len(config.search.time_budgets),
    }
    mismatches = {
        name: (int(payload[name]), int(expected))
        for name, expected in expected_dimensions.items()
        if int(payload[name]) != int(expected)
    }
    if mismatches:
        raise ValueError(f"checkpoint model dimension mismatch: {mismatches}")
    checkpoint_seed = int(payload["training"]["seed"])
    torch.manual_seed(checkpoint_seed)
    model = HypergraphDGRL(
        int(payload["feature_dim"]),
        int(payload["hidden_dim"]),
        int(payload["layers"]),
        int(payload["context_dim"]),
        int(payload["release_levels"]),
        int(payload["time_levels"]),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {key: value for key, value in payload.items() if key != "model_state"}
    metadata["checkpoint_seed"] = checkpoint_seed
    metadata["checkpoint_sha256"] = _file_hash(checkpoint_path)
    return model, metadata


def load_frozen_ordinary_policy(
    checkpoint_path,
    config,
    development_manifest_sha256,
    *,
    expected_source_sha256=None,
    expected_ablation_source_sha256=None,
):
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("algorithm") != "CHDGRL-LNS":
        raise ValueError("ordinary-graph checkpoint algorithm mismatch")
    if payload.get("algorithm_core_version") != DGRL_CORE_VERSION:
        raise ValueError("ordinary-graph DGRL core version mismatch")
    if payload.get("encoder_ablation_version") != ENCODER_ABLATION_VERSION:
        raise ValueError("ordinary-graph encoder version mismatch")
    if payload.get("model_kind") != "ordinary_graph_dgrl":
        raise ValueError("ordinary-graph checkpoint model kind mismatch")
    if payload.get("status") != "frozen":
        raise ValueError("ordinary-graph checkpoint is not frozen")
    if int(payload.get("test_instances_used", -1)) != 0:
        raise ValueError("ordinary-graph checkpoint used test instances")
    if payload.get("manifest_hash") != str(development_manifest_sha256):
        raise ValueError("ordinary-graph development manifest hash mismatch")
    if (
        expected_source_sha256 is not None
        and payload.get("source_sha256") != str(expected_source_sha256)
    ):
        raise ValueError("ordinary-graph base source fingerprint mismatch")
    if (
        expected_ablation_source_sha256 is not None
        and payload.get("ablation_source_sha256")
        != str(expected_ablation_source_sha256)
    ):
        raise ValueError("ordinary-graph ablation source fingerprint mismatch")
    expected_dimensions = {
        "hidden_dim": config.model.hidden_dim,
        "layers": config.model.layers,
        "context_dim": config.model.context_dim,
        "release_levels": len(config.search.release_profiles),
        "time_levels": len(config.search.time_budgets),
    }
    mismatches = {
        name: (int(payload.get(name, -1)), int(expected))
        for name, expected in expected_dimensions.items()
        if int(payload.get(name, -1)) != int(expected)
    }
    if mismatches:
        raise ValueError(f"ordinary-graph model dimension mismatch: {mismatches}")
    checkpoint_seed = int(payload["training"]["seed"])
    torch.manual_seed(checkpoint_seed)
    model = OrdinaryGraphDGRL(
        int(payload["feature_dim"]),
        int(payload["hidden_dim"]),
        int(payload["layers"]),
        int(payload["context_dim"]),
        int(payload["release_levels"]),
        int(payload["time_levels"]),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {key: value for key, value in payload.items() if key != "model_state"}
    metadata["checkpoint_seed"] = checkpoint_seed
    metadata["checkpoint_sha256"] = _file_hash(checkpoint_path)
    return model, metadata


class FixedBudgetDGRLController:
    def __init__(self, base_controller, release_index, time_index):
        self.base_controller = base_controller
        self.release_index = int(release_index)
        self.time_index = int(time_index)
        if not 0 <= self.release_index < len(base_controller.release_profiles):
            raise ValueError("fixed release index is outside the action space")
        if not 0 <= self.time_index < len(base_controller.time_budgets):
            raise ValueError("fixed time index is outside the action space")

    def _project(self, state, graph, action, remaining_budget_s, locked_tuple_ids, locked_arc_ids):
        return project_action(
            state,
            graph,
            action,
            self.base_controller.release_profiles,
            time_budgets=self.base_controller.time_budgets,
            remaining_budget_s=remaining_budget_s,
            locked_tuple_ids=locked_tuple_ids,
            locked_arc_ids=locked_arc_ids,
        )

    def select_action(
        self,
        *,
        state,
        graph,
        context,
        remaining_budget_s,
        locked_tuple_ids,
        locked_arc_ids,
        training=False,
    ):
        decision = self.base_controller.select_action(
            state=state,
            graph=graph,
            context=context,
            remaining_budget_s=remaining_budget_s,
            locked_tuple_ids=locked_tuple_ids,
            locked_arc_ids=locked_arc_ids,
            training=training,
        )
        if decision.action is not None:
            action = JointAction(
                decision.action.neighborhood_type,
                decision.action.seed_id,
                self.release_index,
                self.time_index,
            )
            projection = self._project(
                state,
                graph,
                action,
                remaining_budget_s,
                locked_tuple_ids,
                locked_arc_ids,
            )
            if projection.accepted:
                return replace(decision, action=action, projection=projection)

        fallback = DeterministicActionController(
            self.base_controller.release_profiles,
            self.base_controller.time_budgets,
            release_index=self.release_index,
            time_index=self.time_index,
        ).select_action(
            state=state,
            graph=graph,
            context=context,
            remaining_budget_s=remaining_budget_s,
            locked_tuple_ids=locked_tuple_ids,
            locked_arc_ids=locked_arc_ids,
            training=False,
        )
        return replace(fallback, used_fallback=True)


class DeterministicActionController:
    def __init__(
        self,
        release_profiles,
        time_budgets,
        *,
        release_index,
        time_index,
    ):
        self.release_profiles = tuple(release_profiles)
        self.time_budgets = tuple(float(value) for value in time_budgets)
        self.release_index = int(release_index)
        self.time_index = int(time_index)
        if not 0 <= self.release_index < len(self.release_profiles):
            raise ValueError("deterministic release index is outside the action space")
        if not 0 <= self.time_index < len(self.time_budgets):
            raise ValueError("deterministic time index is outside the action space")

    def select_action(
        self,
        *,
        state,
        graph,
        context,
        remaining_budget_s,
        locked_tuple_ids,
        locked_arc_ids,
        training=False,
    ):
        del context, training
        ranked_nodes = sorted(
            graph.nodes,
            key=lambda node: (
                node.node_type != "task",
                float(node.selected),
                -float(node.priority),
                node.node_id,
            ),
        )
        attempted = 0
        for node in ranked_nodes:
            neighborhood_type = _neighborhood_type(node.node_type)
            if neighborhood_type is None:
                continue
            attempted += 1
            action = JointAction(
                neighborhood_type,
                node.node_id,
                self.release_index,
                self.time_index,
            )
            projection = project_action(
                state,
                graph,
                action,
                self.release_profiles,
                time_budgets=self.time_budgets,
                remaining_budget_s=remaining_budget_s,
                locked_tuple_ids=locked_tuple_ids,
                locked_arc_ids=locked_arc_ids,
            )
            if projection.accepted:
                return DGRLDecision(action, projection, False, attempted, None)
        return DGRLDecision(None, None, True, attempted, None)


@dataclass(frozen=True)
class ControllerSearchSolver:
    method_family = "joint_action_search"

    model: object
    experiment_config: object
    evaluation_seed: int
    controller_kind: str = "learned"
    fixed_release_index: int | None = None
    fixed_time_index: int | None = None
    guard_count_override: int | None = None

    def _controller(self):
        config = self.experiment_config
        if self.controller_kind == "deterministic":
            return DeterministicActionController(
                config.search.release_profiles,
                config.search.time_budgets,
                release_index=int(self.fixed_release_index),
                time_index=int(self.fixed_time_index),
            )
        base = DGRLController(
            self.model,
            release_profiles=config.search.release_profiles,
            time_budgets=config.search.time_budgets,
            sample_count=config.search.controller_sample_count,
            radius=config.search.controller_radius,
            seed=int(self.evaluation_seed),
            exploration_temperature=config.trainer.sdn_exploration_temperature,
            guard_count=(
                config.search.controller_guard_count
                if self.guard_count_override is None
                else int(self.guard_count_override)
            ),
        )
        if self.controller_kind == "fixed_budget":
            return FixedBudgetDGRLController(
                base,
                int(self.fixed_release_index),
                int(self.fixed_time_index),
            )
        if self.controller_kind != "learned":
            raise ValueError(f"unknown controller kind: {self.controller_kind}")
        return base

    def solve(
        self,
        *,
        state,
        task_ids,
        miss_penalty,
        validation_state,
        operation_capacity,
        config,
    ):
        return run_search(
            state=state,
            task_ids=task_ids,
            miss_penalty=miss_penalty,
            validation_state=validation_state,
            operation_capacity=operation_capacity,
            config=config,
            controller=self._controller(),
        )


def build_chdgrl_methods(
    instance,
    checkpoint_path,
    config,
    development_manifest_sha256,
    budget_id,
    *,
    include_exact=True,
    expected_source_sha256=None,
    ordinary_checkpoint_path=None,
    expected_ablation_source_sha256=None,
):
    model, metadata = load_frozen_policy(
        checkpoint_path,
        config,
        development_manifest_sha256,
        expected_source_sha256=expected_source_sha256,
    )
    checkpoint_seed = int(metadata["checkpoint_seed"])
    checkpoint_hash = metadata["checkpoint_sha256"]
    fixed_release_index = len(config.search.release_profiles) // 2
    fixed_time_index = len(config.search.time_budgets) // 2

    def seed_for(name):
        return stable_evaluation_seed(checkpoint_seed, instance.instance_id, name)

    common = {
        "budget_id": str(budget_id),
        "evaluation_seed": checkpoint_seed,
    }
    learned = ControllerSearchSolver(
        model,
        config,
        seed_for("chdgrl_lns"),
        "learned",
    )
    methods = {
        "chdgrl_lns": MethodSpec(
            learned,
            checkpoint_sha256=checkpoint_hash,
            model_kind="role_aware_hypergraph_dgrl",
            **common,
        ),
        "chdgrl_lns_fixed_budget": MethodSpec(
            ControllerSearchSolver(
                model,
                config,
                seed_for("chdgrl_lns_fixed_budget"),
                "fixed_budget",
                fixed_release_index,
                fixed_time_index,
            ),
            checkpoint_sha256=checkpoint_hash,
            model_kind="fixed_budget_ablation",
            **common,
        ),
        "chdgrl_lns_no_guard": MethodSpec(
            ControllerSearchSolver(
                model,
                config,
                seed_for("chdgrl_lns_no_guard"),
                "learned",
                guard_count_override=0,
            ),
            checkpoint_sha256=checkpoint_hash,
            model_kind="no_feasible_guard_ablation",
            **common,
        ),
        "chdgrl_lns_no_stability": MethodSpec(
            ControllerSearchSolver(
                model,
                config,
                seed_for("chdgrl_lns_no_stability"),
                "learned",
            ),
            config_overrides=(("plan_change_weight", 0.0),),
            checkpoint_sha256=checkpoint_hash,
            model_kind="no_stability_ablation",
            **common,
        ),
        "deterministic_joint": MethodSpec(
            ControllerSearchSolver(
                None,
                config,
                seed_for("deterministic_joint"),
                "deterministic",
                fixed_release_index,
                fixed_time_index,
            ),
            shared_reference=True,
            model_kind="deterministic_joint_controller",
            **common,
        ),
        "manual_alns": MethodSpec(
            ManualAlnsBaseline(
                seed=seed_for("manual_alns"),
                max_iterations=config.search.max_iterations,
                projection_limit=max(1, config.search.max_iterations),
                time_limit_s=max(config.search.time_budgets),
            ),
            model_kind="manual_alns",
            **common,
        ),
        "risk_greedy": MethodSpec(
            RiskPriorityGreedyBaseline(
                time_limit_s=max(config.search.time_budgets),
                projection_limit=max(1, config.search.max_iterations),
            ),
            shared_reference=True,
            model_kind="risk_priority_greedy",
            **common,
        ),
    }
    if ordinary_checkpoint_path is not None:
        ordinary_model, ordinary_metadata = load_frozen_ordinary_policy(
            ordinary_checkpoint_path,
            config,
            development_manifest_sha256,
            expected_source_sha256=expected_source_sha256,
            expected_ablation_source_sha256=expected_ablation_source_sha256,
        )
        if int(ordinary_metadata["checkpoint_seed"]) != checkpoint_seed:
            raise ValueError("encoder checkpoint seed does not match policy seed")
        methods["chdgrl_lns_ordinary_graph"] = MethodSpec(
            ControllerSearchSolver(
                ordinary_model,
                config,
                seed_for("chdgrl_lns_ordinary_graph"),
                "learned",
            ),
            checkpoint_sha256=ordinary_metadata["checkpoint_sha256"],
            model_kind="ordinary_graph_dgrl",
            **common,
        )
    if include_exact and len(instance.task_ids) <= 12:
        methods["exact_milp"] = MethodSpec(
            ExactMilpBaseline(max_tasks=12, time_limit_s=5.0),
            shared_reference=True,
            model_kind="complete_lexicographic_milp",
            **common,
        )
    return dict(sorted(methods.items())), metadata


__all__ = [
    "ControllerSearchSolver",
    "DeterministicActionController",
    "EVALUATION_SOURCE_FILES",
    "FixedBudgetDGRLController",
    "build_chdgrl_methods",
    "evaluation_source_hashes",
    "load_frozen_policy",
    "load_frozen_ordinary_policy",
    "stable_evaluation_seed",
]
