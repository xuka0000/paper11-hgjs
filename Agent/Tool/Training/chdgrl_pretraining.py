import csv
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import fmean

import torch
from torch import nn
from torch.nn import functional

from Agent.Proposed.dgrl import HypergraphDGRL
from Agent.Proposed.dgrl_actions import JointAction
from Agent.Proposed.hgnn import tensorize_graph
from Agent.Proposed.hypergraph import build_constraint_hypergraph
from Agent.Proposed.search import _dgrl_context
from Agent.Tool.Benchmark.formal_v3_loader import FormalV3InstanceLoader
from Agent.Tool.Training.chdgrl_config import load_chdgrl_experiment_config
from Agent.Tool.Training.chdgrl_expert_labels import (
    compare_expert_action_labels,
    read_expert_action_bundles,
)
from Mask.ServiceTuple import SearchState


EXPERT_PRETRAINING_VERSION = "chdgrl-expert-joint-ranking-v1"
PRETRAINING_SOURCE_FILES = (
    "Agent/Proposed/dgrl.py",
    "Agent/Proposed/dgrl_actions.py",
    "Agent/Proposed/hgnn.py",
    "Agent/Proposed/hypergraph.py",
    "Agent/Proposed/search.py",
    "Agent/Tool/Benchmark/formal_v3_loader.py",
    "Agent/Tool/Training/chdgrl_config.py",
    "Agent/Tool/Training/chdgrl_expert_labels.py",
    "Agent/Tool/Training/chdgrl_pretraining.py",
    "Mask/ServiceTuple.py",
)


@dataclass(frozen=True)
class ExpertPretrainingConfig:
    seed: int
    epochs: int
    minimum_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    ranking_weight: float
    actor_weight: float
    projection_weight: float
    outcome_weight: float
    solver_cost_weight: float
    gradient_clip: float
    num_threads: int
    top_ks: tuple[int, ...]

    def __post_init__(self):
        if self.epochs <= 0 or self.patience <= 0:
            raise ValueError("pretraining epochs and patience must be positive")
        if not 0 <= self.minimum_epochs <= self.epochs:
            raise ValueError("minimum_epochs must be between zero and epochs")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("pretraining optimizer parameters are invalid")
        weights = (
            self.ranking_weight,
            self.actor_weight,
            self.projection_weight,
            self.outcome_weight,
            self.solver_cost_weight,
        )
        if any(value < 0 for value in weights) or not any(value > 0 for value in weights):
            raise ValueError("pretraining loss weights must be nonnegative and nonzero")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if not self.top_ks or any(value <= 0 for value in self.top_ks):
            raise ValueError("top_ks must contain positive values")
        if tuple(sorted(set(self.top_ks))) != self.top_ks:
            raise ValueError("top_ks must be unique and increasing")


@dataclass(frozen=True)
class ExpertRankingExample:
    instance_id: str
    split: str
    state_hash: str
    graph: object
    context: torch.Tensor
    actions: tuple[JointAction, ...]
    labels: tuple[object, ...]
    ordered_pair_indices: tuple[tuple[int, int], ...]
    oracle_index: int
    primary_relative_tolerance: float = 0.0


class ExpertAuxiliaryHeads(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.projection = nn.Linear(int(hidden_dim), 1)
        self.outcome = nn.Linear(int(hidden_dim), 1)
        self.solver_cost = nn.Linear(int(hidden_dim), 1)

    def forward(self, action_features):
        hidden = self.shared(action_features)
        return (
            self.projection(hidden).squeeze(-1),
            self.outcome(hidden).squeeze(-1),
            torch.sigmoid(self.solver_cost(hidden).squeeze(-1)),
        )


def load_expert_pretraining_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("pretraining_version") != EXPERT_PRETRAINING_VERSION:
        raise ValueError("expert pretraining configuration version mismatch")
    return ExpertPretrainingConfig(
        seed=int(payload["seed"]),
        epochs=int(payload["epochs"]),
        minimum_epochs=int(payload.get("minimum_epochs", 0)),
        patience=int(payload["patience"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload["weight_decay"]),
        ranking_weight=float(payload["loss_weights"]["ranking"]),
        actor_weight=float(payload["loss_weights"]["actor"]),
        projection_weight=float(payload["loss_weights"]["projection"]),
        outcome_weight=float(payload["loss_weights"]["outcome"]),
        solver_cost_weight=float(payload["loss_weights"]["solver_cost"]),
        gradient_clip=float(payload["gradient_clip"]),
        num_threads=int(payload.get("num_threads", 1)),
        top_ks=tuple(int(value) for value in payload["top_ks"]),
    )


def _file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _source_fingerprint():
    root = Path(__file__).resolve().parents[3]
    hashes = {name: _file_hash(root / name) for name in PRETRAINING_SOURCE_FILES}
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashes, sha256(encoded).hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path, rows):
    path = Path(path)
    rows = tuple(rows)
    if not rows:
        raise ValueError("pretraining history must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _action_id(action):
    return (
        f"{action.neighborhood_type}|{action.seed_id}|"
        f"r{action.release_profile}|t{action.time_budget}"
    )


def _state_hash(bundle):
    payload = (
        str(bundle.instance_hash),
        int(bundle.state_index),
        tuple(sorted(bundle.selected_arc_ids)),
        tuple(sorted(bundle.selected_tuple_ids)),
    )
    return sha256(repr(payload).encode("utf-8")).hexdigest()


def _best_label_index(labels, primary_relative_tolerance=0.0):
    if not labels:
        raise ValueError("an expert state must contain action labels")
    best = 0
    for index in range(1, len(labels)):
        comparison = compare_expert_action_labels(
            labels[index],
            labels[best],
            primary_relative_tolerance=primary_relative_tolerance,
        )
        if comparison > 0 or (
            comparison == 0 and labels[index].action_id < labels[best].action_id
        ):
            best = index
    return best


def _validate_label_artifact(
    labels_path,
    *,
    expected_split,
    loader,
    experiment_config_path,
):
    labels_path = Path(labels_path)
    report_path = labels_path.with_suffix(".report.json")
    if not labels_path.exists() or not report_path.exists():
        raise FileNotFoundError("expert labels and their report are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("dataset_kind") != "expert_joint_action_ranking":
        raise ValueError("expert label dataset kind mismatch")
    if report.get("output_sha256") != _file_hash(labels_path):
        raise ValueError("expert label output hash mismatch")
    if report.get("manifest_sha256") != loader.manifest_sha256:
        raise ValueError("expert label manifest hash mismatch")
    if report.get("config_sha256") != _file_hash(experiment_config_path):
        raise ValueError("expert label experiment configuration hash mismatch")
    if tuple(report.get("splits", ())) != (str(expected_split),):
        raise ValueError("expert label split mismatch")
    if int(report.get("test_instances_used", -1)) != 0:
        raise ValueError("test instances entered expert labels")
    return report


def prepare_expert_ranking_examples(
    manifest_path,
    experiment_config_path,
    labels_path,
    *,
    expected_split,
):
    if expected_split not in {"train", "validation"}:
        raise ValueError("expert pretraining accepts only train or validation splits")
    loader = FormalV3InstanceLoader(manifest_path)
    config = load_chdgrl_experiment_config(experiment_config_path)
    label_report = _validate_label_artifact(
        labels_path,
        expected_split=expected_split,
        loader=loader,
        experiment_config_path=experiment_config_path,
    )
    primary_relative_tolerance = float(
        label_report.get("primary_relative_tolerance", 0.0)
    )
    if primary_relative_tolerance < 0:
        raise ValueError("expert label primary relative tolerance is invalid")
    bundles = read_expert_action_bundles(labels_path)
    if not bundles:
        raise ValueError("expert label dataset is empty")
    if len({bundle.state_hash for bundle in bundles}) != len(bundles):
        raise ValueError("expert label dataset contains duplicate states")
    root_objectives = {}
    for bundle in sorted(bundles, key=lambda item: (item.instance_id, item.state_index)):
        root_objectives.setdefault(
            bundle.instance_id,
            (bundle.initial_primary, bundle.initial_secondary),
        )
    instance_cache = {}
    examples = []
    for bundle in bundles:
        if bundle.split != expected_split:
            raise ValueError("expert bundle is outside the requested split")
        if bundle.instance_id not in loader.instance_ids:
            raise ValueError("expert bundle instance is absent from the manifest")
        if loader.split_for(bundle.instance_id) != expected_split:
            raise ValueError("expert bundle conflicts with the manifest split")
        if bundle.instance_id not in instance_cache:
            instance_cache[bundle.instance_id] = loader(bundle.instance_id)
        instance = instance_cache[bundle.instance_id]
        if bundle.instance_hash != instance.instance_hash:
            raise ValueError("expert bundle instance hash mismatch")
        if bundle.state_hash != _state_hash(bundle):
            raise ValueError("expert bundle state hash mismatch")
        state = SearchState(
            instance.search_state.arcs,
            instance.search_state.tuples,
            frozenset(bundle.selected_arc_ids),
            frozenset(bundle.selected_tuple_ids),
        )
        graph = tensorize_graph(
            build_constraint_hypergraph(
                state,
                instance.task_ids,
                task_weights=instance.miss_penalty,
            )
        )
        actions = tuple(
            JointAction(
                label.neighborhood_type,
                label.seed_id,
                label.release_profile,
                label.time_budget,
            )
            for label in bundle.labels
        )
        if any(
            _action_id(action) != label.action_id
            for action, label in zip(actions, bundle.labels)
        ):
            raise ValueError("expert action identity mismatch")
        action_index = {label.action_id: index for index, label in enumerate(bundle.labels)}
        pairs = []
        for better, worse in bundle.ordered_pairs:
            if better not in action_index or worse not in action_index:
                raise ValueError("expert ordered pair references an unknown action")
            better_index = action_index[better]
            worse_index = action_index[worse]
            if (
                bundle.labels[better_index].time_budget
                != bundle.labels[worse_index].time_budget
            ):
                raise ValueError("expert ordered pair mixes repair time budgets")
            pairs.append((better_index, worse_index))
        root_primary, root_secondary = root_objectives[bundle.instance_id]
        max_iterations = max(1, int(config.search.max_iterations))
        elapsed_fraction = min(1.0, float(bundle.state_index) / max_iterations)
        context = _dgrl_context(
            initial_primary=root_primary,
            initial_secondary=root_secondary,
            best_primary=bundle.initial_primary,
            best_secondary=bundle.initial_secondary,
            iteration=bundle.state_index,
            max_iterations=max_iterations,
            no_improve=0,
            no_improve_limit=config.search.no_improve_limit,
            remaining_budget_s=(1.0 - elapsed_fraction)
            * float(config.search.total_time_limit_s),
            total_budget_s=config.search.total_time_limit_s,
            last_runtime_s=0.0,
            last_failure=False,
        )
        if context.numel() != config.model.context_dim:
            raise ValueError("expert context dimension does not match the model")
        examples.append(
            ExpertRankingExample(
                bundle.instance_id,
                bundle.split,
                bundle.state_hash,
                graph,
                context,
                actions,
                bundle.labels,
                tuple(pairs),
                _best_label_index(
                    bundle.labels,
                    primary_relative_tolerance,
                ),
                primary_relative_tolerance,
            )
        )
    return tuple(examples)


def build_expert_pretraining_model(examples, experiment_config, pretraining_config):
    examples = tuple(examples)
    if not examples:
        raise ValueError("expert pretraining examples are required")
    torch.manual_seed(int(pretraining_config.seed))
    model = HypergraphDGRL(
        examples[0].graph.feature_dim,
        experiment_config.model.hidden_dim,
        experiment_config.model.layers,
        experiment_config.model.context_dim,
        len(experiment_config.search.release_profiles),
        len(experiment_config.search.time_budgets),
    )
    auxiliary = ExpertAuxiliaryHeads(
        model.hidden_dim + model.action_dim,
        model.hidden_dim,
    )
    return model, auxiliary


def _forward_example(model, auxiliary, example):
    node_state, global_state = model.encode_state(example.graph, example.context)
    action_embeddings = model._action_embeddings_from_nodes(
        example.graph,
        node_state,
        example.actions,
    )
    action_features = torch.cat(
        (global_state.expand(action_embeddings.shape[0], -1), action_embeddings),
        dim=1,
    )
    q1 = model.critic1(action_features).squeeze(-1)
    q2 = model.critic2(action_features).squeeze(-1)
    projection_logits, outcome_logits, solver_cost = auxiliary(action_features)
    actor_vector = model.proto_action_from_state(global_state).vector
    return (
        q1,
        q2,
        projection_logits,
        outcome_logits,
        solver_cost,
        actor_vector,
        action_embeddings,
    )


def _positive_weight(examples, attribute):
    values = [
        bool(getattr(label, attribute))
        for example in examples
        for label in example.labels
    ]
    positive = sum(values)
    negative = len(values) - positive
    return float(negative / positive) if positive and negative else 1.0


def expert_pretraining_loss(
    model,
    auxiliary,
    example,
    config,
    *,
    projection_positive_weight=1.0,
    outcome_positive_weight=1.0,
    _outputs=None,
):
    (
        q1,
        q2,
        projection_logits,
        outcome_logits,
        solver_cost,
        actor_vector,
        action_embeddings,
    ) = _outputs if _outputs is not None else _forward_example(model, auxiliary, example)
    if example.ordered_pair_indices:
        better = torch.tensor(
            [pair[0] for pair in example.ordered_pair_indices],
            dtype=torch.long,
            device=q1.device,
        )
        worse = torch.tensor(
            [pair[1] for pair in example.ordered_pair_indices],
            dtype=torch.long,
            device=q1.device,
        )
        ranking = 0.5 * (
            functional.softplus(-(q1[better] - q1[worse])).mean()
            + functional.softplus(-(q2[better] - q2[worse])).mean()
        )
    else:
        ranking = q1.sum() * 0.0
    projection_target = projection_logits.new_tensor(
        [float(label.projection_feasible) for label in example.labels]
    )
    outcome_target = outcome_logits.new_tensor(
        [float(label.outcome_feasible) for label in example.labels]
    )
    projection = functional.binary_cross_entropy_with_logits(
        projection_logits,
        projection_target,
        pos_weight=projection_logits.new_tensor(float(projection_positive_weight)),
    )
    outcome = functional.binary_cross_entropy_with_logits(
        outcome_logits,
        outcome_target,
        pos_weight=outcome_logits.new_tensor(float(outcome_positive_weight)),
    )
    solver_mask = projection_target.bool()
    if bool(solver_mask.any()):
        solver_target = solver_cost.new_tensor(
            [solver_effort_target(label) for label in example.labels]
        )
        solver = functional.smooth_l1_loss(
            solver_cost[solver_mask],
            solver_target[solver_mask],
        )
    else:
        solver = solver_cost.sum() * 0.0
    oracle = example.labels[example.oracle_index]
    if oracle.projection_feasible:
        actor = functional.smooth_l1_loss(
            actor_vector,
            action_embeddings[example.oracle_index].detach(),
        )
    else:
        actor = actor_vector.sum() * 0.0
    total = (
        config.ranking_weight * ranking
        + config.actor_weight * actor
        + config.projection_weight * projection
        + config.outcome_weight * outcome
        + config.solver_cost_weight * solver
    )
    return {
        "total": total,
        "ranking": ranking,
        "actor": actor,
        "projection": projection,
        "outcome": outcome,
        "solver_cost": solver,
    }


def solver_effort_target(label):
    if label.repair_budget_fraction is not None:
        return min(1.0, max(0.0, float(label.repair_budget_fraction)))
    if label.repair_node_limit is not None:
        return 1.0
    return min(
        1.0,
        max(0.0, float(label.solver_runtime_s))
        / max(float(label.time_limit_s), 1e-9),
    )


def _actually_best_index(
    labels,
    indices,
    primary_relative_tolerance=0.0,
):
    indices = tuple(indices)
    best = indices[0]
    for index in indices[1:]:
        comparison = compare_expert_action_labels(
            labels[index],
            labels[best],
            primary_relative_tolerance=primary_relative_tolerance,
        )
        if comparison > 0 or (
            comparison == 0 and labels[index].action_id < labels[best].action_id
        ):
            best = index
    return best


def evaluate_expert_ranking(
    model,
    auxiliary,
    examples,
    config,
    *,
    projection_positive_weight=None,
    outcome_positive_weight=None,
):
    examples = tuple(examples)
    if not examples:
        raise ValueError("expert ranking evaluation requires examples")
    projection_positive_weight = (
        _positive_weight(examples, "projection_feasible")
        if projection_positive_weight is None
        else float(projection_positive_weight)
    )
    outcome_positive_weight = (
        _positive_weight(examples, "outcome_feasible")
        if outcome_positive_weight is None
        else float(outcome_positive_weight)
    )
    model.eval()
    auxiliary.eval()
    losses = []
    pair_correct = 0.0
    pair_count = 0
    projection_correct = 0
    outcome_correct = 0
    action_count = 0
    top_rows = {value: [] for value in config.top_ks}
    with torch.no_grad():
        for example in examples:
            outputs = _forward_example(model, auxiliary, example)
            components = expert_pretraining_loss(
                model,
                auxiliary,
                example,
                config,
                projection_positive_weight=projection_positive_weight,
                outcome_positive_weight=outcome_positive_weight,
                _outputs=outputs,
            )
            losses.append({key: float(value) for key, value in components.items()})
            (
                q1,
                q2,
                projection_logits,
                outcome_logits,
                _,
                _,
                _,
            ) = outputs
            scores = torch.minimum(q1, q2).cpu().tolist()
            ranked = sorted(
                range(len(scores)),
                key=lambda index: (-float(scores[index]), example.labels[index].action_id),
            )
            for better, worse in example.ordered_pair_indices:
                difference = float(scores[better] - scores[worse])
                pair_correct += 1.0 if difference > 0 else 0.5 if difference == 0 else 0.0
                pair_count += 1
            projection_prediction = projection_logits >= 0
            outcome_prediction = outcome_logits >= 0
            projection_target = torch.tensor(
                [label.projection_feasible for label in example.labels],
                dtype=torch.bool,
            )
            outcome_target = torch.tensor(
                [label.outcome_feasible for label in example.labels],
                dtype=torch.bool,
            )
            projection_correct += int((projection_prediction.cpu() == projection_target).sum())
            outcome_correct += int((outcome_prediction.cpu() == outcome_target).sum())
            action_count += len(example.labels)
            oracle = example.labels[example.oracle_index]
            for top_k in config.top_ks:
                candidates = ranked[: min(int(top_k), len(ranked))]
                selected_index = _actually_best_index(
                    example.labels,
                    candidates,
                    example.primary_relative_tolerance,
                )
                selected = example.labels[selected_index]
                better_count = sum(
                    compare_expert_action_labels(
                        label,
                        selected,
                        primary_relative_tolerance=(
                            example.primary_relative_tolerance
                        ),
                    )
                    > 0
                    for label in example.labels
                )
                denominator = max(1, len(example.labels) - 1)
                primary_regret = max(
                    0.0,
                    float(oracle.primary_gain - selected.primary_gain),
                )
                primary_equivalence_tolerance = 1e-8 + float(
                    example.primary_relative_tolerance
                ) * max(
                    1.0,
                    abs(float(oracle.primary_gain)),
                    abs(float(selected.primary_gain)),
                )
                secondary_regret = (
                    max(0.0, float(oracle.secondary_gain - selected.secondary_gain))
                    if abs(
                        float(oracle.primary_gain - selected.primary_gain)
                    )
                    <= primary_equivalence_tolerance
                    else 0.0
                )
                top_rows[top_k].append(
                    {
                        "rank_regret": better_count / denominator,
                        "primary_regret": primary_regret,
                        "secondary_regret": secondary_regret,
                        "oracle_recall": float(
                            compare_expert_action_labels(
                                selected,
                                oracle,
                                primary_relative_tolerance=(
                                    example.primary_relative_tolerance
                                ),
                            )
                            == 0
                        ),
                        "projection_feasible_hit": float(
                            selected.projection_feasible
                        ),
                        "outcome_feasible_hit": float(selected.outcome_feasible),
                        "strict_improvement_hit": float(selected.strict_improvement),
                    }
                )
    metrics = {
        f"loss_{name}": fmean(row[name] for row in losses)
        for name in losses[0]
    }
    metrics.update(
        {
            "pair_accuracy": pair_correct / max(1, pair_count),
            "pair_count": pair_count,
            "projection_accuracy": projection_correct / max(1, action_count),
            "outcome_accuracy": outcome_correct / max(1, action_count),
            "state_count": len(examples),
            "action_count": action_count,
        }
    )
    for top_k, rows in top_rows.items():
        for name in rows[0]:
            metrics[f"top_{top_k}_{name}"] = fmean(row[name] for row in rows)
    return metrics


def _flatten_epoch_row(epoch, train_metrics, validation_metrics):
    return {
        "epoch": int(epoch),
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
    }


def _save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _pretraining_contract(
    *,
    manifest_path,
    experiment_config_path,
    pretraining_config_path,
    train_labels_path,
    validation_labels_path,
    loader,
    source_sha256,
):
    train_label_report = json.loads(
        Path(train_labels_path)
        .with_suffix(".report.json")
        .read_text(encoding="utf-8")
    )
    validation_label_report = json.loads(
        Path(validation_labels_path)
        .with_suffix(".report.json")
        .read_text(encoding="utf-8")
    )
    train_tolerance = float(
        train_label_report.get("primary_relative_tolerance", 0.0)
    )
    validation_tolerance = float(
        validation_label_report.get("primary_relative_tolerance", 0.0)
    )
    if train_tolerance != validation_tolerance:
        raise ValueError("expert label primary relative tolerances differ")
    return {
        "pretraining_version": EXPERT_PRETRAINING_VERSION,
        "manifest_sha256": loader.manifest_sha256,
        "experiment_config_sha256": _file_hash(experiment_config_path),
        "pretraining_config_sha256": _file_hash(pretraining_config_path),
        "train_labels_sha256": _file_hash(train_labels_path),
        "validation_labels_sha256": _file_hash(validation_labels_path),
        "source_sha256": source_sha256,
        "primary_relative_tolerance": train_tolerance,
        "test_instances_used": 0,
    }


def _validate_contract(payload, contract):
    for key, value in contract.items():
        if payload.get(key) != value:
            raise ValueError(f"expert pretraining resume {key} mismatch")


def train_expert_action_ranker(
    manifest_path,
    experiment_config_path,
    pretraining_config_path,
    train_labels_path,
    validation_labels_path,
    output_dir,
    *,
    resume=False,
):
    manifest_path = Path(manifest_path)
    experiment_config_path = Path(experiment_config_path)
    pretraining_config_path = Path(pretraining_config_path)
    train_labels_path = Path(train_labels_path)
    validation_labels_path = Path(validation_labels_path)
    output_dir = Path(output_dir)
    pretrained_path = output_dir / "pretrained.pt"
    last_path = output_dir / "last.pt"
    report_path = output_dir / "pretraining_report.json"
    history_path = output_dir / "epochs.csv"
    if pretrained_path.exists() and not resume:
        raise FileExistsError(pretrained_path)
    loader = FormalV3InstanceLoader(manifest_path)
    experiment_config = load_chdgrl_experiment_config(experiment_config_path)
    pretraining_config = load_expert_pretraining_config(pretraining_config_path)
    previous_num_threads = torch.get_num_threads()
    torch.set_num_threads(pretraining_config.num_threads)
    source_hashes, source_sha256 = _source_fingerprint()
    contract = _pretraining_contract(
        manifest_path=manifest_path,
        experiment_config_path=experiment_config_path,
        pretraining_config_path=pretraining_config_path,
        train_labels_path=train_labels_path,
        validation_labels_path=validation_labels_path,
        loader=loader,
        source_sha256=source_sha256,
    )
    if resume and pretrained_path.exists() and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _validate_contract(report, contract)
        if report.get("status") == "pretraining_complete":
            torch.set_num_threads(previous_num_threads)
            return report
    train_examples = prepare_expert_ranking_examples(
        manifest_path,
        experiment_config_path,
        train_labels_path,
        expected_split="train",
    )
    validation_examples = prepare_expert_ranking_examples(
        manifest_path,
        experiment_config_path,
        validation_labels_path,
        expected_split="validation",
    )
    train_states = {example.state_hash for example in train_examples}
    validation_states = {example.state_hash for example in validation_examples}
    if train_states & validation_states:
        raise ValueError("expert training and validation state overlap")
    train_instances = {example.instance_id for example in train_examples}
    validation_instances = {example.instance_id for example in validation_examples}
    if train_instances & validation_instances:
        raise ValueError("expert training and validation instance overlap")
    model, auxiliary = build_expert_pretraining_model(
        train_examples,
        experiment_config,
        pretraining_config,
    )
    optimizer = torch.optim.AdamW(
        tuple(model.parameters()) + tuple(auxiliary.parameters()),
        lr=pretraining_config.learning_rate,
        weight_decay=pretraining_config.weight_decay,
    )
    projection_positive_weight = _positive_weight(
        train_examples,
        "projection_feasible",
    )
    outcome_positive_weight = _positive_weight(train_examples, "outcome_feasible")
    rows = []
    start_epoch = 1
    best_state = None
    best_auxiliary_state = None
    best_score = None
    best_epoch = 0
    stale_epochs = 0
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        _validate_contract(checkpoint, contract)
        model.load_state_dict(checkpoint["model_state"])
        auxiliary.load_state_dict(checkpoint["auxiliary_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        rows = list(checkpoint["rows"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_state = checkpoint["best_state"]
        best_auxiliary_state = checkpoint["best_auxiliary_state"]
        best_score = tuple(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
    if not rows:
        rows.append(
            _flatten_epoch_row(
                0,
                evaluate_expert_ranking(
                    model,
                    auxiliary,
                    train_examples,
                    pretraining_config,
                    projection_positive_weight=projection_positive_weight,
                    outcome_positive_weight=outcome_positive_weight,
                ),
                evaluate_expert_ranking(
                    model,
                    auxiliary,
                    validation_examples,
                    pretraining_config,
                    projection_positive_weight=projection_positive_weight,
                    outcome_positive_weight=outcome_positive_weight,
                ),
            )
        )
        baseline = rows[0]
        best_score = (
            -baseline["validation_top_1_rank_regret"],
            baseline["validation_pair_accuracy"],
            -baseline["validation_top_1_primary_regret"],
            baseline["validation_top_3_oracle_recall"],
        )
        best_state = deepcopy(model.state_dict())
        best_auxiliary_state = deepcopy(auxiliary.state_dict())
        best_epoch = 0
    for epoch in range(start_epoch, pretraining_config.epochs + 1):
        order = list(range(len(train_examples)))
        Random(pretraining_config.seed + epoch).shuffle(order)
        model.train()
        auxiliary.train()
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            components = expert_pretraining_loss(
                model,
                auxiliary,
                train_examples[index],
                pretraining_config,
                projection_positive_weight=projection_positive_weight,
                outcome_positive_weight=outcome_positive_weight,
            )
            components["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                tuple(model.parameters()) + tuple(auxiliary.parameters()),
                pretraining_config.gradient_clip,
            )
            optimizer.step()
        train_metrics = evaluate_expert_ranking(
            model,
            auxiliary,
            train_examples,
            pretraining_config,
            projection_positive_weight=projection_positive_weight,
            outcome_positive_weight=outcome_positive_weight,
        )
        validation_metrics = evaluate_expert_ranking(
            model,
            auxiliary,
            validation_examples,
            pretraining_config,
            projection_positive_weight=projection_positive_weight,
            outcome_positive_weight=outcome_positive_weight,
        )
        row = _flatten_epoch_row(epoch, train_metrics, validation_metrics)
        rows.append(row)
        score = (
            -row["validation_top_1_rank_regret"],
            row["validation_pair_accuracy"],
            -row["validation_top_1_primary_regret"],
            row["validation_top_3_oracle_recall"],
        )
        if best_score is None or score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_auxiliary_state = deepcopy(auxiliary.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        _save_checkpoint(
            last_path,
            {
                **contract,
                "status": "pretraining_in_progress",
                "epoch": epoch,
                "model_state": model.state_dict(),
                "auxiliary_state": auxiliary.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "rows": tuple(rows),
                "best_state": best_state,
                "best_auxiliary_state": best_auxiliary_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
            },
        )
        _write_csv(history_path, rows)
        if (
            epoch >= pretraining_config.minimum_epochs
            and stale_epochs >= pretraining_config.patience
        ):
            break
    if best_state is None or best_auxiliary_state is None:
        raise RuntimeError("expert pretraining produced no selectable model")
    model.load_state_dict(best_state)
    auxiliary.load_state_dict(best_auxiliary_state)
    best_row = next(row for row in rows if int(row["epoch"]) == best_epoch)
    checkpoint_payload = {
        **contract,
        "algorithm": "CHDGRL-LNS",
        "status": "pretrained",
        "stage": "expert_joint_action_ranking",
        "seed": int(pretraining_config.seed),
        "best_epoch": int(best_epoch),
        "best_score": list(best_score),
        "feature_dim": model.encoder.input_projection.in_features,
        "hidden_dim": model.hidden_dim,
        "layers": len(model.encoder.layers),
        "context_dim": model.context_projection.in_features,
        "release_levels": model.release_levels,
        "time_levels": model.time_levels,
        "train_instance_ids": tuple(sorted(train_instances)),
        "validation_instance_ids": tuple(sorted(validation_instances)),
        "train_state_hashes": tuple(sorted(train_states)),
        "validation_state_hashes": tuple(sorted(validation_states)),
        "model_state": model.state_dict(),
        "auxiliary_state": auxiliary.state_dict(),
    }
    _save_checkpoint(pretrained_path, checkpoint_payload)
    report = {
        **contract,
        "algorithm": "CHDGRL-LNS",
        "status": "pretraining_complete",
        "stage": "expert_joint_action_ranking",
        "seed": int(pretraining_config.seed),
        "epochs_completed": len(rows) - 1,
        "best_epoch": int(best_epoch),
        "best_score": list(best_score),
        "train_instance_count": len(train_instances),
        "validation_instance_count": len(validation_instances),
        "train_state_count": len(train_examples),
        "validation_state_count": len(validation_examples),
        "checkpoint_sha256": _file_hash(pretrained_path),
        "source_hashes": source_hashes,
        "best_metrics": best_row,
    }
    _write_json(report_path, report)
    torch.set_num_threads(previous_num_threads)
    return report


def load_expert_pretrained_checkpoint(path, model, *, freeze=False):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("algorithm") != "CHDGRL-LNS":
        raise ValueError("expert pretrained checkpoint algorithm mismatch")
    if payload.get("pretraining_version") != EXPERT_PRETRAINING_VERSION:
        raise ValueError("expert pretrained checkpoint version mismatch")
    if payload.get("status") != "pretrained":
        raise ValueError("expert pretrained checkpoint status mismatch")
    if int(payload.get("test_instances_used", -1)) != 0:
        raise ValueError("expert pretrained checkpoint used test instances")
    architecture = {
        "feature_dim": model.encoder.input_projection.in_features,
        "hidden_dim": model.hidden_dim,
        "layers": len(model.encoder.layers),
        "context_dim": model.context_projection.in_features,
        "release_levels": model.release_levels,
        "time_levels": model.time_levels,
    }
    if any(int(payload.get(key, -1)) != int(value) for key, value in architecture.items()):
        raise ValueError("expert pretrained checkpoint architecture mismatch")
    model.load_state_dict(payload["model_state"])
    model.eval() if freeze else model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(not freeze)
    excluded = {"model_state", "auxiliary_state"}
    return {key: value for key, value in payload.items() if key not in excluded}


__all__ = [
    "EXPERT_PRETRAINING_VERSION",
    "ExpertAuxiliaryHeads",
    "ExpertPretrainingConfig",
    "ExpertRankingExample",
    "build_expert_pretraining_model",
    "evaluate_expert_ranking",
    "expert_pretraining_loss",
    "load_expert_pretrained_checkpoint",
    "load_expert_pretraining_config",
    "prepare_expert_ranking_examples",
    "solver_effort_target",
    "train_expert_action_ranker",
]
