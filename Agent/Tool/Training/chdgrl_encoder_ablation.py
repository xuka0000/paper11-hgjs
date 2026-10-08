from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean

import torch

from Agent.Proposed.dgrl import HypergraphDGRL
from Agent.Proposed.gnn import OrdinaryGraphGNN
from Agent.Proposed.hgnn import tensorize_graph
from Agent.Proposed.hypergraph import build_constraint_hypergraph
from Agent.Tool.Benchmark.formal_v3_loader import FormalV3InstanceLoader
from Agent.Tool.Training.chdgrl_config import load_chdgrl_experiment_config
from Agent.Tool.Training.chdgrl_experiment import (
    _file_hash,
    _plot_learning_curve,
    _save_last,
    _source_fingerprint,
    _validation_metrics,
    _write_csv,
    _write_json,
)
from Agent.Tool.Training.dgrl_training import DGRLReplayBuffer
from Agent.Tool.Training.dgrl_training_runner import (
    DGRL_CORE_VERSION,
    build_dgrl_trainer,
)


ENCODER_ABLATION_VERSION = "ordinary-graph-dgrl-v1"


class OrdinaryGraphDGRL(HypergraphDGRL):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        layers,
        context_dim,
        release_levels,
        time_levels,
    ):
        super().__init__(
            feature_dim,
            hidden_dim,
            layers,
            context_dim,
            release_levels,
            time_levels,
        )
        self.encoder = OrdinaryGraphGNN(feature_dim, hidden_dim, layers)


def _model_for_instance(instance, config, seed):
    graph = tensorize_graph(
        build_constraint_hypergraph(
            instance.search_state,
            instance.task_ids,
            instance.miss_penalty,
        )
    )
    torch.manual_seed(int(seed))
    return OrdinaryGraphDGRL(
        graph.feature_dim,
        config.model.hidden_dim,
        config.model.layers,
        config.model.context_dim,
        len(config.search.release_profiles),
        len(config.search.time_budgets),
    )


def _ablation_source_sha256():
    project_root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "Agent/Tool/Training/chdgrl_encoder_ablation.py",
        "Agent/Tool/Training/chdgrl_encoder_equal_update.py",
        "Agent/Proposed/gnn.py",
    )
    payload = {
        relative_path: sha256((project_root / relative_path).read_bytes()).hexdigest()
        for relative_path in relative_paths
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


def save_frozen_ordinary_graph_checkpoint(
    path,
    model,
    trainer_config,
    *,
    manifest_hash,
    train_instance_ids,
    validation_instance_ids,
    base_source_sha256,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "CHDGRL-LNS",
        "base_algorithm": "DGRL",
        "algorithm_core_version": DGRL_CORE_VERSION,
        "encoder_ablation_version": ENCODER_ABLATION_VERSION,
        "model_kind": "ordinary_graph_dgrl",
        "status": "frozen",
        "training": asdict(trainer_config),
        "manifest_hash": str(manifest_hash),
        "train_instance_ids": tuple(sorted(set(train_instance_ids))),
        "validation_instance_ids": tuple(sorted(set(validation_instance_ids))),
        "test_instances_used": 0,
        "source_sha256": str(base_source_sha256),
        "ablation_source_sha256": _ablation_source_sha256(),
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


def train_ordinary_graph_seed_grid(
    manifest_path,
    config_path,
    dataset_path,
    output_dir,
    *,
    resume=False,
):
    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    loader = FormalV3InstanceLoader(manifest_path)
    config = load_chdgrl_experiment_config(config_path)
    config_hash = _file_hash(config_path)
    dataset_hash = _file_hash(dataset_path)
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    _, current_source_sha256 = _source_fingerprint()
    if dataset.get("config_sha256") != config_hash:
        raise ValueError("training dataset configuration hash mismatch")
    if dataset.get("manifest_sha256") != loader.manifest_sha256:
        raise ValueError("training dataset manifest hash mismatch")
    if int(dataset.get("test_instances_used", -1)) != 0:
        raise ValueError("test instance entered encoder ablation training")
    if dataset.get("source_sha256") != current_source_sha256:
        raise ValueError("training dataset source hash mismatch")
    transitions = tuple(dataset["transitions"])
    train_ids = tuple(dataset["train_instance_ids"])
    validation_ids = tuple(
        instance_id
        for instance_id in loader.instance_ids
        if loader.split_for(instance_id) == "validation"
    )
    if not transitions or not train_ids or not validation_ids:
        raise ValueError("training transitions and development splits are required")

    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_paths = []
    for seed in config.training_seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        frozen_path = seed_dir / "frozen.pt"
        if frozen_path.exists() and resume:
            frozen_paths.append(str(frozen_path.resolve()))
            continue
        model = _model_for_instance(loader(train_ids[0]), config, seed)
        trainer_config = replace(config.trainer, seed=int(seed))
        trainer = build_dgrl_trainer(model, trainer_config)
        buffer = DGRLReplayBuffer(
            max(config.training.replay_capacity, len(transitions)),
            seed,
        )
        for transition in transitions:
            buffer.append(transition)

        resolved = json.loads(config_path.read_text(encoding="utf-8"))
        resolved.update(
            {
                "active_seed": int(seed),
                "model_kind": "ordinary_graph_dgrl",
                "encoder_ablation_version": ENCODER_ABLATION_VERSION,
                "config_sha256": config_hash,
                "manifest_sha256": loader.manifest_sha256,
                "dataset_sha256": dataset_hash,
                "train_instance_ids": list(train_ids),
                "validation_instance_ids": list(validation_ids),
                "test_instances_used": 0,
            }
        )
        _write_json(seed_dir / "resolved_config.json", resolved)

        start_epoch = 1
        rows = []
        best_state = None
        best_score = None
        best_epoch = 0
        stale_epochs = 0
        last_path = seed_dir / "last.pt"
        if resume and last_path.exists():
            checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
            if checkpoint.get("model_kind") != "ordinary_graph_dgrl":
                raise ValueError("resume checkpoint encoder kind mismatch")
            if checkpoint["config_sha256"] != config_hash:
                raise ValueError("resume configuration hash mismatch")
            if checkpoint["manifest_sha256"] != loader.manifest_sha256:
                raise ValueError("resume manifest hash mismatch")
            model.load_state_dict(checkpoint["model_state"])
            trainer.target_model.load_state_dict(checkpoint["target_model_state"])
            trainer.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state"])
            trainer.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state"])
            start_epoch = int(checkpoint["epoch"]) + 1
            rows = list(checkpoint["rows"])
            best_state = checkpoint["best_state"]
            best_score = tuple(checkpoint["best_score"])
            best_epoch = int(checkpoint["best_epoch"])
            stale_epochs = int(checkpoint["stale_epochs"])

        for epoch in range(start_epoch, config.training.epochs + 1):
            stats = tuple(
                trainer.train_step(buffer.sample(config.training.batch_size))
                for _ in range(config.training.updates_per_epoch)
            )
            metrics = _validation_metrics(
                loader,
                validation_ids,
                model,
                config,
                seed,
            )
            row = {
                "epoch": epoch,
                "actor_loss": fmean(item.actor_loss for item in stats),
                "critic_loss": fmean(item.critic_loss for item in stats),
                "replay_size": len(buffer),
                **metrics,
            }
            rows.append(row)
            score = (
                row["validation_primary_gain"],
                row["validation_secondary_gain"],
                row["validation_acceptance_rate"],
                -row["validation_fallback_rate"],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            _save_last(
                last_path,
                {
                    "model_kind": "ordinary_graph_dgrl",
                    "epoch": epoch,
                    "config_sha256": config_hash,
                    "manifest_sha256": loader.manifest_sha256,
                    "dataset_sha256": dataset_hash,
                    "model_state": model.state_dict(),
                    "target_model_state": trainer.target_model.state_dict(),
                    "actor_optimizer_state": trainer.actor_optimizer.state_dict(),
                    "critic_optimizer_state": trainer.critic_optimizer.state_dict(),
                    "rows": tuple(rows),
                    "best_state": best_state,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs,
                    "test_instances_used": 0,
                },
            )
            if stale_epochs >= config.training.patience:
                break

        if best_state is None:
            raise RuntimeError("encoder ablation training produced no model state")
        model.load_state_dict(best_state)
        save_frozen_ordinary_graph_checkpoint(
            frozen_path,
            model,
            trainer_config,
            manifest_hash=loader.manifest_sha256,
            train_instance_ids=train_ids,
            validation_instance_ids=validation_ids,
            base_source_sha256=current_source_sha256,
        )
        _write_csv(seed_dir / "epochs.csv", rows)
        _plot_learning_curve(seed_dir / "epochs.csv", seed_dir / "learning_curve.png")
        _write_json(
            seed_dir / "training_report.json",
            {
                "algorithm": "CHDGRL-LNS",
                "model_kind": "ordinary_graph_dgrl",
                "status": "frozen",
                "seed": int(seed),
                "epochs_completed": len(rows),
                "best_epoch": best_epoch,
                "best_score": list(best_score),
                "config_sha256": config_hash,
                "manifest_sha256": loader.manifest_sha256,
                "dataset_sha256": dataset_hash,
                "test_instances_used": 0,
            },
        )
        frozen_paths.append(str(frozen_path.resolve()))

    report = {
        "algorithm": "CHDGRL-LNS",
        "model_kind": "ordinary_graph_dgrl",
        "encoder_ablation_version": ENCODER_ABLATION_VERSION,
        "status": "encoder_ablation_frozen",
        "config_sha256": config_hash,
        "manifest_sha256": loader.manifest_sha256,
        "dataset_sha256": dataset_hash,
        "training_seeds": list(config.training_seeds),
        "frozen_checkpoints": frozen_paths,
        "test_instances_used": 0,
        "source_sha256": current_source_sha256,
        "ablation_source_sha256": _ablation_source_sha256(),
    }
    _write_json(output_dir / "encoder_freeze_report.json", report)
    return report


__all__ = [
    "ENCODER_ABLATION_VERSION",
    "OrdinaryGraphDGRL",
    "save_frozen_ordinary_graph_checkpoint",
    "train_ordinary_graph_seed_grid",
]
