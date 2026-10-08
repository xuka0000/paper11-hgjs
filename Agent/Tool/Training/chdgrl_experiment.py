import csv
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean

import torch

from Agent.Proposed.dgrl import HypergraphDGRL
from Agent.Proposed.hgnn import tensorize_graph
from Agent.Proposed.hypergraph import build_constraint_hypergraph
from Agent.Tool.Benchmark.formal_v3_loader import FormalV3InstanceLoader
from Agent.Tool.Training.chdgrl_config import load_chdgrl_experiment_config
from Agent.Tool.Training.chdgrl_pretraining import (
    load_expert_pretrained_checkpoint,
)
from Agent.Tool.Training.dgrl_episode import (
    collect_dgrl_episode,
    evaluate_dgrl_policy,
)
from Agent.Tool.Training.dgrl_training import (
    DGRLReplayBuffer,
    encode_lexicographic_reward,
)
from Agent.Tool.Training.dgrl_training_runner import (
    build_dgrl_trainer,
    save_frozen_dgrl_checkpoint,
)


SOURCE_FILES = (
    "Env/IrregularDualNetworkFactory.py",
    "Agent/Proposed/closure.py",
    "Agent/Proposed/dgrl.py",
    "Agent/Proposed/dgrl_actions.py",
    "Agent/Proposed/dgrl_controller.py",
    "Agent/Proposed/hgnn.py",
    "Agent/Proposed/hypergraph.py",
    "Agent/Proposed/milp.py",
    "Agent/Proposed/regions.py",
    "Agent/Proposed/repair.py",
    "Agent/Proposed/search.py",
    "Agent/Proposed/validator.py",
    "Agent/Tool/Benchmark/benchmark_v3.py",
    "Agent/Tool/Training/chdgrl_config.py",
    "Agent/Tool/Training/chdgrl_expert_labels.py",
    "Agent/Tool/Training/chdgrl_experiment.py",
    "Agent/Tool/Training/dgrl_episode.py",
    "Agent/Tool/Training/dgrl_training.py",
    "Agent/Tool/Training/dgrl_training_runner.py",
    "Mask/Geometry.py",
    "Mask/PowerNetwork.py",
    "Mask/Road.py",
    "Mask/Scenario.py",
    "Mask/ServiceTuple.py",
    "Mask/TimeNetwork.py",
    "Mask/TupleFactory.py",
    "Mask/UAV.py",
    "Mask/Vehicle.py",
    "Mask/Weather.py",
)


def _file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _source_fingerprint():
    root = Path(__file__).resolve().parents[3]
    hashes = {name: _file_hash(root / name) for name in SOURCE_FILES}
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
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = tuple(rows)
    if not rows:
        raise ValueError("CSV rows must not be empty")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _model_for_instance(instance, config, seed):
    graph = tensorize_graph(
        build_constraint_hypergraph(
            instance.search_state,
            instance.task_ids,
            instance.miss_penalty,
        )
    )
    torch.manual_seed(int(seed))
    return HypergraphDGRL(
        graph.feature_dim,
        config.model.hidden_dim,
        config.model.layers,
        config.model.context_dim,
        len(config.search.release_profiles),
        len(config.search.time_budgets),
    )


def generate_dgrl_training_dataset(
    manifest_path,
    config_path,
    output_path,
    *,
    resume=False,
):
    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    output_path = Path(output_path)
    loader = FormalV3InstanceLoader(manifest_path)
    config = load_chdgrl_experiment_config(config_path)
    config_hash = _file_hash(config_path)
    source_hashes, source_sha256 = _source_fingerprint()
    report_path = output_path.with_suffix(".report.json")
    progress_path = output_path.with_suffix(".progress.json")
    if output_path.exists():
        if not resume:
            raise FileExistsError(output_path)
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        if payload.get("config_sha256") != config_hash:
            raise ValueError("existing training dataset configuration hash mismatch")
        if payload.get("manifest_sha256") != loader.manifest_sha256:
            raise ValueError("existing training dataset manifest hash mismatch")
        if int(payload.get("test_instances_used", -1)) != 0:
            raise ValueError("test instance entered existing training dataset")
        if payload.get("source_sha256") != source_sha256:
            raise ValueError("existing training dataset source hash mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not progress_path.exists():
            _write_json(
                progress_path,
                {
                    "algorithm": "CHDGRL-LNS",
                    "status": "complete",
                    "config_sha256": config_hash,
                    "manifest_sha256": loader.manifest_sha256,
                    "completed_instance_ids": list(payload["train_instance_ids"]),
                    "expected_instance_count": len(payload["train_instance_ids"]),
                    "trajectory_count": int(
                        report.get(
                            "trajectory_count",
                            len(payload["train_instance_ids"]),
                        )
                    ),
                    "transition_count": len(payload["transitions"]),
                    "dataset_sha256": report["dataset_sha256"],
                    "test_instances_used": 0,
                    "source_sha256": source_sha256,
                },
            )
        return report
    train_ids = tuple(
        item for item in loader.instance_ids if loader.split_for(item) == "train"
    )
    if not train_ids:
        raise ValueError("development manifest contains no training instances")
    if any(loader.split_for(item) == "test" for item in train_ids):
        raise ValueError("test instance entered training dataset")
    model = _model_for_instance(loader(train_ids[0]), config, config.behavior_seed)
    transitions = []
    rows = []
    parts_dir = output_path.with_suffix(".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    completed_trajectories = []
    trajectory_count = len(train_ids) * config.training.trajectories_per_instance
    for index, instance_id in enumerate(train_ids):
        instance_key = sha256(instance_id.encode("utf-8")).hexdigest()[:20]
        for replica in range(config.training.trajectories_per_instance):
            trajectory_index = (
                index * config.training.trajectories_per_instance + replica
            )
            trajectory_key = f"{instance_id}#{replica}"
            part_path = parts_dir / f"{instance_key}_{replica}.pt"
            if resume and part_path.exists():
                part = torch.load(part_path, map_location="cpu", weights_only=False)
                if part.get("instance_id") != instance_id:
                    raise ValueError("training dataset part instance mismatch")
                if int(part.get("replica", -1)) != replica:
                    raise ValueError("training dataset part replica mismatch")
                if part.get("config_sha256") != config_hash:
                    raise ValueError("training dataset part configuration mismatch")
                if part.get("manifest_sha256") != loader.manifest_sha256:
                    raise ValueError("training dataset part manifest mismatch")
                if part.get("source_sha256") != source_sha256:
                    raise ValueError("training dataset part source hash mismatch")
            else:
                episode = collect_dgrl_episode(
                    loader(instance_id),
                    model,
                    config,
                    seed=config.behavior_seed + trajectory_index,
                    training=True,
                )
                part = {
                    "instance_id": instance_id,
                    "replica": replica,
                    "config_sha256": config_hash,
                    "manifest_sha256": loader.manifest_sha256,
                    "transitions": episode.transitions,
                    "rows": tuple(
                        {"trajectory_index": trajectory_index, **row}
                        for row in episode.step_rows
                    ),
                    "test_instances_used": 0,
                    "source_sha256": source_sha256,
                }
                temporary_part = part_path.with_suffix(".pt.tmp")
                torch.save(part, temporary_part)
                temporary_part.replace(part_path)
            transitions.extend(part["transitions"])
            rows.extend(part["rows"])
            completed_trajectories.append(trajectory_key)
            _write_json(
                progress_path,
                {
                    "algorithm": "CHDGRL-LNS",
                    "status": "collecting",
                    "config_sha256": config_hash,
                    "manifest_sha256": loader.manifest_sha256,
                    "completed_instance_ids": completed,
                    "completed_trajectory_keys": completed_trajectories,
                    "expected_instance_count": len(train_ids),
                    "expected_trajectory_count": trajectory_count,
                    "transition_count": len(transitions),
                    "test_instances_used": 0,
                    "source_sha256": source_sha256,
                },
            )
        completed.append(instance_id)
    if not transitions:
        raise RuntimeError("training instances produced no DGRL transitions")
    for transition in transitions:
        encode_lexicographic_reward(transition.reward, config.reward)

    payload = {
        "algorithm": "CHDGRL-LNS",
        "status": "training_only",
        "config_sha256": config_hash,
        "manifest_sha256": loader.manifest_sha256,
        "behavior_seed": config.behavior_seed,
        "train_instance_ids": train_ids,
        "validation_instance_ids": (),
        "test_instances_used": 0,
        "source_hashes": source_hashes,
        "source_sha256": source_sha256,
        "transitions": tuple(transitions),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    _write_csv(output_path.with_suffix(".csv"), rows)
    report = {
        "algorithm": "CHDGRL-LNS",
        "status": "training_dataset_complete",
        "config_sha256": payload["config_sha256"],
        "manifest_sha256": payload["manifest_sha256"],
        "dataset_sha256": _file_hash(output_path),
        "train_instance_count": len(train_ids),
        "trajectory_count": trajectory_count,
        "transition_count": len(transitions),
        "test_instances_used": 0,
        "source_sha256": source_sha256,
    }
    _write_json(report_path, report)
    _write_json(
        progress_path,
        {
            "algorithm": "CHDGRL-LNS",
            "status": "complete",
            "config_sha256": config_hash,
            "manifest_sha256": loader.manifest_sha256,
            "completed_instance_ids": list(train_ids),
            "completed_trajectory_keys": completed_trajectories,
            "expected_instance_count": len(train_ids),
            "expected_trajectory_count": trajectory_count,
            "transition_count": len(transitions),
            "dataset_sha256": report["dataset_sha256"],
            "test_instances_used": 0,
            "source_sha256": source_sha256,
        },
    )
    return report


def _stable_seed(seed, instance_id):
    payload = f"{int(seed)}|{instance_id}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _validation_metrics(loader, validation_ids, model, config, seed):
    evaluation_model = deepcopy(model)
    results = tuple(
        evaluate_dgrl_policy(
            loader(instance_id),
            evaluation_model,
            config,
            seed=_stable_seed(seed, instance_id),
        )
        for instance_id in validation_ids
    )
    if not results:
        raise ValueError("development manifest contains no validation instances")
    return {
        "validation_primary_gain": fmean(
            item.initial_primary - item.best_primary for item in results
        ),
        "validation_secondary_gain": fmean(
            item.initial_secondary - item.best_secondary for item in results
        ),
        "validation_acceptance_rate": fmean(
            float(bool(item.accepted_records)) for item in results
        ),
        "validation_fallback_rate": fmean(
            fmean(float(record.used_fallback) for record in item.trace)
            if item.trace
            else 1.0
            for item in results
        ),
    }


def _save_last(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _plot_learning_curve(csv_path, output_path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    actor = [float(row["actor_loss"]) for row in rows]
    critic = [float(row["critic_loss"]) for row in rows]
    primary = [float(row["validation_primary_gain"]) for row in rows]
    secondary = [float(row["validation_secondary_gain"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    axes[0].plot(epochs, actor, label="Actor loss", linewidth=1.8)
    axes[0].plot(epochs, critic, label="Critic loss", linewidth=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, primary, label="Primary gain", linewidth=1.8)
    axes[1].plot(epochs, secondary, label="Secondary gain", linewidth=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation gain")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(alpha=0.2, linewidth=0.6)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train_dgrl_seed_grid(
    manifest_path,
    config_path,
    dataset_path,
    output_dir,
    *,
    resume=False,
    initialization_checkpoint=None,
):
    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    loader = FormalV3InstanceLoader(manifest_path)
    config = load_chdgrl_experiment_config(config_path)
    config_hash = _file_hash(config_path)
    dataset_hash = _file_hash(dataset_path)
    initialization_path = (
        Path(initialization_checkpoint) if initialization_checkpoint is not None else None
    )
    initialization_sha256 = (
        _file_hash(initialization_path) if initialization_path is not None else ""
    )
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    _, current_source_sha256 = _source_fingerprint()
    if dataset.get("config_sha256") != config_hash:
        raise ValueError("training dataset configuration hash mismatch")
    if dataset.get("manifest_sha256") != loader.manifest_sha256:
        raise ValueError("training dataset manifest hash mismatch")
    if int(dataset.get("test_instances_used", -1)) != 0:
        raise ValueError("test instance entered training dataset")
    if dataset.get("source_sha256") != current_source_sha256:
        raise ValueError("training dataset source hash mismatch")
    transitions = tuple(dataset["transitions"])
    train_ids = tuple(dataset["train_instance_ids"])
    validation_ids = tuple(
        item
        for item in loader.instance_ids
        if loader.split_for(item) == "validation"
    )
    if not transitions or not validation_ids:
        raise ValueError("training transitions and validation instances are required")

    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_paths = []
    for seed in config.training_seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        frozen_path = seed_dir / "frozen.pt"
        if frozen_path.exists() and resume:
            frozen_payload = torch.load(
                frozen_path,
                map_location="cpu",
                weights_only=False,
            )
            frozen_initialization = frozen_payload.get("initialization", {})
            if (
                frozen_initialization.get("checkpoint_sha256", "")
                != initialization_sha256
            ):
                raise ValueError("frozen initialization checkpoint hash mismatch")
            frozen_paths.append(str(frozen_path.resolve()))
            continue
        model = _model_for_instance(loader(train_ids[0]), config, seed)
        initialization = {}
        if initialization_path is not None:
            metadata = load_expert_pretrained_checkpoint(
                initialization_path,
                model,
                freeze=False,
            )
            if metadata.get("manifest_sha256") != loader.manifest_sha256:
                raise ValueError("pretrained initialization manifest hash mismatch")
            if metadata.get("experiment_config_sha256") != config_hash:
                raise ValueError("pretrained initialization configuration hash mismatch")
            initialization = {
                "stage": metadata.get("stage"),
                "pretraining_version": metadata.get("pretraining_version"),
                "checkpoint_sha256": initialization_sha256,
                "train_labels_sha256": metadata.get("train_labels_sha256"),
                "validation_labels_sha256": metadata.get(
                    "validation_labels_sha256"
                ),
                "best_epoch": metadata.get("best_epoch"),
            }
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
                "config_sha256": config_hash,
                "manifest_sha256": loader.manifest_sha256,
                "dataset_sha256": dataset_hash,
                "train_instance_ids": list(train_ids),
                "validation_instance_ids": list(validation_ids),
                "test_instances_used": 0,
                "initialization": initialization,
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
            if checkpoint["config_sha256"] != config_hash:
                raise ValueError("resume configuration hash mismatch")
            if checkpoint["manifest_sha256"] != loader.manifest_sha256:
                raise ValueError("resume manifest hash mismatch")
            if checkpoint.get("initialization_sha256", "") != initialization_sha256:
                raise ValueError("resume initialization checkpoint hash mismatch")
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
                    "initialization_sha256": initialization_sha256,
                },
            )
            if stale_epochs >= config.training.patience:
                break

        if best_state is None:
            raise RuntimeError("training produced no selectable model state")
        model.load_state_dict(best_state)
        save_frozen_dgrl_checkpoint(
            frozen_path,
            model,
            trainer_config,
            manifest_hash=loader.manifest_sha256,
            train_instance_ids=train_ids,
            validation_instance_ids=validation_ids,
            test_instances_used=0,
            source_sha256=current_source_sha256,
            initialization=initialization,
        )
        _write_csv(seed_dir / "epochs.csv", rows)
        _plot_learning_curve(seed_dir / "epochs.csv", seed_dir / "learning_curve.png")
        _write_json(
            seed_dir / "training_report.json",
            {
                "algorithm": "CHDGRL-LNS",
                "status": "frozen",
                "seed": int(seed),
                "epochs_completed": len(rows),
                "best_epoch": best_epoch,
                "best_score": list(best_score),
                "config_sha256": config_hash,
                "manifest_sha256": loader.manifest_sha256,
                "dataset_sha256": dataset_hash,
                "test_instances_used": 0,
                "initialization": initialization,
            },
        )
        frozen_paths.append(str(frozen_path.resolve()))

    report = {
        "algorithm": "CHDGRL-LNS",
        "status": "method_frozen",
        "config_sha256": config_hash,
        "manifest_sha256": loader.manifest_sha256,
        "dataset_sha256": dataset_hash,
        "training_seeds": list(config.training_seeds),
        "frozen_checkpoints": frozen_paths,
        "test_instances_used": 0,
        "initialization_checkpoint_sha256": initialization_sha256,
        "source_sha256": current_source_sha256,
    }
    _write_json(output_dir / "method_freeze_report.json", report)
    return report


__all__ = ["generate_dgrl_training_dataset", "train_dgrl_seed_grid"]
