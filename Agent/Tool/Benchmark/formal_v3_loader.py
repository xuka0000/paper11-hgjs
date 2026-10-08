from hashlib import sha256
from pathlib import Path

from Agent.Tool.Benchmark.benchmark_v3 import (
    FormalV3Spec,
    generate_formal_v3_instance,
    read_manifest,
)


class FormalV3InstanceLoader:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        rows = read_manifest(self.manifest_path)
        self._rows = {row.instance_id: row for row in rows}
        if len(self._rows) != len(rows):
            raise ValueError("duplicate Formal V3 instance id")
        self.manifest_sha256 = sha256(self.manifest_path.read_bytes()).hexdigest()
        self.instance_ids = tuple(sorted(self._rows))

    def row_for(self, instance_id):
        try:
            return self._rows[instance_id]
        except KeyError as error:
            raise KeyError(f"unknown Formal V3 instance: {instance_id}") from error

    def split_for(self, instance_id):
        return self.row_for(instance_id).split

    def __call__(self, instance_id):
        row = self.row_for(instance_id)
        instance = generate_formal_v3_instance(
            FormalV3Spec(
                row.split,
                row.difficulty,
                row.topology_family,
                row.seed,
                row.topology_slot,
            )
        )
        if instance.instance_hash != row.instance_hash:
            raise ValueError(f"Formal V3 instance hash mismatch: {instance_id}")
        if instance.topology_hash != row.topology_hash:
            raise ValueError(f"Formal V3 topology hash mismatch: {instance_id}")
        if instance.scenario_hash != row.scenario_hash:
            raise ValueError(
                f"Formal V3 physical scenario hash mismatch: {instance_id}"
            )
        if instance.candidate_pool_hash != row.candidate_pool_hash:
            raise ValueError(
                f"Formal V3 physical candidate pool hash mismatch: {instance_id}"
            )
        if instance.split != row.split:
            raise ValueError(f"Formal V3 split mismatch: {instance_id}")
        return instance


__all__ = ["FormalV3InstanceLoader"]
