"""Isolated lock-interface guards and validation-only exposure audit.

This module never replaces the frozen repairer or rewrites stored trajectories.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import csv
import gzip
import json

from Agent.Proposed.repair import repair_region

PHASE = ROOT / "Data/Paper/publication_rolling_locks_20260910"


class LockIntegrityError(ValueError):
    pass


def lock_input_violations(search, rolling):
    tuples = {item.tuple_id: item for item in search.tuples}
    arcs = {arc.arc_id: arc for arc in search.arcs}
    violations = []
    for name, locked, domain in (("tuple", rolling.locked_tuple_ids, tuples), ("arc", rolling.locked_arc_ids, arcs)):
        violations.extend("missing_locked_" + name + ":" + key for key in sorted(set(locked) - set(domain)))
    for key in sorted(rolling.locked_tuple_ids & search.selected_tuple_ids & tuples.keys()):
        item = tuples[key]
        if not item.admitted:
            violations.append("unadmitted_selected_lock:" + key)
        for missing in sorted(set(item.required_arc_ids) - set(arcs)):
            violations.append("missing_locked_dependency:" + missing)
    for uid, uav in rolling.uavs.items():
        if not uav.is_airborne:
            continue
        residuals = [item for item in search.tuples if item.uav_id == uid and item.is_air_residual and item.admitted
                     and item.tuple_id in rolling.locked_tuple_ids and item.tuple_id in search.selected_tuple_ids]
        if len(residuals) != 1:
            violations.append("airborne_obligation_not_locked:" + str(uid))
    return tuple(violations)


def lock_output_violations(search, rolling, selected_tuples, selected_arcs):
    violations = []
    for name, locked, before, after in (
        ("tuple", rolling.locked_tuple_ids, search.selected_tuple_ids, set(selected_tuples)),
        ("arc", rolling.locked_arc_ids, search.selected_arc_ids, set(selected_arcs)),
    ):
        violations.extend("changed_locked_" + name + ":" + key for key in sorted(locked)
                          if (key in before) != (key in after))
    for uid, uav in rolling.uavs.items():
        if uav.is_airborne and sum(item.is_air_residual and item.uav_id == uid and item.tuple_id in selected_tuples
                                  for item in search.tuples) != 1:
            violations.append("airborne_recovery_omitted:" + str(uid))
    return tuple(violations)


def guarded_repair_region(*, state, validation_state, **kwargs):
    """Reject inconsistent ordinary-repair inputs before invoking the solver."""
    if kwargs.get("unsafe_tuple_ids"):
        raise LockIntegrityError("Emergency lock release requires a separately validated recovery mechanism")
    violations = lock_input_violations(state, validation_state)
    if violations:
        raise LockIntegrityError(";".join(violations))
    answer = repair_region(state=state, validation_state=validation_state, **kwargs)
    if answer.accepted:
        violations = lock_output_violations(state, validation_state, answer.selected_tuple_ids, answer.selected_arc_ids)
        if violations:
            raise LockIntegrityError(";".join(violations))
    return answer


def diagnostic_cases():
    from Agent.Test.conftest import base_state
    from Agent.Test.test_repair import _released_two_tuple_case

    state, closure = _released_two_tuple_case("locked")
    rolling = replace(base_state.__wrapped__(), remaining_tasks=frozenset({1}), locked_tuple_ids=frozenset({"locked"}))
    missing = replace(state, tuples=tuple(item for item in state.tuples if item.tuple_id != "locked"), selected_tuple_ids=frozenset())
    unadmitted = replace(state, tuples=tuple(replace(item, admitted=False) if item.tuple_id == "locked" else item for item in state.tuples))
    empty = replace(state, tuples=(), selected_tuple_ids=frozenset(), selected_arc_ids=frozenset())
    airborne = replace(rolling, locked_tuple_ids=frozenset(),
                       uavs={1: replace(rolling.uavs[1], is_airborne=True, remaining_tasks=(1,))}, airborne_tasks={1: (1,)})
    return {
        "valid": (state, closure, rolling),
        "missing_locked_tuple": (missing, closure, rolling),
        "unadmitted_locked_tuple": (unadmitted, closure, rolling),
        "missing_locked_arc": (state, closure, replace(rolling, locked_arc_ids=frozenset({"missing_arc"}))),
        "missing_airborne_obligation": (empty, closure, airborne),
    }


def run_diagnostics():
    results = []
    for name, (state, closure, rolling) in diagnostic_cases().items():
        kwargs = dict(state=state, closure=closure, task_ids=(1,), miss_penalty={1: 10.},
                      validation_state=rolling, operation_capacity={}, time_limit_s=2.)
        legacy = repair_region(**kwargs)
        try:
            strict = guarded_repair_region(**kwargs)
            blocked, reason = False, strict.reason
        except LockIntegrityError as error:
            blocked, reason = True, str(error)
        if not legacy.accepted or blocked != (name != "valid"):
            raise ValueError("Unexpected diagnostic outcome: " + name)
        results.append(dict(case=name, legacy_accepted=legacy.accepted, legacy_checker_ok=legacy.validation.ok,
                            legacy_selected_tuples=legacy.selected_tuple_ids, legacy_primary=legacy.primary_value,
                            guard_blocked=blocked, guard_reason=reason))
    return results


def audit_validation_traces():
    from Agent.Run.evaluate_publication_long_operations import decode

    folder = ROOT / "Data/Result/Diagnostic/publication_long_operations_20260909/validation"
    with (folder / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        entries = list(csv.DictReader(stream))
    counts = Counter()
    exposed = []
    hashes = {}
    for entry in entries:
        path = ROOT / entry["trace_path"]
        data = path.read_bytes()
        value = sha256(data).hexdigest()
        if value != entry["trace_sha256"]:
            raise ValueError("Changed raw validation trace")
        hashes[entry["trace_path"]] = value
        document = json.loads(gzip.decompress(data))
        counts["trajectories"] += 1
        for encoded in document["records"]:
            record = decode(encoded)
            problem = record["problem"]
            search, rolling = problem.search_state, problem.rolling_state
            counts["records"] += 1
            issues = lock_input_violations(search, rolling)
            if "selected" in record:
                selected = record["selected"]
                issues += lock_output_violations(search, rolling, selected.selected_tuple_ids, selected.selected_arc_ids)
                counts["accepted_decisions"] += 1
                counts["accepted_airborne_records"] += int(any(uav.is_airborne for uav in rolling.uavs.values()))
                counts["accepted_locked_tuple_observations"] += len(rolling.locked_tuple_ids)
                counts["accepted_locked_arc_observations"] += len(rolling.locked_arc_ids)
                if issues:
                    counts["accepted_with_violation"] += 1
            else:
                counts["unexecuted_failure_records"] += 1
                if issues:
                    counts["failed_with_input_violation"] += 1
            if issues:
                exposed.append(dict(trace=entry["trace_path"], layer=record["layer"], accepted="selected" in record,
                                    upstream_safety_status=problem.safety_status, issues=issues))
    return dict(counts=dict(counts), exposures=exposed, trace_hashes=hashes,
                summary_sha256=sha256((folder / "summary.csv").read_bytes()).hexdigest(),
                boundary="Only the registered long-operation validation panel is checked. Test panels and other dynamic generators are not certified. Records are repeated decisions, not independent physical samples.")


def main():
    from Agent.Tool.Evaluation.chdgrl_methods import evaluation_source_hashes

    PHASE.mkdir(exist_ok=True)
    before = evaluation_source_hashes(ROOT)
    diagnoses = run_diagnostics()
    report = dict(status="DIAGNOSTIC_AND_VALIDATION_AUDIT_COMPLETE", diagnostics=diagnoses,
                  validation_panel=audit_validation_traces(), production_sources=before,
                  implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  manuscript_sha256=sha256((ROOT / "Paper/AE_LaTeX_Template/main.tex").read_bytes()).hexdigest(),
                  boundary="Isolated pre/post-solve guard only. No production changes, new performance experiment, manuscript edit or claim of complete rolling-state verification.")
    if before != evaluation_source_hashes(ROOT):
        raise ValueError("Production source changed during audit")
    (PHASE / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "diagnostic_cases": len(diagnoses), **report["validation_panel"]["counts"]}))


if __name__ == "__main__":
    main()
