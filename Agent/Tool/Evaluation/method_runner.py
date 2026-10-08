from dataclasses import dataclass, replace

from Agent.Proposed.search import run_search


@dataclass(frozen=True)
class MethodSpec:
    method: object
    config_overrides: tuple[tuple[str, object], ...] = ()
    budget_id: str = ""
    evaluation_seed: int | None = None
    shared_reference: bool = False
    checkpoint_sha256: str = ""
    model_kind: str = ""

    def apply(self, config):
        names = [name for name, _ in self.config_overrides]
        if len(names) != len(set(names)):
            raise ValueError("duplicate search config override")
        updates = dict(self.config_overrides)
        unknown = sorted(name for name in updates if not hasattr(config, name))
        if unknown:
            raise ValueError(f"unknown search config override: {unknown}")
        return replace(config, **updates)


@dataclass(frozen=True)
class MethodExecution:
    result: object
    method_family: str
    config: object
    budget_id: str = ""
    evaluation_seed: int | None = None
    shared_reference: bool = False
    checkpoint_sha256: str = ""
    model_kind: str = ""


def _method_spec(method):
    return method if isinstance(method, MethodSpec) else MethodSpec(method)


def method_spec(method):
    return _method_spec(method)


def method_family(method):
    actual = _method_spec(method).method
    declared = getattr(actual, "method_family", None)
    if declared is not None:
        declared = str(declared).strip()
        if not declared:
            raise ValueError("declared method family must not be empty")
        return declared
    return (
        "independent_solver"
        if callable(getattr(actual, "solve", None))
        else "region_selector"
    )


def run_method(
    method,
    *,
    state,
    task_ids,
    miss_penalty,
    validation_state,
    operation_capacity,
    config,
):
    specification = _method_spec(method)
    actual = specification.method
    family = method_family(specification)
    applied_config = specification.apply(config)
    kwargs = {
        "state": state,
        "task_ids": task_ids,
        "miss_penalty": miss_penalty,
        "validation_state": validation_state,
        "operation_capacity": operation_capacity,
        "config": applied_config,
    }
    result = (
        actual.solve(**kwargs)
        if callable(getattr(actual, "solve", None))
        else run_search(**kwargs, scorer=actual)
    )
    return MethodExecution(
        result,
        family,
        applied_config,
        specification.budget_id,
        specification.evaluation_seed,
        specification.shared_reference,
        specification.checkpoint_sha256,
        specification.model_kind,
    )


__all__ = [
    "MethodExecution",
    "MethodSpec",
    "method_family",
    "method_spec",
    "run_method",
]
