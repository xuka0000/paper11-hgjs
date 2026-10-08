from dataclasses import dataclass
from hashlib import sha256

from Agent.Proposed.dgrl import LexicographicValue
from Agent.Proposed.dgrl_actions import build_independent_dbu_action_pool
from Agent.Proposed.dgrl_controller import DGRLController
from Agent.Proposed.hgnn import tensorize_graph
from Agent.Proposed.search import run_search
from Agent.Tool.Training.dgrl_training import DGRLTransition


@dataclass(frozen=True)
class DGRLEpisode:
    instance_id: str
    transitions: tuple[DGRLTransition, ...]
    step_rows: tuple[dict, ...]
    search_result: object


def _pool_seed(seed, instance_id, iteration):
    payload = f"{int(seed)}|{instance_id}|{int(iteration)}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def collect_dgrl_episode(
    instance,
    model,
    config,
    *,
    seed,
    training,
    search_config=None,
):
    controller = DGRLController(
        model,
        release_profiles=config.search.release_profiles,
        time_budgets=config.search.time_budgets,
        sample_count=config.search.controller_sample_count,
        radius=config.search.controller_radius,
        seed=seed,
        exploration_temperature=(
            config.trainer.sdn_exploration_temperature
        ),
        guard_count=config.search.controller_guard_count,
    )
    events = []
    result = run_search(
        state=instance.search_state,
        task_ids=instance.task_ids,
        miss_penalty=instance.miss_penalty,
        validation_state=instance.rolling_state,
        operation_capacity=instance.operation_capacity,
        config=search_config or config.search.to_search_config(),
        controller=controller,
        controller_training=training,
        step_observer=events.append,
    )
    pools = tuple(
        build_independent_dbu_action_pool(
            state=event.state_before,
            graph=event.graph,
            release_profiles=config.search.release_profiles,
            time_budgets=config.search.time_budgets,
            remaining_budget_s=event.remaining_budget_s,
            locked_tuple_ids=instance.rolling_state.locked_tuple_ids,
            locked_arc_ids=instance.rolling_state.locked_arc_ids,
            max_candidates=config.training.independent_pool_size,
            seed=_pool_seed(seed, instance.instance_id, event.iteration),
        )
        for event in events
    )
    if any(not pool for pool in pools):
        raise RuntimeError("a recorded DGRL step has no independent action pool")

    tensor_graphs = tuple(tensorize_graph(event.graph) for event in events)
    transitions = []
    step_rows = []
    for index, event in enumerate(events):
        terminal = index == len(events) - 1
        next_index = index + 1 if not terminal else index
        transitions.append(
            DGRLTransition(
                graph=tensor_graphs[index],
                context=event.context,
                action=event.action,
                reward=LexicographicValue(
                    primary_gain=event.primary_gain,
                    secondary_gain=event.secondary_gain,
                    runtime_cost=event.runtime_cost,
                    failure_cost=event.failure_cost,
                ),
                dbu_action_pool=pools[index],
                next_graph=tensor_graphs[next_index],
                next_context=events[next_index].context,
                next_actions=() if terminal else pools[next_index],
                done=terminal,
            )
        )
        step_rows.append(
            {
                "instance_id": instance.instance_id,
                "iteration": event.iteration,
                "neighborhood_type": event.action.neighborhood_type,
                "seed_id": event.action.seed_id,
                "release_profile": event.action.release_profile,
                "time_budget": event.action.time_budget,
                "primary_gain": event.primary_gain,
                "secondary_gain": event.secondary_gain,
                "failure_cost": event.failure_cost,
                "runtime_cost": event.runtime_cost,
                "independent_pool_size": len(pools[index]),
                "used_fallback": event.used_fallback,
                "done": terminal,
            }
        )
    return DGRLEpisode(
        instance.instance_id,
        tuple(transitions),
        tuple(step_rows),
        result,
    )


def evaluate_dgrl_policy(instance, model, config, *, seed):
    controller = DGRLController(
        model,
        release_profiles=config.search.release_profiles,
        time_budgets=config.search.time_budgets,
        sample_count=config.search.controller_sample_count,
        radius=config.search.controller_radius,
        seed=seed,
        exploration_temperature=config.trainer.sdn_exploration_temperature,
        guard_count=config.search.controller_guard_count,
    )
    return run_search(
        state=instance.search_state,
        task_ids=instance.task_ids,
        miss_penalty=instance.miss_penalty,
        validation_state=instance.rolling_state,
        operation_capacity=instance.operation_capacity,
        config=config.search.to_search_config(),
        controller=controller,
    )


__all__ = ["DGRLEpisode", "collect_dgrl_episode", "evaluate_dgrl_policy"]
