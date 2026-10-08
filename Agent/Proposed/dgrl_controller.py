from dataclasses import dataclass, replace
from math import isfinite
from random import Random
from time import perf_counter

import torch
from torch.nn import functional

from .dgrl import clipped_double_q, select_scalar_action_index
from .dgrl_actions import (
    ActionProjection,
    JointAction,
    NEIGHBORHOOD_TYPES,
    SEED_NODE_TYPES,
    build_independent_dbu_action_pool,
    project_action,
    sample_dynamic_neighborhood,
)
from .hgnn import tensorize_graph


@dataclass(frozen=True)
class DGRLDecision:
    action: JointAction | None
    projection: object | None
    used_fallback: bool
    candidate_count: int
    value: float | None = None


def _neighborhood_type(node_type):
    for name in NEIGHBORHOOD_TYPES:
        if node_type in SEED_NODE_TYPES[name]:
            return name
    return None


class DGRLController:
    def __init__(
        self,
        model,
        *,
        release_profiles,
        time_budgets,
        sample_count,
        radius,
        seed,
        exploration_temperature=0.8,
        guard_count=0,
    ):
        if len(release_profiles) != model.release_levels:
            raise ValueError("release profiles must match model levels")
        if len(time_budgets) != model.time_levels:
            raise ValueError("time budgets must match model levels")
        if sample_count < 0:
            raise ValueError("sample_count must be nonnegative")
        if radius < 0:
            raise ValueError("radius must be nonnegative")
        if guard_count < 0:
            raise ValueError("guard_count must be nonnegative")
        if not 0 < exploration_temperature <= 1:
            raise ValueError("exploration_temperature must be in (0, 1]")
        self.model = model
        self.release_profiles = tuple(release_profiles)
        self.time_budgets = tuple(float(value) for value in time_budgets)
        self.sample_count = int(sample_count)
        self.radius = float(radius)
        self.guard_count = int(guard_count)
        self.exploration_temperature = float(exploration_temperature)
        self.rng = Random(seed)
        self.guard_rng = Random(int(seed) ^ 0x9E3779B97F4A7C15)

    def _freeze(self):
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _decode_center(
        self,
        graph,
        tensor_graph,
        context,
        *,
        training=False,
        node_state=None,
        global_state=None,
    ):
        if node_state is None or global_state is None:
            node_state, global_state = self.model.encode_state(
                tensor_graph,
                context,
            )
        proto = self.model.proto_action_from_state(global_state).vector
        type_count = len(NEIGHBORHOOD_TYPES)
        type_index = select_scalar_action_index(
            proto[:type_count],
            training=training,
            temperature=self.exploration_temperature,
            rng=self.rng,
        )
        neighborhood_type = NEIGHBORHOOD_TYPES[type_index]
        seed_query = proto[
            type_count : type_count + self.model.hidden_dim
        ]
        node_index = {
            node_id: index for index, node_id in enumerate(tensor_graph.node_ids)
        }
        eligible = [
            node.node_id
            for node in graph.nodes
            if node.node_type in SEED_NODE_TYPES[neighborhood_type]
        ]
        if not eligible:
            return None
        seed_scores = torch.stack(
            [
                functional.cosine_similarity(
                    seed_query.unsqueeze(0),
                    node_state[node_index[node_id]].unsqueeze(0),
                ).squeeze(0)
                for node_id in eligible
            ]
        )
        seed_id = eligible[
            select_scalar_action_index(
                seed_scores,
                training=training,
                temperature=self.exploration_temperature,
                rng=self.rng,
            )
        ]
        release_scores = -torch.abs(
            torch.linspace(
                0.0,
                1.0,
                steps=self.model.release_levels,
                dtype=proto.dtype,
                device=proto.device,
            )
            - proto[-2]
        )
        time_scores = -torch.abs(
            torch.linspace(
                0.0,
                1.0,
                steps=self.model.time_levels,
                dtype=proto.dtype,
                device=proto.device,
            )
            - proto[-1]
        )
        return JointAction(
            neighborhood_type,
            seed_id,
            select_scalar_action_index(
                release_scores,
                training=training,
                temperature=self.exploration_temperature,
                rng=self.rng,
            ),
            select_scalar_action_index(
                time_scores,
                training=training,
                temperature=self.exploration_temperature,
                rng=self.rng,
            ),
        )

    def _fallback(
        self,
        state,
        graph,
        remaining_budget_s,
        locked_tuple_ids,
        locked_arc_ids,
        projection_for=None,
        stop_requested=None,
    ):
        if projection_for is None:
            projection_for = lambda action: project_action(
                state,
                graph,
                action,
                self.release_profiles,
                time_budgets=self.time_budgets,
                remaining_budget_s=remaining_budget_s,
                locked_tuple_ids=locked_tuple_ids,
                locked_arc_ids=locked_arc_ids,
            )
        ranked_nodes = sorted(
            graph.nodes,
            key=lambda node: (
                node.node_type != "task",
                float(node.selected),
                -float(node.priority),
                node.node_id,
            ),
        )
        for node in ranked_nodes:
            if stop_requested is not None and stop_requested():
                return None, None
            neighborhood_type = _neighborhood_type(node.node_type)
            if neighborhood_type is None:
                continue
            for release_index in range(len(self.release_profiles)):
                for time_index in range(len(self.time_budgets)):
                    if stop_requested is not None and stop_requested():
                        return None, None
                    action = JointAction(
                        neighborhood_type,
                        node.node_id,
                        release_index,
                        time_index,
                    )
                    projection = projection_for(action)
                    if projection.accepted:
                        return action, projection
        return None, None

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
        if not training:
            self._freeze()
        selection_started = perf_counter()
        available_budget_s = max(0.0, float(remaining_budget_s))
        minimum_repair_budget_s = min(self.time_budgets)
        if available_budget_s < minimum_repair_budget_s:
            return DGRLDecision(None, None, True, 0, None)
        deadline = (
            selection_started + available_budget_s
            if isfinite(available_budget_s)
            else float("inf")
        )
        stop_requested = lambda: perf_counter() >= deadline
        tensor_graph = tensorize_graph(graph)
        adjacency = graph.adjacency()
        node_by_id = {node.node_id: node for node in graph.nodes}
        projection_cache = {}
        structural_projection_cache = {}

        def projection_for(action):
            if action not in projection_cache:
                time_limit_s = self.time_budgets[action.time_budget]
                current_budget_s = max(0.0, deadline - perf_counter())
                if time_limit_s > current_budget_s:
                    projection_cache[action] = ActionProjection(
                        False,
                        "time_budget_exceeded",
                        action,
                    )
                else:
                    structural_key = (
                        action.neighborhood_type,
                        action.seed_id,
                        action.release_profile,
                    )
                    if structural_key not in structural_projection_cache:
                        structural_projection_cache[structural_key] = project_action(
                            state,
                            graph,
                            action,
                            self.release_profiles,
                            time_budgets=self.time_budgets,
                            remaining_budget_s=remaining_budget_s,
                            locked_tuple_ids=locked_tuple_ids,
                            locked_arc_ids=locked_arc_ids,
                            adjacency=adjacency,
                            node_by_id=node_by_id,
                        )
                    base = structural_projection_cache[structural_key]
                    if time_limit_s > max(0.0, deadline - perf_counter()):
                        projection_cache[action] = ActionProjection(
                            False,
                            "time_budget_exceeded",
                            action,
                        )
                    else:
                        projection_cache[action] = replace(
                            base,
                            action=action,
                            time_limit_s=time_limit_s,
                        )
            return projection_cache[action]

        with torch.no_grad():
            node_state, global_state = self.model.encode_state(
                tensor_graph,
                context,
            )
            center = self._decode_center(
                graph,
                tensor_graph,
                context,
                training=training,
                node_state=node_state,
                global_state=global_state,
            )
            actions = ()
            if center is not None and self.sample_count:
                actions = sample_dynamic_neighborhood(
                    graph,
                    center,
                    release_levels=len(self.release_profiles),
                    time_levels=len(self.time_budgets),
                    sample_count=self.sample_count,
                    radius=self.radius,
                    rng=self.rng,
                    feasibility=lambda action: projection_for(action).accepted,
                    adjacency=adjacency,
                    node_by_id=node_by_id,
                    stop_requested=stop_requested,
                )
            guard_actions = ()
            if self.guard_count and not stop_requested():
                guard_actions = build_independent_dbu_action_pool(
                    state=state,
                    graph=graph,
                    release_profiles=self.release_profiles,
                    time_budgets=self.time_budgets,
                    remaining_budget_s=remaining_budget_s,
                    locked_tuple_ids=locked_tuple_ids,
                    locked_arc_ids=locked_arc_ids,
                    max_candidates=self.guard_count,
                    seed=self.guard_rng.getrandbits(64),
                    projection_for=projection_for,
                    stop_requested=stop_requested,
                )
            actions = tuple(dict.fromkeys((*actions, *guard_actions)))
            feasible = tuple(
                action for action in actions if projection_for(action).accepted
            )
            if feasible and not stop_requested():
                q1, q2 = self.model.q_values_from_state(
                    tensor_graph,
                    node_state,
                    global_state,
                    feasible,
                )
                values = clipped_double_q(q1, q2)
                index = select_scalar_action_index(
                    values,
                    training=training,
                    temperature=self.exploration_temperature,
                    rng=self.rng,
                )
                selected = feasible[index]
                selected_projection = projection_for(selected)
                if not selected_projection.accepted:
                    selected = None
                if selected is not None:
                    return DGRLDecision(
                        selected,
                        selected_projection,
                        False,
                        len(feasible),
                        float(values[index].cpu()),
                    )

        action, projection = self._fallback(
            state,
            graph,
            remaining_budget_s,
            locked_tuple_ids,
            locked_arc_ids,
            projection_for,
            stop_requested,
        )
        return DGRLDecision(
            action,
            projection,
            True,
            0,
            None,
        )


__all__ = ["DGRLController", "DGRLDecision"]
