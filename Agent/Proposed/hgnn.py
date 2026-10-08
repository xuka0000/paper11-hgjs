from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TensorHypergraph:
    node_features: torch.Tensor
    node_type_ids: torch.Tensor
    edge_type_ids: torch.Tensor
    incidence_node_ids: torch.Tensor
    incidence_edge_ids: torch.Tensor
    role_ids: torch.Tensor
    node_ids: tuple[str, ...] = ()

    @property
    def feature_dim(self) -> int:
        return int(self.node_features.shape[1])


NODE_TYPES = (
    "task",
    "tuple",
    "arc",
    "time_node",
    "uav",
    "interval_resource",
    "operation_resource",
    "vehicle",
)

EDGE_TYPES = (
    "tuple_role",
    "coverage",
    "vehicle_flow",
    "vehicle_use",
    "launch",
    "recover",
    "uav_interval",
    "operation_capacity",
    "wait_dependency",
)

BASE_ROLES = (
    "anchor",
    "platform",
    "prepare",
    "takeoff",
    "recover",
    "task",
    "cover",
    "node",
    "incoming",
    "outgoing",
    "vehicle",
    "service",
    "tuple",
    "endpoint",
    "resource",
    "consumer",
    "wait_arc",
)

NODE_TYPE_INDEX = {name: index for index, name in enumerate(NODE_TYPES)}
EDGE_TYPE_INDEX = {name: index for index, name in enumerate(EDGE_TYPES)}
BASE_ROLE_INDEX = {name: index for index, name in enumerate(BASE_ROLES)}


def _role_id(role: str) -> int:
    if role.startswith("visit_"):
        return 32 + int(role.split("_", 1)[1])
    return BASE_ROLE_INDEX[role]


def tensorize_graph(graph) -> TensorHypergraph:
    node_ids = tuple(node.node_id for node in graph.nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    edge_index = {edge.edge_id: index for index, edge in enumerate(graph.edges)}
    degree = {node_id: 0 for node_id in node_ids}
    for edge in graph.edges:
        for incidence in edge.incidences:
            degree[incidence.node_id] += 1
    max_vehicle = max((node.vehicle_id or 0 for node in graph.nodes), default=1) or 1
    max_layer = max((node.time_layer or 0 for node in graph.nodes), default=1) or 1
    max_degree = max(degree.values(), default=1) or 1
    max_priority = max((node.priority for node in graph.nodes), default=1.0) or 1.0
    features = torch.tensor(
        [
            [
                1.0,
                float(node.vehicle_id or 0) / max_vehicle,
                float(node.time_layer or 0) / max_layer,
                float(degree[node.node_id]) / max_degree,
                float(node.priority) / max_priority,
                float(node.selected),
            ]
            for node in graph.nodes
        ],
        dtype=torch.float32,
    )
    incidence_node_ids = []
    incidence_edge_ids = []
    role_ids = []
    for edge in graph.edges:
        for incidence in edge.incidences:
            incidence_node_ids.append(node_index[incidence.node_id])
            incidence_edge_ids.append(edge_index[edge.edge_id])
            role_ids.append(_role_id(incidence.role))
    return TensorHypergraph(
        node_features=features,
        node_type_ids=torch.tensor(
            [NODE_TYPE_INDEX[node.node_type] for node in graph.nodes],
            dtype=torch.long,
        ),
        edge_type_ids=torch.tensor(
            [EDGE_TYPE_INDEX[edge.edge_type] for edge in graph.edges],
            dtype=torch.long,
        ),
        incidence_node_ids=torch.tensor(incidence_node_ids, dtype=torch.long),
        incidence_edge_ids=torch.tensor(incidence_edge_ids, dtype=torch.long),
        role_ids=torch.tensor(role_ids, dtype=torch.long),
        node_ids=node_ids,
    )


class _MessageLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.vertex_to_edge = nn.Linear(2 * hidden_dim, hidden_dim)
        self.edge_to_vertex = nn.Linear(2 * hidden_dim, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _mean_aggregate(messages, indices, count):
        output = messages.new_zeros((count, messages.shape[1]))
        output.index_add_(0, indices, messages)
        degree = messages.new_zeros((count, 1))
        degree.index_add_(0, indices, messages.new_ones((messages.shape[0], 1)))
        return output / degree.clamp_min(1.0)

    def forward(self, node_state, edge_state, role_state, graph):
        node_messages = torch.tanh(
            self.vertex_to_edge(
                torch.cat(
                    [node_state[graph.incidence_node_ids], role_state],
                    dim=1,
                )
            )
        )
        edge_update = self._mean_aggregate(
            node_messages,
            graph.incidence_edge_ids,
            edge_state.shape[0],
        )
        edge_state = self.edge_norm(edge_state + edge_update)
        edge_messages = torch.tanh(
            self.edge_to_vertex(
                torch.cat(
                    [edge_state[graph.incidence_edge_ids], role_state],
                    dim=1,
                )
            )
        )
        node_update = self._mean_aggregate(
            edge_messages,
            graph.incidence_node_ids,
            node_state.shape[0],
        )
        node_state = self.node_norm(node_state + node_update)
        return node_state, edge_state


class RoleAwareHGNN(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        layers: int,
        max_node_types: int = 64,
        max_edge_types: int = 64,
        max_roles: int = 128,
    ):
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.node_type_embedding = nn.Embedding(max_node_types, hidden_dim)
        self.edge_type_embedding = nn.Embedding(max_edge_types, hidden_dim)
        self.role_embedding = nn.Embedding(max_roles, hidden_dim)
        self.layers = nn.ModuleList(_MessageLayer(hidden_dim) for _ in range(layers))
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, graph: TensorHypergraph) -> torch.Tensor:
        node_state = self.input_projection(graph.node_features)
        node_state = node_state + self.node_type_embedding(graph.node_type_ids)
        edge_state = self.edge_type_embedding(graph.edge_type_ids)
        role_state = self.role_embedding(graph.role_ids)
        for layer in self.layers:
            node_state, edge_state = layer(node_state, edge_state, role_state, graph)
        return node_state

    def forward(self, graph: TensorHypergraph, regions) -> torch.Tensor:
        node_state = self.encode(graph)
        pooled = []
        for region in regions:
            if not region:
                raise ValueError("region must contain at least one node")
            indices = torch.as_tensor(region, dtype=torch.long, device=node_state.device)
            pooled.append(node_state[indices].mean(dim=0))
        if not pooled:
            return node_state.new_empty((0,))
        return self.scorer(torch.stack(pooled)).squeeze(-1)
