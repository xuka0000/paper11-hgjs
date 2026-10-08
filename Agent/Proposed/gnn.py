from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TensorPairGraph:
    node_features: torch.Tensor
    node_type_ids: torch.Tensor
    source_ids: torch.Tensor
    target_ids: torch.Tensor
    relation_type_ids: torch.Tensor
    node_ids: tuple[str, ...] = ()

    @property
    def feature_dim(self) -> int:
        return int(self.node_features.shape[1])


def expand_hypergraph_to_pairs(graph):
    sources = []
    targets = []
    relation_types = []
    edge_count = int(graph.edge_type_ids.shape[0])
    for edge_id in range(edge_count):
        members = graph.incidence_node_ids[
            graph.incidence_edge_ids == edge_id
        ].tolist()
        for source in members:
            for target in members:
                if source == target:
                    continue
                sources.append(source)
                targets.append(target)
                relation_types.append(int(graph.edge_type_ids[edge_id]))
    device = graph.node_features.device
    return TensorPairGraph(
        graph.node_features,
        graph.node_type_ids,
        torch.tensor(sources, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
        torch.tensor(relation_types, dtype=torch.long, device=device),
        graph.node_ids,
    )


class _PairMessageLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.message = nn.Linear(2 * hidden_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_state, relation_state, graph):
        if graph.source_ids.numel() == 0:
            return node_state
        messages = torch.tanh(
            self.message(
                torch.cat(
                    [node_state[graph.source_ids], relation_state],
                    dim=1,
                )
            )
        )
        update = messages.new_zeros(node_state.shape)
        update.index_add_(0, graph.target_ids, messages)
        degree = messages.new_zeros((node_state.shape[0], 1))
        degree.index_add_(
            0,
            graph.target_ids,
            messages.new_ones((messages.shape[0], 1)),
        )
        update = update / degree.clamp_min(1.0)
        return self.node_norm(node_state + update)


class OrdinaryGraphGNN(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        layers,
        max_node_types=64,
        max_edge_types=64,
    ):
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.node_type_embedding = nn.Embedding(max_node_types, hidden_dim)
        self.edge_type_embedding = nn.Embedding(max_edge_types, hidden_dim)
        self.layers = nn.ModuleList(
            _PairMessageLayer(hidden_dim) for _ in range(layers)
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, graph):
        pair_graph = expand_hypergraph_to_pairs(graph)
        node_state = self.input_projection(pair_graph.node_features)
        node_state = node_state + self.node_type_embedding(
            pair_graph.node_type_ids
        )
        relation_state = self.edge_type_embedding(
            pair_graph.relation_type_ids
        )
        for layer in self.layers:
            node_state = layer(node_state, relation_state, pair_graph)
        return node_state

    def forward(self, graph, regions):
        node_state = self.encode(graph)
        pooled = []
        for region in regions:
            if not region:
                raise ValueError("region must contain at least one node")
            indices = torch.as_tensor(
                region,
                dtype=torch.long,
                device=node_state.device,
            )
            pooled.append(node_state[indices].mean(dim=0))
        if not pooled:
            return node_state.new_empty((0,))
        return self.scorer(torch.stack(pooled)).squeeze(-1)


__all__ = ["OrdinaryGraphGNN", "TensorPairGraph", "expand_hypergraph_to_pairs"]
