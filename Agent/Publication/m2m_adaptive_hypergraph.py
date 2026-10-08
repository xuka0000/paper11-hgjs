"""Development candidate grounded in Wang2026 and HMAGAT, not a reproduction.

Wang arXiv2503.10421v2, PDF15–20: adaptive constraint-specific sender sets.
HMAGAT arXiv2602.06733v2, PDF5, equations1–4: two directed attention stages.
The physical incidence/candidate domain stays complete. Reconstruction selects
representation messages only. All modes share the same joint-action decoder.
This module is isolated from the launched seed17 source and selected weights.
"""
from math import sqrt
import torch
from torch import nn
from Agent.Publication.m2m_action_ranker import ActionRanker


def segment_weights(logits, groups, size, membership=None):
    """Stable receiver-local normalization, with a differentiable membership gate."""
    maxima = logits.new_full((size, logits.shape[1]), -torch.inf)
    maxima.scatter_reduce_(0, groups[:, None].expand_as(logits), logits,
                           reduce='amax', include_self=True)
    values = torch.exp(logits - maxima[groups])
    if membership is not None:
        values = values * membership[:, None]
    total = values.new_zeros((size, values.shape[1])).index_add_(0, groups, values)
    return values / total[groups].clamp_min(1e-8)


def reconstruct(logits, groups, size, threshold, training):
    """Keep selected senders or all tied maxima when a factor would be empty."""
    hard = logits > threshold
    count = logits.new_zeros(size).index_add_(0, groups, hard.to(logits.dtype))
    maximum = logits.new_full((size,), -torch.inf)
    maximum.scatter_reduce_(0, groups, logits, reduce='amax', include_self=True)
    hard = hard | ((count[groups] == 0) & (logits == maximum[groups]))
    gate = hard.to(logits.dtype)
    if training:
        soft = torch.sigmoid(logits - threshold)
        gate = gate + (soft - soft.detach())
    return gate, hard


class AdaptiveConstraintRanker(ActionRanker):
    """Matched development family; no stored physical output is a feature.

    ``history`` uses the current retained-service ledger, rather than introducing
    an unobserved sequence of past routes. The history channel sums logits before
    external action selection; it does not replicate Wang's autoregressive RL.
    """
    MODES = ('mlp', 'pairwise', 'fixed_sum', 'static', 'dynamic')

    def __init__(self, mode='dynamic', hidden=64, layers=3, heads=4,
                 history=True, threshold=0., collect_trace=False):
        if mode not in self.MODES or heads < 1 or hidden % heads:
            raise ValueError('Valid mode and positive divisor head count required')
        # Identical action-conditioning/readout for every representation mode.
        super().__init__('action_hypergraph', hidden=hidden, layers=layers)
        self.mode, self.heads, self.history = mode, heads, bool(history)
        self.threshold, self.collect_trace = float(threshold), bool(collect_trace)
        self.sender_query = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.sender_key = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.receiver_query = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.receiver_key = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.history_query = nn.Linear(hidden, hidden)
        self.history_key = nn.Linear(2 * hidden, hidden)
        self.last_trace = ()

    def _messages(self, encoded, device):
        s = torch.relu(self.service_in(encoded.service_features.to(device)))
        f = torch.relu(self.factor_in(encoded.factor_features.to(device)))
        es, ef = encoded.edge_service.to(device), encoded.edge_factor.to(device)
        edge = torch.relu(self.edge_in(encoded.edge_features.to(device)))
        trace = []; width = self.hidden // self.heads
        if self.mode == 'mlp' or not es.numel():
            self.last_trace = (); return s, f
        for layer in range(self.layers):
            if self.mode == 'pairwise':
                own = s[es] + edge
                total = torch.zeros_like(f).index_add_(0, ef, own)
                degree = s.new_zeros((f.shape[0], 1)).index_add_(0, ef, s.new_ones((ef.numel(), 1)))
                valid = (degree[ef] > 1).to(s.dtype)
                neighbor = (total[ef] - own) / (degree[ef] - 1).clamp_min(1)
                aggregate = torch.zeros_like(s).index_add_(0, es, neighbor * valid)
                count = s.new_zeros((s.shape[0], 1)).index_add_(0, es, valid)
                s = torch.relu(self.pair_layers[layer](torch.cat((s, aggregate / count.clamp_min(1)), -1)))
                continue
            queries = self.sender_query[layer](s[es]).reshape(-1, self.heads, width)
            keys = self.sender_key[layer](f[ef] + edge).reshape(-1, self.heads, width)
            logits = (queries * keys).sum(-1) / sqrt(width)
            membership = logits.new_ones(es.numel()); hard = membership.bool()
            if self.mode == 'dynamic':
                membership, hard = reconstruct(logits.mean(-1), ef, f.shape[0],
                                                self.threshold, self.training)
            alpha = segment_weights(logits, ef, f.shape[0], membership)
            if self.mode == 'fixed_sum':
                alpha = torch.ones_like(alpha)
            messages = torch.relu(self.factor_message[layer](torch.cat((s[es], f[ef], edge), -1)))
            messages = (messages.reshape(-1, self.heads, width) * alpha[:, :, None]).reshape(-1, self.hidden)
            aggregate = torch.zeros_like(f).index_add_(0, ef, messages)
            f = torch.relu(self.factor_update[layer](torch.cat((f, aggregate), -1)))
            queries = self.receiver_query[layer](s[es]).reshape(-1, self.heads, width)
            keys = self.receiver_key[layer](f[ef] + edge).reshape(-1, self.heads, width)
            beta = segment_weights((queries * keys).sum(-1) / sqrt(width), es, s.shape[0])
            if self.mode == 'fixed_sum':
                degree = s.new_zeros(s.shape[0]).index_add_(0, es, s.new_ones(es.numel()))
                beta = (1. / degree[es])[:, None].expand_as(beta)
            returned = ((f[ef] + edge).reshape(-1, self.heads, width) * beta[:, :, None]).reshape(-1, self.hidden)
            returned = torch.zeros_like(s).index_add_(0, es, returned)
            s = torch.relu(self.service_update[layer](torch.cat((s, returned), -1)))
            if self.collect_trace:
                trace.append(dict(layer=layer, membership=hard.detach().cpu(),
                                  node_to_factor=alpha.detach().cpu(),
                                  factor_to_node=beta.detach().cpu(),
                                  edge_service=es.detach().cpu(), edge_factor=ef.detach().cpu()))
        self.last_trace = tuple(trace)
        return s, f

    def forward(self, encoded, actions, *, embeddings=None):
        actions = tuple(actions)
        device = self.pool_token.device
        values = embeddings if embeddings is not None else self._messages(encoded, device)
        scores = super().forward(encoded, actions, embeddings=values)
        if not self.history or not actions or not encoded.selected:
            return scores
        services = values[0]
        index = {key: i for i, key in enumerate(encoded.service_ids)}
        retained = [index[key] for key in encoded.selected]
        query = self.history_query(services[retained].mean(0))
        def pool(keys):
            return services[[index[key] for key in keys]].mean(0) if keys else services.new_zeros(self.hidden)
        keys = torch.stack([torch.cat((pool(a.add), pool(a.release)), -1) for a in actions])
        historical = (self.history_key(keys) * query).sum(-1) / sqrt(self.hidden)
        return scores + historical
