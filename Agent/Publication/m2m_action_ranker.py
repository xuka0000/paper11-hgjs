"""Input-only scorers for bounded joint service edits.

The search owns admissibility and physical replay. Scores are preferences, never
feasibility certificates. No descriptor, solver outcome, or future observation is
used as a feature. A single encoded incumbent can score many beam extensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn


POOL_NAMES = ('add', 'release', 'quota', 'resource', 'route_event')
FEATURE_WIDTH = 10


def _scalar(value, scale=1.):
    value = float(value or 0.) / scale
    return max(-10., min(10., value)) if isfinite(value) else 0.


@dataclass(frozen=True)
class EncodedContext:
    """CPU tensors and stable IDs; move tensors in forward without rebuilding edges."""

    service_ids: tuple
    factor_ids: tuple
    factor_types: tuple
    service_features: torch.Tensor
    factor_features: torch.Tensor
    edge_service: torch.Tensor
    edge_factor: torch.Tensor
    edge_features: torch.Tensor
    pair_source: torch.Tensor
    pair_target: torch.Tensor
    service_edges: tuple
    selected: frozenset
    immutable: frozenset
    pool_factors: dict


def encode_context(context, *, no_incumbent=False):
    """Build declared service/quota/resource/route-event incidence once.

    A route event is a declared waiting-arc requirement. Its absence is retained
    as an empty typed pool. Object IDs are indexing keys only, not numeric labels.
    """
    view = context.view
    ids = tuple(sorted(view.services))
    service_index = {key: i for i, key in enumerate(ids)}
    qids, rids = tuple(sorted(view.quotas)), tuple(sorted(view.resources))
    event_members = {}
    for key in ids:
        for event in view.route_requirements.get(key, ()):
            if event is not None:
                event_members.setdefault(repr(event), set()).add(key)
    eids = tuple(sorted(event_members))
    factor_ids = tuple(('quota', x) for x in qids) + tuple(('resource', x) for x in rids) + tuple(('route_event', x) for x in eids)
    factor_types = tuple(x[0] for x in factor_ids)
    factor_index = {key: i for i, key in enumerate(factor_ids)}
    service_features = []
    selected = frozenset() if no_incumbent else context.selected
    for key in ids:
        service = view.services[key]
        layers = [end for _, end in service.completion_layers]
        service_features.append([
            float(key in selected), float(key in context.immutable and not no_incumbent),
            _scalar(context.costs.get(key, 0.), 100.), _scalar(len(layers), 4.),
            _scalar(min(layers, default=view.now)-view.now, 100.),
            _scalar(max(layers, default=view.now)-view.now, 100.),
            _scalar(len(context.groups[key]), 4.), _scalar(len(context.rows[key]), 4.),
            _scalar(len(view.route_requirements.get(key, ())), 4.), 1.,
        ])
    factor_features = []
    for gid in qids:
        q = view.quotas[gid]
        observed = set() if no_incumbent else set(context.base_groups[gid])
        factor_features.append([1., 0., 0., _scalar(q.minimum, 4.), _scalar(q.priority, 10.),
                                _scalar(len(observed), 4.), _scalar(max(0, q.minimum-len(observed)), 4.),
                                _scalar(q.deadline-view.now, 100.), _scalar(len(q.task_ids), 4.), 1.])
    for rid in rids:
        r = view.resources[rid]
        load = r.fixed_load if no_incumbent else context.base_loads[rid]
        factor_features.append([0., 1., 0., _scalar(r.capacity, 4.), _scalar(load, 4.),
                                _scalar(r.capacity-load, 4.), _scalar(r.fixed_load, 4.),
                                _scalar(len(r.members), 4.), 0., 1.])
    for eid in eids:
        factor_features.append([0., 0., 1., _scalar(len(event_members[eid]), 4.), 0., 0., 0., 0., 0., 1.])
    edges = []
    pools = {name: {} for name in ('quota', 'resource', 'route_event')}
    for key in ids:
        s = view.services[key]
        for gid in sorted(context.groups[key]):
            q = view.quotas[gid]
            valid = context.valid_tasks[key].get(gid, ())
            edges.append((service_index[key], factor_index[('quota', gid)],
                          [_scalar(len(valid), 4.), _scalar(len(context.tasks[key] & q.task_ids), 4.), 0.]))
            pools['quota'].setdefault(key, set()).add(factor_index[('quota', gid)])
        for rid, coeff in sorted(context.rows[key].items()):
            edges.append((service_index[key], factor_index[('resource', rid)], [_scalar(coeff, 4.), 0., 1.]))
            pools['resource'].setdefault(key, set()).add(factor_index[('resource', rid)])
        for event in view.route_requirements.get(key, ()):
            if event is None:
                continue
            fi = factor_index[('route_event', repr(event))]
            edges.append((service_index[key], fi, [1., 0., 0.]))
            pools['route_event'].setdefault(key, set()).add(fi)
    # Pairwise projection is evaluated as exact clique sums through incidence
    # in O(E). No arbitrary ID-based neighbor truncation or dense NxN array.
    service_edges = [[] for _ in ids]
    for ei, (si, _, _) in enumerate(edges):
        service_edges[si].append(ei)
    return EncodedContext(
        ids, factor_ids, factor_types,
        torch.tensor(service_features, dtype=torch.float32).reshape(-1, FEATURE_WIDTH),
        torch.tensor(factor_features, dtype=torch.float32).reshape(-1, FEATURE_WIDTH),
        torch.tensor([x[0] for x in edges], dtype=torch.long),
        torch.tensor([x[1] for x in edges], dtype=torch.long),
        torch.tensor([x[2] for x in edges], dtype=torch.float32).reshape(-1, 3),
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
        tuple(tuple(edge_ids) for edge_ids in service_edges),
        context.selected, context.immutable,
        {kind: {key: frozenset(value) for key, value in mapping.items()} for kind, mapping in pools.items()},
    )


class ActionRanker(nn.Module):
    """Four controlled action scorers with a common typed five-pool decoder."""

    def __init__(self, variant='action_hypergraph', hidden=64, layers=3, ablation=None):
        super().__init__()
        if variant not in ('action_mlp', 'action_pairwise', 'action_factor', 'action_hypergraph'):
            raise ValueError('Unknown ranker variant')
        if ablation not in (None, 'no_incumbent', 'legacy_pool'):
            raise ValueError('Only retraining ablations belong to the ranker')
        if hidden < 1 or layers < 1:
            raise ValueError('Positive width and depth required')
        self.variant, self.hidden, self.layers, self.ablation = variant, hidden, layers, ablation
        self.service_in = nn.Linear(FEATURE_WIDTH, hidden)
        self.factor_in = nn.Linear(FEATURE_WIDTH, hidden)
        self.edge_in = nn.Linear(3, hidden)
        self.pair_layers = nn.ModuleList(nn.Linear(2*hidden, hidden) for _ in range(layers))
        self.factor_message = nn.ModuleList(nn.Linear(3*hidden, hidden) for _ in range(layers))
        self.factor_update = nn.ModuleList(nn.Linear(2*hidden, hidden) for _ in range(layers))
        self.service_update = nn.ModuleList(nn.Linear(2*hidden, hidden) for _ in range(layers))
        self.attention = nn.ModuleList(nn.Linear(3*hidden, 1) for _ in range(layers))
        self.action_role = nn.Parameter(torch.empty(2, hidden))
        nn.init.normal_(self.action_role, std=.02)
        self.action_message = nn.Linear(3*hidden, hidden)
        self.action_attention = nn.Linear(3*hidden, 1)
        self.action_factor_update = nn.Linear(2*hidden+2, hidden)
        self.action_service_update = nn.Linear(2*hidden, hidden)
        self.pool_token = nn.Parameter(torch.zeros(5, hidden))
        self.decoder = nn.Sequential(nn.Linear(5*hidden+7, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, context):
        return encode_context(context, no_incumbent=self.ablation == 'no_incumbent')

    def _messages(self, encoded, device):
        s = torch.relu(self.service_in(encoded.service_features.to(device)))
        f = torch.relu(self.factor_in(encoded.factor_features.to(device)))
        es, ef = encoded.edge_service.to(device), encoded.edge_factor.to(device)
        edge = torch.relu(self.edge_in(encoded.edge_features.to(device)))
        if self.variant == 'action_mlp':
            return s, f
        for layer in range(self.layers):
            if self.variant == 'action_pairwise':
                aggregate = torch.zeros_like(s)
                counts = torch.zeros((s.shape[0], 1), device=device)
                if es.numel():
                    own = s[es] + edge
                    factor_sum = torch.zeros_like(f).index_add_(0, ef, own)
                    factor_degree = torch.zeros((f.shape[0], 1), device=device)
                    factor_degree.index_add_(0, ef, torch.ones((ef.numel(), 1), device=device))
                    neighbor = (factor_sum[ef]-own)/(factor_degree[ef]-1).clamp_min(1)
                    valid = (factor_degree[ef] > 1).to(s.dtype)
                    aggregate.index_add_(0, es, neighbor*valid)
                    counts.index_add_(0, es, valid)
                s = torch.relu(self.pair_layers[layer](torch.cat((s, aggregate/counts.clamp_min(1)), -1)))
                continue
            if not es.numel():
                continue
            incoming = torch.cat((s[es], f[ef], edge), -1)
            msg = torch.relu(self.factor_message[layer](incoming))
            if self.variant == 'action_factor':
                logits = self.attention[layer](incoming).squeeze(-1)
                maxima = torch.full((f.shape[0],), -torch.inf, device=device)
                maxima.scatter_reduce_(0, ef, logits, reduce='amax', include_self=True)
                weight = torch.exp(logits-maxima[ef])
                denom = torch.zeros(f.shape[0], device=device).index_add_(0, ef, weight)
                msg = msg * (weight/denom[ef].clamp_min(1e-8)).unsqueeze(-1)
            fac = torch.zeros_like(f).index_add_(0, ef, msg)
            f = torch.relu(self.factor_update[layer](torch.cat((f, fac), -1)))
            returned = torch.zeros_like(s).index_add_(0, es, f[ef] + edge)
            degree = torch.zeros((s.shape[0], 1), device=device)
            degree.index_add_(0, es, torch.ones((es.numel(), 1), device=device))
            s = torch.relu(self.service_update[layer](torch.cat((s, returned/degree.clamp_min(1)), -1)))
        return s, f

    def _action_conditioned(self, encoded, add, release, s, f, edge):
        """Update only factors touched by <=8 edits; static catalogue stays cached."""
        if self.variant not in ('action_factor', 'action_hypergraph'):
            return {}, {}
        index = {key: i for i, key in enumerate(encoded.service_ids)}
        edited = [(index[key], 0) for key in add] + [(index[key], 1) for key in release]
        edge_ids, roles = [], []
        for si, role in edited:
            incident = encoded.service_edges[si]
            edge_ids.extend(incident)
            roles.extend([role]*len(incident))
        if not edge_ids:
            return {}, {}
        device = s.device
        ei = torch.tensor(edge_ids, dtype=torch.long, device=device)
        er = torch.tensor(roles, dtype=torch.long, device=device)
        es = encoded.edge_service.to(device)[ei]
        ef = encoded.edge_factor.to(device)[ei]
        touched = tuple(sorted(set(encoded.edge_factor[edge_ids].tolist())))
        local = {fi: i for i, fi in enumerate(touched)}
        lf = torch.tensor([local[x] for x in encoded.edge_factor[edge_ids].tolist()], dtype=torch.long, device=device)
        input_message = torch.cat((s[es]+self.action_role[er], f[ef], edge[ei]), -1)
        msg = torch.tanh(self.action_message(input_message))
        if self.variant == 'action_factor':
            logits = self.action_attention(input_message).squeeze(-1)
            maxima = torch.full((len(touched),), -torch.inf, device=device)
            maxima.scatter_reduce_(0, lf, logits, reduce='amax', include_self=True)
            weights = torch.exp(logits-maxima[lf])
            denom = torch.zeros(len(touched), device=device).index_add_(0, lf, weights)
            msg = msg*(weights/denom[lf].clamp_min(1e-8)).unsqueeze(-1)
        aggregate = torch.zeros((len(touched), self.hidden), device=device).index_add_(0, lf, msg)
        role_count = torch.zeros((len(touched), 2), device=device)
        role_count.index_add_(0, lf, torch.nn.functional.one_hot(er, 2).to(s.dtype))
        touched_idx = torch.tensor(touched, dtype=torch.long, device=device)
        updated = torch.relu(self.action_factor_update(
            torch.cat((f[touched_idx], aggregate, role_count/4.), -1)))
        factor_map = {fi: updated[i] for fi, i in local.items()}
        service_map = {}
        for si, _ in edited:
            incident = encoded.service_edges[si]
            own_factors = sorted(set(encoded.edge_factor[list(incident)].tolist()))
            if own_factors:
                back = torch.stack([factor_map[fi] for fi in own_factors]).mean(0)
                service_map[si] = torch.relu(self.action_service_update(torch.cat((s[si], back), -1)))
        return service_map, factor_map

    def forward(self, encoded, actions, *, embeddings=None):
        device = self.pool_token.device
        actions = tuple(actions)
        if not actions:
            return torch.empty(0, device=device)
        s, f = embeddings if embeddings is not None else self._messages(encoded, device)
        index = {key: i for i, key in enumerate(encoded.service_ids)}
        edge_factor = encoded.edge_factor.tolist()
        service_slots, factor_slots = [], []
        service_lookup, factor_lookup = {}, {}
        service_pool_rows, service_pool_slots = [], []
        factor_pool_rows, factor_pool_slots = [], []
        action_edge_ids, action_edge_roles, action_edge_factors = [], [], []
        back_service, back_factor = [], []
        pool_counts = [0] * (5*len(actions))
        action_sizes = []
        conditioned = self.variant in ('action_factor', 'action_hypergraph')
        for batch, action in enumerate(actions):
            add, release = tuple(action.add), tuple(action.release)
            if (len(add) > 4 or len(release) > 4 or len(set(add)) != len(add)
                    or len(set(release)) != len(release) or set(add) & set(release)
                    or set(add) & encoded.selected or not set(release) <= encoded.selected
                    or set(release) & encoded.immutable or not set(add) <= index.keys()):
                raise ValueError('Action outside declared add/release bounds')
            action_sizes.append((len(add)/4., len(release)/4.))
            pools = (tuple(index[key] for key in add), tuple(index[key] for key in release))
            touched = set(add) | set(release)
            factor_pools = [tuple(sorted(set().union(*(
                encoded.pool_factors[kind].get(key, ()) for key in touched))))
                for kind in ('quota', 'resource', 'route_event')]
            for p, members in enumerate((*pools, *factor_pools)):
                row = 5*batch+p
                pool_counts[row] = len(members)
                for member in members:
                    if p < 2:
                        key = (batch, member)
                        if key not in service_lookup:
                            service_lookup[key] = len(service_slots)
                            service_slots.append(member)
                        service_pool_rows.append(row)
                        service_pool_slots.append(service_lookup[key])
                    else:
                        key = (batch, member)
                        if key not in factor_lookup:
                            factor_lookup[key] = len(factor_slots)
                            factor_slots.append(member)
                        factor_pool_rows.append(row)
                        factor_pool_slots.append(factor_lookup[key])
            if conditioned:
                for role, keys in enumerate((add, release)):
                    for key in keys:
                        si = index[key]
                        service_slot = service_lookup[(batch, si)]
                        unique_factor_slots = set()
                        for ei in encoded.service_edges[si]:
                            factor_slot = factor_lookup[(batch, edge_factor[ei])]
                            action_edge_ids.append(ei)
                            action_edge_roles.append(role)
                            action_edge_factors.append(factor_slot)
                            unique_factor_slots.add(factor_slot)
                        for factor_slot in sorted(unique_factor_slots):
                            back_service.append(service_slot)
                            back_factor.append(factor_slot)
        def tensor_index(values):
            return torch.tensor(values, dtype=torch.long, device=device)
        service_ids = tensor_index(service_slots)
        factor_ids = tensor_index(factor_slots)
        service_values = s[service_ids]
        factor_values = f[factor_ids]
        if conditioned and action_edge_ids:
            ei = tensor_index(action_edge_ids)
            role = tensor_index(action_edge_roles)
            factor_slot = tensor_index(action_edge_factors)
            es = encoded.edge_service.to(device)[ei]
            ef = encoded.edge_factor.to(device)[ei]
            edge = torch.relu(self.edge_in(encoded.edge_features.to(device)))[ei]
            incoming = torch.cat((s[es]+self.action_role[role], f[ef], edge), -1)
            message = torch.tanh(self.action_message(incoming))
            if self.variant == 'action_factor':
                logits = self.action_attention(incoming).squeeze(-1)
                maxima = torch.full((len(factor_slots),), -torch.inf, device=device)
                maxima.scatter_reduce_(0, factor_slot, logits, reduce='amax', include_self=True)
                weight = torch.exp(logits-maxima[factor_slot])
                denominator = torch.zeros(len(factor_slots), device=device).index_add_(0, factor_slot, weight)
                message = message*(weight/denominator[factor_slot].clamp_min(1e-8)).unsqueeze(-1)
            aggregate = torch.zeros_like(factor_values).index_add_(0, factor_slot, message)
            role_count = torch.zeros((len(factor_slots), 2), device=device)
            role_count.index_add_(0, factor_slot, torch.nn.functional.one_hot(role, 2).to(s.dtype))
            factor_values = torch.relu(self.action_factor_update(
                torch.cat((factor_values, aggregate, role_count/4.), -1)))
            bs, bf = tensor_index(back_service), tensor_index(back_factor)
            back_sum = torch.zeros_like(service_values).index_add_(0, bs, factor_values[bf])
            back_count = torch.zeros((len(service_slots), 1), device=device)
            back_count.index_add_(0, bs, torch.ones((len(back_service), 1), device=device))
            updated_service = torch.relu(self.action_service_update(
                torch.cat((service_values, back_sum/back_count.clamp_min(1)), -1)))
            service_values = torch.where(back_count > 0, updated_service, service_values)
        pooled = torch.zeros((5*len(actions), self.hidden), device=device)
        if service_pool_rows:
            pooled.index_add_(0, tensor_index(service_pool_rows), service_values[tensor_index(service_pool_slots)])
        if factor_pool_rows:
            pooled.index_add_(0, tensor_index(factor_pool_rows), factor_values[tensor_index(factor_pool_slots)])
        counts = torch.tensor(pool_counts, dtype=s.dtype, device=device).unsqueeze(-1)
        tokens = self.pool_token.unsqueeze(0).expand(len(actions), -1, -1).reshape(-1, self.hidden)
        vectors = torch.where(counts > 0, pooled/counts.clamp_min(1), tokens)
        masks = (counts > 0).to(s.dtype)
        if self.ablation == 'legacy_pool':
            shaped, present = vectors.reshape(len(actions), 5, self.hidden), masks.reshape(len(actions), 5, 1)
            merged = (shaped*present).sum(1)/present.sum(1).clamp_min(1)
            merged = torch.where(present.sum(1) > 0, merged, self.pool_token[0])
            vectors = merged[:, None, :].expand(-1, 5, -1).reshape(-1, self.hidden)
            masks = (present.sum(1) > 0).to(s.dtype)[:, None, :].expand(-1, 5, -1).reshape(-1, 1)
        rows = torch.cat((vectors.reshape(len(actions), 5*self.hidden), masks.reshape(len(actions), 5),
                          torch.tensor(action_sizes, dtype=s.dtype, device=device)), -1)
        return self.decoder(rows).squeeze(-1)


class NeuralActionPolicy:
    """Drop-in search scoring adapter with one encoding per immutable context."""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self._context = None
        self._encoded = None
        self._embeddings = None

    def score(self, context, partial, extensions, descriptors):
        if len(extensions) != len(descriptors):
            raise ValueError('Extensions and descriptors differ in length')
        if context is not self._context:
            self._encoded = self.model.encode(context)
            self._context = context
            with torch.no_grad():
                self._embeddings = self.model._messages(self._encoded, self.device)
        with torch.no_grad():
            values = self.model(self._encoded, extensions, embeddings=self._embeddings).detach().cpu().tolist()
        if not all(isfinite(value) for value in values):
            raise ValueError('Nonfinite neural action score')
        return values
