"""Focused interface and mechanism checks; no scientific performance claim."""
import io
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from Agent.Publication.joint_service_actions import ActionProposal
from Agent.Publication.joint_service_search import ActionContext
from Agent.Publication.m2m_action_ranker import ActionRanker, NeuralActionPolicy
from Agent.Publication.many_to_many_model import (
    CompletionQuota, ManyToManyModel, ServiceAlternative, SharedResource,
)


def fixture_context(order=('a', 'b', 'c'), *, route=True, selected=('a',)):
    services = {
        'a': ServiceAlternative('a', (('t1', 2),)),
        'b': ServiceAlternative('b', (('t2', 3),)),
        'c': ServiceAlternative('c', (('t3', 4),)),
    }
    quota = CompletionQuota('q', frozenset(('t1', 't2', 't3')), 2, 3., 5)
    resource = SharedResource('r', 'capacity', ('a', 'b', 'c'), (1., 1., 1.), 3.)
    waits = {'a': ('w1',), 'b': ('w1',), 'c': ('w2',)} if route else {}
    view = ManyToManyModel([services[key] for key in order], (quota,), (resource,), route_requirements=waits)
    return ActionContext(view, selected, costs={'a': 2., 'b': 3., 'c': 4.})


@pytest.mark.parametrize('variant', ['action_mlp', 'action_pairwise', 'action_factor', 'action_hypergraph'])
def test_variants_score_and_backpropagate(variant):
    torch.manual_seed(11)
    model = ActionRanker(variant)
    context = fixture_context()
    actions = [ActionProposal(('b',)), ActionProposal(('c',), ('a',))]
    encoded = model.encode(context)
    assert encoded.factor_types == ('quota', 'resource', 'route_event', 'route_event')
    assert encoded.edge_service.numel() > 0
    scores = model(encoded, actions)
    assert scores.shape == (2,) and torch.isfinite(scores).all()
    scores.sum().backward()
    assert model.service_in.weight.grad is not None
    if variant in ('action_factor', 'action_hypergraph'):
        assert model.factor_message[0].weight.grad is not None
        assert model.factor_message[0].weight.grad.abs().sum() > 0


def test_permutation_and_state_effects():
    torch.manual_seed(7)
    model = ActionRanker()
    action = [ActionProposal(('b',))]
    score = model(model.encode(fixture_context()), action)
    reordered = model(model.encode(fixture_context(('c', 'a', 'b'))), action)
    assert torch.allclose(score, reordered, atol=1e-6)
    changed = model(model.encode(fixture_context(selected=('c',))), action)
    assert not torch.allclose(score, changed)


def test_same_incidence_and_missing_typed_pool():
    context = fixture_context(route=False)
    factor = ActionRanker('action_factor').encode(context)
    hyper = ActionRanker('action_hypergraph').encode(context)
    assert torch.equal(factor.edge_service, hyper.edge_service)
    assert torch.equal(factor.edge_factor, hyper.edge_factor)
    assert factor.factor_types == hyper.factor_types == ('quota', 'resource')
    model = ActionRanker()
    encoded = model.encode(context)
    assert all(kind != 'route_event' for kind in encoded.factor_types)
    assert torch.isfinite(model(encoded, [ActionProposal(('b',))])).all()


def test_relations_change_hypergraph_score():
    torch.manual_seed(22)
    model = ActionRanker()
    encoded = model.encode(fixture_context())
    action = [ActionProposal(('b',))]
    original = model(encoded, action)
    # Remove b's quota incidence, preserving dimensions and other input data.
    keep = ~((encoded.edge_service == 1) & (encoded.edge_factor == 0))
    altered = replace(encoded, edge_service=encoded.edge_service[keep],
                      edge_factor=encoded.edge_factor[keep], edge_features=encoded.edge_features[keep])
    assert not torch.allclose(original, model(altered, action))


def test_pairwise_equivariant_to_arbitrary_id_renaming():
    original = fixture_context(route=False)
    rename = {'a': 'z', 'b': 'm', 'c': 'aa'}
    old = original.view
    services = [ServiceAlternative(rename[key], value.completion_layers)
                for key, value in old.services.items()]
    r = old.resources['r']
    renamed = ManyToManyModel(services, tuple(old.quotas.values()),
                              (SharedResource(r.resource_id, r.kind,
                                              tuple(rename[key] for key in r.members),
                                              r.coefficients, r.capacity),))
    changed = ActionContext(renamed, (rename['a'],),
                            costs={rename[key]: value for key, value in original.costs.items()})
    torch.manual_seed(19)
    model = ActionRanker('action_pairwise')
    one = model(model.encode(original), [ActionProposal(('b',), ('a',))])
    two = model(model.encode(changed), [ActionProposal(('m',), ('z',))])
    assert torch.allclose(one, two, atol=1e-6)


@pytest.mark.parametrize('variant', ['action_factor', 'action_hypergraph'])
def test_partial_action_changes_joint_factor_messages_with_same_pool_masks(variant):
    services = [ServiceAlternative(key, ((task, 2),))
                for key, task in [('a', 't1'), ('b', 't2'), ('c', 't3')]]
    quota = CompletionQuota('q', frozenset(('t1', 't2', 't3')), 2, 1., 3)
    resource = SharedResource('r', 'capacity', ('a', 'b', 'c'), (1., 1., 1.), 3.)
    context = ActionContext(ManyToManyModel(services, (quota,), (resource,)), ('a',))
    torch.manual_seed(23)
    model = ActionRanker(variant)
    encoded = model.encode(context)
    s, f = model._messages(encoded, model.pool_token.device)
    edge = torch.relu(model.edge_in(encoded.edge_features))
    _, one = model._action_conditioned(encoded, ('b',), (), s, f, edge)
    _, two = model._action_conditioned(encoded, ('b', 'c'), (), s, f, edge)
    assert one.keys() == two.keys()
    assert any(not torch.allclose(one[key], two[key]) for key in one)
    scores = model(encoded, [ActionProposal(('b',)), ActionProposal(('b', 'c'))])
    assert not torch.allclose(scores[0], scores[1])
    scores.sum().backward()
    assert model.action_message.weight.grad is not None
    assert model.action_message.weight.grad.abs().sum() > 0


def test_no_incumbent_retains_release_legality_and_legacy_retrains():
    context = fixture_context()
    action = [ActionProposal(('b',), ('a',))]
    for ablation in ('no_incumbent', 'legacy_pool'):
        model = ActionRanker(ablation=ablation)
        encoded = model.encode(context)
        assert torch.isfinite(model(encoded, action)).all()
        if ablation == 'no_incumbent':
            assert encoded.service_features[:, 0].sum() == 0


def test_policy_ignores_future_labels_and_caches_context():
    model = ActionRanker()
    policy = NeuralActionPolicy(model)
    context = fixture_context()
    actions = [ActionProposal(('b',))]
    with patch.object(model, '_messages', wraps=model._messages) as message_call:
        first = policy.score(context, ActionProposal(), actions, [{'primary_gain': 10000}])
        cached = policy._encoded
        second = policy.score(context, ActionProposal(), actions, [{'primary_gain': -10000}])
        assert message_call.call_count == 1
    assert first == second and policy._encoded is cached
    other = fixture_context(selected=('c',))
    policy.score(other, ActionProposal(), actions, [{}])
    assert policy._encoded is not cached


def test_bounds_and_serialization_roundtrip():
    model = ActionRanker('action_factor')
    encoded = model.encode(fixture_context())
    action = [ActionProposal(('b',), ('a',))]
    before = model(encoded, action).detach()
    file = io.BytesIO()
    torch.save(model.state_dict(), file)
    file.seek(0)
    restored = ActionRanker('action_factor')
    restored.load_state_dict(torch.load(file, weights_only=True))
    assert torch.allclose(before, restored(restored.encode(fixture_context()), action))
    with pytest.raises(ValueError):
        model(encoded, [ActionProposal(('b',), ('a', 'a'))])
    with pytest.raises(ValueError):
        model(encoded, [ActionProposal(('a',))])


def _reference_action_loop(model, encoded, actions):
    """Small independent loop matching the pre-batch per-action equations."""
    s, f = model._messages(encoded, model.pool_token.device)
    edge = torch.relu(model.edge_in(encoded.edge_features))
    index = {key: i for i, key in enumerate(encoded.service_ids)}
    rows = []
    for action in actions:
        add, release = action.add, action.release
        touched = set(add) | set(release)
        pools = [tuple(index[key] for key in add), tuple(index[key] for key in release)]
        pools.extend(tuple(sorted(set().union(*(
            encoded.pool_factors[kind].get(key, ()) for key in touched))))
            for kind in ('quota', 'resource', 'route_event'))
        service_delta, factor_delta = model._action_conditioned(encoded, add, release, s, f, edge)
        vectors, masks = [], []
        for p, members in enumerate(pools):
            masks.append(float(bool(members)))
            if not members:
                vectors.append(model.pool_token[p])
            else:
                tensor = s if p < 2 else f
                changed = service_delta if p < 2 else factor_delta
                vectors.append(torch.stack([changed.get(i, tensor[i]) for i in members]).mean(0))
        if model.ablation == 'legacy_pool':
            active = [v for v, m in zip(vectors, masks) if m]
            merged = torch.stack(active).mean(0) if active else model.pool_token[0]
            vectors, masks = [merged]*5, [float(bool(active))]*5
        rows.append(torch.cat((*vectors, torch.tensor(masks),
                               torch.tensor((len(add)/4., len(release)/4.)))))
    return model.decoder(torch.stack(rows)).squeeze(-1)


@pytest.mark.parametrize('variant,ablation', [
    ('action_mlp', None), ('action_pairwise', None), ('action_factor', None),
    ('action_hypergraph', None), ('action_hypergraph', 'legacy_pool'),
    ('action_hypergraph', 'no_incumbent'),
])
def test_batched_forward_matches_reference_values_and_gradients(variant, ablation):
    torch.manual_seed(31)
    model = ActionRanker(variant, ablation=ablation)
    encoded = model.encode(fixture_context())
    actions = [ActionProposal(('b',)), ActionProposal(('c',), ('a',)),
               ActionProposal(('b', 'c'), ('a',)), ActionProposal()]
    batched = model(encoded, actions)
    reference = _reference_action_loop(model, encoded, actions)
    assert torch.allclose(batched, reference, atol=1e-6, rtol=1e-5)
    batched.sum().backward(retain_graph=True)
    gradient = model.decoder[0].weight.grad.detach().clone()
    action_gradient = (model.action_message.weight.grad.detach().clone()
                       if variant in ('action_factor', 'action_hypergraph') else None)
    model.zero_grad(set_to_none=True)
    reference.sum().backward()
    assert torch.allclose(gradient, model.decoder[0].weight.grad, atol=1e-6, rtol=1e-5)
    if action_gradient is not None:
        assert torch.allclose(action_gradient, model.action_message.weight.grad,
                              atol=1e-6, rtol=1e-5)
