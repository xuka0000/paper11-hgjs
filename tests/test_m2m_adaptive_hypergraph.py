"""Representation/decoder isolation checks; no training or performance experiment."""
import pytest
import torch
from Agent.Publication.m2m_adaptive_hypergraph import AdaptiveConstraintRanker, reconstruct
from Agent.Publication.joint_service_actions import ActionProposal
from Agent.Publication.joint_service_search import ActionContext
from Agent.Publication.many_to_many_model import CompletionQuota, ManyToManyModel, ServiceAlternative, SharedResource


def context(order=('a', 'b', 'c'), selected=('a',)):
    services = {key: ServiceAlternative(key, ((task, layer),)) for key, task, layer in
                [('a', 't1', 2), ('b', 't2', 3), ('c', 't3', 4)]}
    quota = CompletionQuota('q', frozenset(('t1', 't2', 't3')), 2, 3., 5)
    resource = SharedResource('r', 'capacity', ('a', 'b', 'c'), (1., 1., 1.), 3.)
    view = ManyToManyModel([services[key] for key in order], (quota,), (resource,))
    return ActionContext(view, selected, costs={'a': 2., 'b': 3., 'c': 4.})


@pytest.mark.parametrize('mode', AdaptiveConstraintRanker.MODES)
def test_family_same_physical_domain_and_gradient(mode):
    torch.manual_seed(17)
    model = AdaptiveConstraintRanker(mode, hidden=16, layers=2)
    state = context(); encoded = model.encode(state)
    before = {field: getattr(encoded, field).clone() for field in
              ['edge_service', 'edge_factor', 'edge_features', 'service_features', 'factor_features']}
    actions = [ActionProposal(('b',)), ActionProposal(('c',), ('a',))]
    scores = model(encoded, actions)
    assert scores.shape == (2,) and torch.isfinite(scores).all()
    scores.sum().backward()
    assert model.decoder[0].weight.grad.abs().sum() > 0
    for name, original in before.items():
        assert torch.equal(original, getattr(encoded, name))
    assert encoded.selected == frozenset(('a',))
    if mode == 'dynamic':
        assert model.sender_query[0].weight.grad.abs().sum() > 0


def test_two_directed_attention_normalizations():
    torch.manual_seed(31)
    model = AdaptiveConstraintRanker(hidden=16, layers=2, collect_trace=True).eval()
    model(model.encode(context()), [ActionProposal(('b',))])
    assert len(model.last_trace) == 2
    for layer in model.last_trace:
        for weight_name, group_name in [('node_to_factor', 'edge_factor'), ('factor_to_node', 'edge_service')]:
            weights, groups = layer[weight_name], layer[group_name]
            for receiver in groups.unique():
                assert torch.allclose(weights[groups == receiver].sum(0), torch.ones(model.heads), atol=1e-6)
        assert (layer['node_to_factor'][~layer['membership']] == 0).all()


def test_reconstruction_retains_ties_and_changes_sender_sets():
    groups = torch.tensor([0, 0, 0, 1, 1])
    first = torch.tensor([-2., -2., -3., .4, -.3], requires_grad=True)
    gate, hard = reconstruct(first, groups, 2, 0., True)
    assert hard.tolist() == [True, True, False, True, False]
    # Same complete physical incidence; changing state-derived affinity changes messages.
    second = torch.tensor([-2., -2., .3, -.4, .3])
    _, changed = reconstruct(second, groups, 2, 0., False)
    assert changed.tolist() == [False, False, True, False, True]
    gate.sum().backward()
    assert torch.isfinite(first.grad).all() and (first.grad != 0).all()


def test_all_sender_control_matches_static_model_exactly():
    torch.manual_seed(7)
    adaptive = AdaptiveConstraintRanker(hidden=16, layers=2, threshold=-1e6).eval()
    static = AdaptiveConstraintRanker('static', hidden=16, layers=2).eval()
    static.load_state_dict(adaptive.state_dict())
    actions = [ActionProposal(('b',)), ActionProposal(('c',), ('a',))]
    encoded = adaptive.encode(context())
    assert torch.equal(adaptive(encoded, actions), static(encoded, actions))


def test_history_intervention_preserves_encoder_and_current_state():
    torch.manual_seed(11)
    model = AdaptiveConstraintRanker(hidden=16, layers=2).eval()
    encoded = model.encode(context())
    actions = [ActionProposal(('b',)), ActionProposal(('c',), ('a',))]
    embeddings = model._messages(encoded, model.pool_token.device)
    with_history = model(encoded, actions, embeddings=embeddings)
    model.history = False
    current_only = model(encoded, actions, embeddings=embeddings)
    assert not torch.allclose(with_history, current_only)
    again = model._messages(encoded, model.pool_token.device)
    assert all(torch.equal(a, b) for a, b in zip(embeddings, again))
    assert encoded.selected == frozenset(('a',))


def test_context_order_and_empty_ledger():
    torch.manual_seed(13)
    model = AdaptiveConstraintRanker(hidden=16, layers=2).eval()
    actions = [ActionProposal(('b',))]
    first = model(model.encode(context()), actions)
    reordered = model(model.encode(context(('c', 'a', 'b'))), actions)
    assert torch.equal(first, reordered)
    assert torch.isfinite(model(model.encode(context(selected=())), actions)).all()
    assert model(model.encode(context()), []).numel() == 0
