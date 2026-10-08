"""Complete constraint-row membership over a declared finite physical pool.

Factor-graph and hypergraph views share this exact incidence list and features.
The pairwise projection intentionally discards factor identity and is labelled
as an information-loss control. No experiment outcome enters these features.
"""
from collections import defaultdict
from dataclasses import dataclass

import torch

from Agent.Run.publication_committed_service_domain import handling


@dataclass(frozen=True)
class ResourceFactor:
    factor_id: str
    kind: str
    members: tuple[str, ...]
    coefficients: tuple[float, ...]
    capacity: float
    fixed_load: float
    layer: int


@dataclass
class GroupGraph:
    node_ids: tuple[str, ...]
    node_features: torch.Tensor
    factor_ids: tuple[str, ...]
    factor_features: torch.Tensor
    incidence_nodes: torch.Tensor
    incidence_factors: torch.Tensor
    incidence_roles: torch.Tensor
    incidence_coefficients: torch.Tensor
    resources: tuple[ResourceFactor, ...]
    pair_source: torch.Tensor
    pair_target: torch.Tensor
    incidence_features: torch.Tensor
    service_node_count: int
    schema_version: str


NODE_FEATURES = ('selected', 'duration', 'energy_ratio', 'direct_cost', 'task_count',
                 'launch', 'recovery', 'required_wait_count', 'priority', 'completed_members',
                 'group_degree', 'group_missing_fraction', 'service', 'witness',
                 'component_miss', 'durable_certificate')
FACTOR_FEATURES = ('task_group', 'uav_occupancy', 'handling', 'capacity', 'fixed_load',
                   'selected_load', 'slack', 'member_count', 'time', 'priority',
                   'required_count', 'history_count', 'max_span', 'role_window',
                   'completion_link', 'lower_time', 'deadline_margin', 'history_age_max',
                   'history_expiry_min', 'history_expiry_bounded', 'durable_certificate')
INCIDENCE_FEATURES = ('coefficient', 'first_completion', 'last_completion',
                      'deadline_margin', 'window_margin')
GRAPH_VERSION = 'inspection-coupling-v2'


def resource_factors(problem):
    domain = problem.domain
    groups = defaultdict(list)
    for copy in problem.full_copies.values():
        for key in handling(domain.instance, copy, domain.context.state.current_layer):
            groups[('handling', key)].append(copy.tuple_id)
        for layer in range(max(domain.context.state.current_layer, copy.launch_layer-domain.instance.preparation_layers),
                           copy.recovery_layer+domain.instance.recovery_service_layers+domain.instance.turnaround_layers):
            groups[('uav_occupancy', (copy.uav_id, layer))].append(copy.tuple_id)
    capacities = {v.vehicle_id: v.operation_capacity for v in domain.instance.physical_scenario.vehicles}
    result = []
    for (kind, key), members in sorted(groups.items()):
        fixed = domain.fixed_operations.get(key, 0) if kind == 'handling' else domain.fixed_busy.get(key, 0)
        result.append(ResourceFactor(f'{kind}:{key}', kind, tuple(sorted(set(members))),
                                     (1.,)*len(set(members)),
                                     float(capacities[key[0]] if kind == 'handling' else 1.), float(fixed), key[-1]))
    return tuple(result)


def build_group_graph(problem, model, weights, *, pairwise=False):
    model.validate_domain(problem.domain)
    copies = problem.full_copies
    ids = list(sorted(copies)); index = {key: i for i, key in enumerate(ids)}
    service_node_count = len(ids)
    selected = problem.state.selected_tuple_ids
    horizon = max(1, max((c.recovery_layer for c in copies.values()), default=model.now))
    max_priority = max(1., max((g.priority for g in model.groups), default=1.))
    membership = defaultdict(list)
    for g in model.groups:
        for t in g.task_ids:
            membership[t].append(g)
    rows = []
    prior = {r.task_id for r in model.records if r.quality_ok and
             (r.valid_until_layer is None or r.valid_until_layer >= model.now)}
    selected_tasks = {t for key in selected for t in copies[key].task_ids} | prior
    for key in ids:
        c = copies[key]
        related = {g.group_id: g for t in c.task_ids for g in membership[t]}
        missing = [sum(max(0, r.minimum-len(set(r.alternatives)&selected_tasks))
                       for r in g.requirements)/sum(r.minimum for r in g.requirements)
                   for g in related.values()]
        rows.append([float(key in selected), (c.recovery_layer-c.launch_layer)/horizon,
                     c.reliable_energy/max(1e-9,c.battery_capacity*(1-c.reserve_ratio)),
                     (c.secondary_cost+weights['energy_weight']*c.expected_energy+weights['risk_weight']*c.risk_exposure)/horizon,
                     len(c.task_ids), c.launch_layer/horizon, c.recovery_layer/horizon,
                     len(handling(problem.domain.instance,c,model.now))/horizon,
                     sum(g.priority for g in related.values())/max_priority,
                      len(set(c.task_ids)&selected_tasks), len(related), sum(missing)/max(1,len(missing)),
                      1.,0.,0.,0.])
    factor_ids, factor_rows, nodes, factors, roles, coeffs = [], [], [], [], [], []
    incidence_rows = []
    pairs = set()

    def add(fid, features, members):
        edge = len(factor_ids);factor_ids.append(fid)
        factor_rows.append(features+[0.]*(len(FACTOR_FEATURES)-len(features)))
        members = tuple(members)
        for member in members:
            key, role, coefficient = member[:3]
            nodes.append(index[key]);factors.append(edge);roles.append(role);coeffs.append(coefficient)
            incidence_rows.append([coefficient]+list(member[3] if len(member)>3 else (0.,0.,0.,0.)))
        if pairwise:
            unique = sorted({index[member[0]] for member in members})
            pairs.update((a,b) for a in unique for b in unique if a != b)

    def auxiliary(key, *, witness=False, miss=False, selected_value=0., end=0, priority=0., durable=False):
        if key in index:
            raise ValueError('Auxiliary node conflicts with physical service identifier')
        index[key]=len(ids);ids.append(key)
        features=[0.]*len(NODE_FEATURES)
        for name,value in dict(selected=selected_value, launch=end/horizon, recovery=end/horizon,
                               priority=priority/max_priority, witness=float(witness),
                               component_miss=float(miss),durable_certificate=float(durable)).items():
            features[NODE_FEATURES.index(name)]=value
        rows.append(features)

    all_windows = model.completion_windows(copies)
    for g in model.groups:
        if len(g.requirements)>32:
            raise ValueError('More than 32 roles requires an encoder version change')
        windows=all_windows[g.group_id]
        feasible=[w.end for w in windows if all(len(r.historical_tasks)+
                  sum(a for key,a in r.terms if key in selected)>=r.requirement.minimum for r in w.roles)]
        chosen_end=min(feasible) if feasible else None
        durable=g.group_id in model.completed_group_ids
        miss_id='component_miss:'+g.group_id
        auxiliary(miss_id,miss=True,selected_value=float(not durable and chosen_end is None),
                  end=g.deadline_layer,priority=g.priority,durable=durable)
        links=[(miss_id,33,1.)]
        for window in windows:
            witness_id=f'witness:{g.group_id}:{window.end}'
            active=float(window.end==chosen_end)
            auxiliary(witness_id,witness=True,selected_value=active,end=window.end,priority=g.priority)
            links.append((witness_id,33,1.))
            for role in window.roles:
                requirement=role.requirement
                history=[r for r in model.records if r.quality_ok and
                         r.task_id in role.historical_tasks and role.lower<=r.completed_layer and
                         r.available_layer<=window.end and
                         (r.valid_until_layer is None or r.valid_until_layer>=window.end)]
                expiries=[(r.valid_until_layer-window.end)/horizon for r in history if r.valid_until_layer is not None]
                hist=len(role.historical_tasks)
                load=sum(a for key,a in role.terms if key in selected)
                members=[]
                for key,coefficient in role.terms:
                    times=[k for t,k in copies[key].completion_layers if t in requirement.alternatives
                           and t not in role.historical_tasks and max(role.lower,model.now)<=k<=window.end]
                    members.append((key,role.role_index+1,coefficient,
                                    (min(times)/horizon,max(times)/horizon,
                                     (g.deadline_layer-max(times))/horizon,(min(times)-role.lower)/horizon)))
                members.append((witness_id,33,-float(requirement.minimum)))
                add(f'role_window:{g.group_id}:{window.end}:{requirement.requirement_id}',
                    [1,0,0,requirement.minimum,hist,load,hist+load-requirement.minimum,
                     len(members),window.end/horizon,g.priority/max_priority,requirement.minimum,hist,
                     1. if g.max_span_layers is None else g.max_span_layers/horizon,1,0,
                     role.lower/horizon,(g.deadline_layer-window.end)/horizon,
                     max(((window.end-r.completed_layer)/horizon for r in history),default=0.),
                     min(expiries,default=0.),float(bool(expiries)),0.],members)
        add('group:'+g.group_id,[1,0,0,float(not durable),0,float(chosen_end is not None),
            float(chosen_end is None and not durable),len(links),g.deadline_layer/horizon,
            g.priority/max_priority,sum(r.minimum for r in g.requirements),0,0,0,1,0,
            (g.deadline_layer-model.now)/horizon,0,0,0,float(durable)],links)
    resources = resource_factors(problem)
    for r in resources:
        load = sum(a for key,a in zip(r.members,r.coefficients) if key in selected)
        add(r.factor_id,[0,int(r.kind=='uav_occupancy'),int(r.kind=='handling'),r.capacity,r.fixed_load,
                         load,r.capacity-r.fixed_load-load,len(r.members),r.layer/horizon,0,0,0,0],
            ((key,0,a) for key,a in zip(r.members,r.coefficients)))
    # Cardinalities and absolute loads are retained; no degree-only substitute.
    pair_list = sorted(pairs)
    return GroupGraph(tuple(ids),torch.tensor(rows,dtype=torch.float32).reshape(-1,len(NODE_FEATURES)),tuple(factor_ids),
                      torch.tensor(factor_rows,dtype=torch.float32).reshape(-1,len(FACTOR_FEATURES)),
                      torch.tensor(nodes,dtype=torch.long),torch.tensor(factors,dtype=torch.long),
                      torch.tensor(roles,dtype=torch.long),torch.tensor(coeffs,dtype=torch.float32),resources,
                      torch.tensor([a for a,b in pair_list],dtype=torch.long),
                      torch.tensor([b for a,b in pair_list],dtype=torch.long),
                      torch.tensor(incidence_rows,dtype=torch.float32).reshape(-1,len(INCIDENCE_FEATURES)),
                      service_node_count,GRAPH_VERSION)


def audit_legacy_graph(graph):
    by_kind = defaultdict(list)
    for edge in graph.edges:
        by_kind[edge.edge_type].append(len({i.node_id for i in edge.incidences}))
    return {kind:dict(edges=len(values),binary=sum(v==2 for v in values),
                      higher_order=sum(v>2 for v in values),minimum=min(values),maximum=max(values))
            for kind,values in sorted(by_kind.items())}
