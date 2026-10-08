"""Paper11 MILP adapter for published vehicle-group LNS.

Derived from milp.py without changing physical rows. Additional bounds freeze
all vehicle-owned decisions outside the selected published LNS groups.
"""
from collections import defaultdict
from math import ceil, isfinite
from time import perf_counter
from ortools.linear_solver import pywraplp
from Agent.Run.publication_committed_service_domain import Plan, copy_terms, handling
from Agent.Run.publication_plan_distance import objective, plan_reference, service_key
from Agent.Run.check_publication_hamming_schedule import check_plan
from .budget import BudgetExpired


class ObjectiveAccountingError(ValueError):
    def __init__(self,evidence,plan):
        super().__init__('Solver objective differs from physical accounting '+repr(evidence))
        self.evidence=evidence;self.plan=plan


SCIP_REPRODUCIBILITY_PARAMETERS = """randomization/randomseedshift = 0
randomization/permutationseed = 0
randomization/lpseed = 0
parallel/maxnthreads = 1"""


def configure_reproducible_scip(solver):
    solver.SetNumThreads(1)
    accepted=solver.SetSolverSpecificParametersAsString(SCIP_REPRODUCIBILITY_PARAMETERS)
    if not accepted:raise ValueError('SCIP rejected reproducibility parameters')


def valid_makespan_projection(evidence):
    """Reconstruct the continuous epigraph from a checked integer schedule.

    The integer assignment, every model row and the projected linear objective
    must already agree with the independently checked physical schedule.
    Reconstruction in either direction requires all projected rows and the
    physical objective to agree. It establishes a feasible primal witness,
    never an optimality certificate.
    """
    return (evidence['stage']=='secondary' and evidence['raw_status'] in
            (pywraplp.Solver.OPTIMAL,pywraplp.Solver.FEASIBLE)
            and not evidence['fractional'] and not evidence['rounded_row_violations']
            and all(isfinite(evidence[k]) for k in
                    ('makespan_variable','exact_makespan','projected_linear_objective','physical_value'))
            and abs(evidence['projected_linear_objective']-evidence['physical_value'])<=1e-6)


def solve_committed(domain, config, *, primary_seconds=10., secondary_seconds=10., explicit=False, deadline=None,
                    node_limit=None, incumbent=None, validation_reserve_s=.002,
                    released_vehicle_ids=None, relative_mip_gap=0.1, model_observer=None, task_groups=None):
    if deadline is not None:deadline.check()
    if validation_reserve_s<0:raise ValueError('Nonnegative validation reserve required')
    tick=perf_counter();instance=domain.instance;state=domain.context.state;now=state.current_layer
    weights=config.objective_weights;solver=pywraplp.Solver.CreateSolver('SCIP');configure_reproducible_scip(solver)
    if not 0 <= relative_mip_gap < 1:raise ValueError('Invalid MIP gap')
    if not solver.SetSolverSpecificParametersAsString(f'limits/gap = {relative_mip_gap}'):
        raise ValueError('SCIP rejected relative gap')
    if node_limit is not None:
        if node_limit<=0:raise ValueError('Positive solver node limit required')
        solver.SetSolverSpecificParametersAsString(f'limits/nodes = {int(node_limit)}')
    cost_domain=getattr(domain,'objective_reference_domain',domain)
    if task_groups is not None:
        task_groups.validate_domain(domain)
        if domain.templates and not explicit:
            raise ValueError('Task-group objective requires explicit finite service copies')
    def account(plan):
        values=objective(cost_domain,plan,weights)
        return values if task_groups is None else (task_groups.evaluate(plan)['primary'],values[1])
    reference=plan_reference(cost_domain);churn=weights['plan_change_weight']
    infinity=solver.infinity();variables=[];hint_rows=[]
    def boolean(name):
        if deadline is not None and len(variables)%128==0:deadline.check()
        v=solver.BoolVar(name);variables.append(v);return v
    def row(terms,lower,upper):
        if deadline is not None and solver.NumConstraints()%128==0:deadline.check()
        result=solver.RowConstraint(lower,upper,'');combined=defaultdict(float)
        for variable,coefficient in terms:combined[variable.index()]+=coefficient
        for index,coefficient in combined.items():
            if coefficient:result.SetCoefficient(variables[index],coefficient)
        hint_rows.append((dict(combined),lower,upper))
    arcs=domain.problem.search_state.arcs
    x={a.arc_id:boolean('x:'+a.arc_id) for a in arcs}
    g={v:boolean('g:'+str(v)) for v in state.vehicles}
    # Every attainable completion/return lies within the absolute time horizon.
    maximum=solver.NumVar(0.,float(max((a.arrival_layer for a in arcs),default=0)),'makespan');variables.append(maximum)
    incidence=defaultdict(list);duty=defaultdict(list);secondary=defaultdict(float)
    task_terms=defaultdict(list);vehicle_terms=defaultdict(list);operations=defaultdict(list)
    starts=defaultdict(list);ends=defaultdict(list)
    for a in arcs:
        tail=('source',-1) if a.tail=='source' else (a.tail,a.departure_layer)
        head=('sink',-1) if a.head=='sink' else (a.head,a.arrival_layer)
        incidence[(a.vehicle_id,tail)].append((x[a.arc_id],-1.));incidence[(a.vehicle_id,head)].append((x[a.arc_id],1.))
        duty[a.vehicle_id].append((x[a.arc_id],a.duty_time))
        secondary[x[a.arc_id].index()]+=a.secondary_cost+weights['time_weight']*a.duty_time
        secondary[x[a.arc_id].index()]+=churn*reference.arc_coefficient(a)
        if a.head=='sink':row([(maximum,1.),(x[a.arc_id],-a.arrival_layer)],0.,infinity)
    for vid in g:
        incidence[(vid,('source',-1))].append((g[vid],1.));incidence[(vid,('sink',-1))].append((g[vid],-1.))
        if state.vehicles[vid].remaining_duty_time is not None:
            row(duty[vid],-infinity,state.vehicles[vid].remaining_duty_time)
    for terms in incidence.values():row(terms,0.,0.)
    for aid in domain.required_arc_ids:row([(x[aid],1.)],1.,1.)
    mandatory=set(domain.required_vehicle_ids)|{a.vehicle_id for a in arcs if a.arc_id in domain.required_arc_ids}
    for vid in mandatory:row([(g[vid],1.)],1.,1.)
    p=instance.preparation_layers;r=instance.recovery_service_layers;turn=instance.turnaround_layers
    towers={int(t.tower_id):t for t in instance.physical_scenario.power.towers}
    chosen_vars={};copy_records={};y={};launch_vars={};recovery_vars={}

    def add_explicit(copy):
        variable=boolean('copy:'+copy.tuple_id);chosen_vars[copy.tuple_id]=variable;copy_records[copy.tuple_id]=copy
        covered,cost=copy_terms(instance,copy,state,weights)
        for task in covered:task_terms[task].append((variable,1.))
        vehicle_terms[copy.vehicle_id].append((variable,-1.))
        row([(variable,1.),(g[copy.vehicle_id],-1.)],-infinity,0.)
        starts[(copy.uav_id,max(now,copy.launch_layer-p))].append((variable,1.))
        ends[(copy.uav_id,copy.recovery_layer+r+turn)].append((variable,-1.))
        for key in handling(instance,copy,now):
            aid=domain.waiting.get(key)
            if aid is None:row([(variable,1.)],0.,0.)
            else:row([(variable,1.),(x[aid],-1.)],-infinity,0.)
            operations[key].append((variable,1.))
        secondary[variable.index()]+=cost
        secondary[variable.index()]+=churn*reference.copy_coefficient(copy)

    for copy in domain.specials:add_explicit(copy)
    for group in domain.groups.values():row([(chosen_vars[cid],1.) for cid in group],1.,1.)
    for t in domain.templates:
        if explicit:
            for launch in t.launches:
                for recovery in domain.recovery_options(t,launch):add_explicit(domain.materialize(t,launch,recovery))
            continue
        tid=t.template_id;y[tid]=boolean('y:'+tid)
        task_terms[t.task_id].append((y[tid],1.));vehicle_terms[t.vehicle_id].append((y[tid],-1.))
        row([(y[tid],1.),(g[t.vehicle_id],-1.)],-infinity,0.)
        ls={k:boolean(f'p:{tid}:{k}') for k in t.launches};rs={k:boolean(f'q:{tid}:{k}') for k in t.recoveries}
        launch_vars[tid]=ls;recovery_vars[tid]=rs
        row([(v,1.) for v in ls.values()]+[(y[tid],-1.)],0.,0.)
        row([(v,1.) for v in rs.values()]+[(y[tid],-1.)],0.,0.)
        duration=[(v,float(k)) for k,v in rs.items()]+[(v,-float(k)) for k,v in ls.items()]
        row(duration+[(y[tid],-t.min_duration)],0.,infinity)
        row(duration+[(y[tid],-t.max_duration)],-infinity,0.)
        completion=ceil(t.inspection_completion_minutes-1e-12);risk=towers[t.task_id].risk
        secondary[y[tid].index()]+=t.flight_minutes+weights['risk_weight']*risk*(1+.05*completion)
        # A chosen physical template has exactly one chosen endpoint pair.
        # Charge its addition once, then subtract twice for an old pair kept.
        secondary[y[tid].index()]+=churn*int(reference.present)
        power=weights['energy_weight']*t.base_mean/t.flight_minutes
        for k,v in ls.items():
            starts[(t.uav_id,k-p)].append((v,1.));secondary[v.index()]+=.05*weights['risk_weight']*risk*k-power*k
            for layer in range(k-p,k):
                key=(t.vehicle_id,t.launch_node,layer)
                row([(v,1.),(x[domain.waiting[key]],-1.)],-infinity,0.);operations[key].append((v,1.))
        for k,v in rs.items():
            ends[(t.uav_id,k+r+turn)].append((v,-1.));secondary[v.index()]+=power*k
            for layer in range(k,k+r):
                key=(t.vehicle_id,t.recovery_node,layer)
                row([(v,1.),(x[domain.waiting[key]],-1.)],-infinity,0.);operations[key].append((v,1.))
    for vid in g:
        if vid not in mandatory:row(vehicle_terms[vid]+[(g[vid],1.)],-infinity,0.)
    capacity={v.vehicle_id:v.operation_capacity for v in instance.physical_scenario.vehicles}
    for key in set(operations)|set(domain.fixed_operations):
        row(operations[key],-infinity,capacity[key[0]]-domain.fixed_operations.get(key,0))
    last=max([now]+[k for uid,k in ends]+[k+1 for uid,k in domain.fixed_busy])
    for uid in state.uavs:
        previous=None
        for k in range(now,last+1):
            busy=boolean(f'busy:{uid}:{k}')
            terms=[(busy,-1.)]+starts[(uid,k)]+ends[(uid,k)]
            if previous is not None:terms.append((previous,1.))
            row(terms,0.,0.)
            row([(busy,1.)],-infinity,1-domain.fixed_busy.get((uid,k),0));previous=busy
        row([(previous,1.)],0.,0.)
    missed={t:boolean('miss:'+str(t)) for t in domain.problem.task_ids}
    for task,v in missed.items():row(task_terms[task]+[(v,1.)],1.,1.)
    primary_terms=[(v,domain.problem.miss_penalty[t]) for t,v in missed.items()]
    group_hint={}
    if task_groups is not None:
        primary_terms,group_hint=task_groups.add_milp(boolean,row,chosen_vars,copy_records,infinity)
    # Exact old-copy correspondence needs a conjunction of its two endpoint
    # indicators. It does not require expanding any other implicit time pairs.
    offset=churn*(len(reference.old_arcs)+len(reference.old_services)) if reference.present else 0.
    for c in domain.previous_copies:
        if not reference.present or service_key(c) not in reference.old_services:continue
        if c.tuple_id not in chosen_vars and c.physical_candidate_id in launch_vars:
            ls=launch_vars[c.physical_candidate_id];rs=recovery_vars[c.physical_candidate_id]
            if c.launch_layer not in ls or c.recovery_layer not in rs:continue
            z=boolean('retained:'+c.tuple_id);a=ls[c.launch_layer];b=rs[c.recovery_layer]
            row([(z,1.),(a,-1.)],-infinity,0.);row([(z,1.),(b,-1.)],-infinity,0.)
            row([(z,1.),(a,-1.),(b,-1.)],-1.,infinity);secondary[z.index()]-=2*churn
    goal=solver.Objective()
    for v,coefficient in primary_terms:goal.SetCoefficient(v,coefficient)
    goal.SetMinimization();assembly=perf_counter()-tick
    names={getattr(pywraplp.Solver,n):n.lower() for n in ('OPTIMAL','FEASIBLE','INFEASIBLE','UNBOUNDED','ABNORMAL','NOT_SOLVED')}
    results=[];stages=[];hint_values=None;initial_result=None
    incumbent_hint=dict(status='not_provided',accepted=None,violation_count=0,
                        violations=[])
    if incumbent is not None:
        check_plan(domain,incumbent,dict(weights,plan_change_weight=0.))
        initial_values=account(incumbent)
        check_plan(cost_domain,incumbent,weights,objective(cost_domain,incumbent,weights))
        chosen_ids={c.tuple_id for c in incumbent.copies};arc_ids={a.arc_id for a in incumbent.arcs}
        names_to_values={v.name():0. for v in variables}
        for aid in x:names_to_values[x[aid].name()]=float(aid in arc_ids)
        active={a.vehicle_id for a in incumbent.arcs}
        for vid in g:names_to_values[g[vid].name()]=float(vid in active)
        for cid in chosen_vars:names_to_values[chosen_vars[cid].name()]=float(cid in chosen_ids)
        covered={t for c in incumbent.copies for t in c.task_ids}
        for task in missed:names_to_values[missed[task].name()]=float(task not in covered)
        if task_groups is not None:
            certificates=task_groups.evaluate(incumbent)['groups']
            for gid,(miss,windows) in group_hint.items():
                cert=certificates[gid]
                names_to_values[miss.name()]=float(not cert['complete'])
                if cert['complete'] and cert['time'] is not None:
                    for end,v in windows:names_to_values[v.name()]=float(end==cert['time'])
        names_to_values[maximum.name()]=float(max([0]+[a.arrival_layer for a in incumbent.arcs if a.head=='sink']+
            [k for c in incumbent.copies for t,k in c.completion_layers if t in state.remaining_tasks]))
        selected_keys={service_key(c) for c in incumbent.copies}
        for c in incumbent.copies:
            tid=c.physical_candidate_id
            if c.tuple_id in chosen_vars or tid not in y:continue
            names_to_values[y[tid].name()]=1.
            names_to_values[launch_vars[tid][c.launch_layer].name()]=1.
            names_to_values[recovery_vars[tid][c.recovery_layer].name()]=1.
        for c in domain.previous_copies:
            name='retained:'+c.tuple_id
            if name in names_to_values:names_to_values[name]=float(service_key(c) in selected_keys)
        for v in variables:
            if v.name().startswith('busy:'):
                _,uid,k=v.name().split(':');uid=int(uid);k=int(k)
                names_to_values[v.name()]=float(sum(c.uav_id==uid and max(now,c.launch_layer-p)<=k<c.recovery_layer+r+turn
                                                   for c in incumbent.copies))
        hint_values=[names_to_values[v.name()] for v in variables]
        violations=[]
        for row_index,(terms,lower,upper) in enumerate(hint_rows):
            value=sum(hint_values[i]*coefficient for i,coefficient in terms.items())
            if value<lower-1e-7 or value>upper+1e-7:
                violations.append(dict(
                    row_index=row_index,value=value,
                    lower=lower if isfinite(lower) else None,
                    upper=upper if isfinite(upper) else None,
                    terms=[(variables[i].name(),coefficient)
                           for i,coefficient in sorted(terms.items())]))
        if deadline is not None:deadline.check()
        if violations:
            incumbent_hint=dict(
                status='disabled_row_violation',accepted=False,
                violation_count=len(violations),violations=violations[:10])
            hint_values=None
        else:
            solver.SetHint(variables,hint_values)
            incumbent_hint=dict(
                status='accepted',accepted=True,violation_count=0,
                violations=[])
        initial_result=dict(plan=incumbent,primary=initial_values[0],secondary=initial_values[1],
                            validated_at=perf_counter(),source=(
                                'verified_incumbent_hint' if hint_values is not None
                                else 'verified_incumbent_without_solver_hint'))
        results.append(initial_result)
        assembly=perf_counter()-tick
    fixed_names=[]
    if released_vehicle_ids is not None:
        if incumbent is None or hint_values is None:
            raise ValueError('Vehicle-group fixing requires a complete valid incumbent assignment')
        release=set(released_vehicle_ids)
        if not release.issubset(state.vehicles):raise ValueError('Unknown released vehicle')
        owned={x[a.arc_id].name():a.vehicle_id for a in arcs}
        owned.update({v.name():vid for vid,v in g.items()})
        owned.update({chosen_vars[cid].name():c.vehicle_id for cid,c in copy_records.items()})
        for t in domain.templates:
            if t.template_id in y:
                owned[y[t.template_id].name()]=t.vehicle_id
                owned.update({v.name():t.vehicle_id for v in launch_vars[t.template_id].values()})
                owned.update({v.name():t.vehicle_id for v in recovery_vars[t.template_id].values()})
        for v in variables:
            if v.name() in owned and owned[v.name()] not in release:
                value=hint_values[v.index()]
                v.SetBounds(value,value)
                fixed_names.append(v.name())
        # Global auxiliaries stay constrained by all original physical rows.
    else:
        owned={}
    if model_observer is not None:
        model_observer(solver, variables, hint_values, owned, fixed_names)
    last_candidate=None
    def solve(stage,seconds):
        nonlocal last_candidate

        if deadline is not None:
            seconds=deadline.allowance(seconds,reserve=validation_reserve_s)
            if seconds<.001:raise BudgetExpired('Insufficient solver budget')
        solver.SetTimeLimit(max(1,int(seconds*1000)));start=perf_counter();raw=solver.Solve()
        bound=goal.BestBound();stages.append(dict(stage=stage,status=names[raw],raw_status=int(raw),
            bound=bound if isfinite(bound) else None,elapsed_s=perf_counter()-start,time_limit_s=seconds,
            validation_reserve_s=validation_reserve_s,solver_nodes=int(solver.nodes())))
        if raw not in (pywraplp.Solver.OPTIMAL,pywraplp.Solver.FEASIBLE):return raw,None
        if deadline is not None:deadline.check()
        copies=[copy_records[k] for k,v in chosen_vars.items() if v.solution_value()>.5]
        for t in domain.templates:
            if t.template_id not in y or y[t.template_id].solution_value()<.5:continue
            ls=[k for k,v in launch_vars[t.template_id].items() if v.solution_value()>.5]
            rs=[k for k,v in recovery_vars[t.template_id].items() if v.solution_value()>.5]
            if len(ls)!=1 or len(rs)!=1:raise ValueError('Invalid selected endpoint pair')
            copies.append(domain.materialize(t,ls[0],rs[0]))
        plan=Plan(tuple(a for a in arcs if x[a.arc_id].solution_value()>.5),tuple(copies))
        values=account(plan)
        check_plan(domain,plan,dict(weights,plan_change_weight=0.))
        check_plan(cost_domain,plan,weights,objective(cost_domain,plan,weights))
        if deadline is not None:deadline.check()
        expected=values[0] if stage=='primary' else values[1]
        if abs(goal.Value()-expected)>1e-6:
            exact_maximum=float(max([0]+[a.arrival_layer for a in plan.arcs if a.head=='sink']+
                [k for c in plan.copies for t,k in c.completion_layers if t in state.remaining_tasks]))
            rounded=[exact_maximum if v is maximum else float(round(v.solution_value())) for v in variables]
            projected_value=sum(goal.GetCoefficient(v)*rounded[i] for i,v in enumerate(variables))+(offset if stage=='secondary' else 0.)
            violations=[]
            for terms,lower,upper in hint_rows:
                value=sum(rounded[i]*coefficient for i,coefficient in terms.items())
                if value<lower-1e-7 or value>upper+1e-7:violations.append((value,lower if isfinite(lower) else None,upper if isfinite(upper) else None))
            evidence=dict(
                stage=stage,solver_value=goal.Value(),physical_value=expected,gap=goal.Value()-expected,
                makespan_variable=maximum.solution_value(),
                exact_makespan=exact_maximum,projected_linear_objective=projected_value,rounded_row_violations=violations,
                actual_return=max((a.arrival_layer for a in plan.arcs if a.head=='sink'),default=0),
                fractional=[(v.name(),v.solution_value()) for v in variables
                            if v is not maximum and abs(v.solution_value()-round(v.solution_value()))>1e-10],
                raw_status=int(raw))
            if not valid_makespan_projection(evidence):raise ObjectiveAccountingError(evidence,plan)
            stages[-1]['auxiliary_projection']=evidence
            stages[-1]['projection_certificate']='same_integer_schedule_all_rows_checked_continuous_epigraph_reconstructed'
            stages[-1]['projection_is_optimality_certificate']=False
        result=dict(plan=plan,primary=values[0],secondary=values[1],validated_at=perf_counter());results.append(result)
        last_candidate=result
        return raw,result
    exhausted=False
    try:
        if initial_result is not None and initial_result['primary']==0. and all(v>=0 for v in domain.problem.miss_penalty.values()):
            raw=pywraplp.Solver.OPTIMAL;first=initial_result
            stages.append(dict(stage='primary',status='optimal',raw_status=None,bound=0.,elapsed_s=0.,time_limit_s=0.,
                               proof='nonnegative_primary_and_row_verified_zero_witness'))
            hints=hint_values
        else:
            raw,first=solve('primary',primary_seconds)
            hints=[v.solution_value() for v in variables] if raw==pywraplp.Solver.OPTIMAL else None
        if raw==pywraplp.Solver.OPTIMAL and secondary_seconds>0:
            row(primary_terms,-infinity,first['primary']+1e-7)
            goal.Clear();secondary[maximum.index()]=weights['makespan_weight']
            for index,coefficient in secondary.items():goal.SetCoefficient(variables[index],coefficient)
            goal.SetOffset(offset);goal.SetMinimization();solver.SetHint(variables,hints)
            solve('secondary',secondary_seconds)
    except BudgetExpired:
        if not results:raise
        exhausted=True
    return dict(candidate=last_candidate, fixed_vehicle_variable_count=len(fixed_names),
                selected=min(results,key=lambda r:(r['primary'],r['secondary'])) if results else None,
                stages=stages,budget_exhausted=exhausted,variables=solver.NumVariables(),constraints=solver.NumConstraints(),
                incumbent_hint=incumbent_hint,
                assembly_elapsed_s=assembly,total_elapsed_s=perf_counter()-tick,explicit_copies=explicit,
                boundary='One external deadline includes assembly, both solves and validation. Backends may return late; late candidates are discarded and observed overrun must be reported.')
