"""Independent full road-domain completeness, embedding and result audit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from math import floor,isclose
from heapq import heappop,heappush
from statistics import mean,median
import json
from Agent.Run.publication_corrected_training import MANIFEST,CONFIG,read,write,digest
from Agent.Run.publication_timing_comparison import signature
from Agent.Run.evaluate_publication_rolling_roads import decode
from Agent.Run.publication_route_choice_model import RouteChoiceDomain
from Agent.Run.validate_publication_timing_comparison import check_schedule
from Agent.Tool.Benchmark.formal_v3_loader import FormalV3InstanceLoader
from Agent.Tool.Benchmark.benchmark_v3 import _formal_v3_weather
from Agent.Tool.Training.chdgrl_config import load_chdgrl_experiment_config

DATA=ROOT/'Data/Result/Diagnostic/publication_route_choice_20260910'
RUN=DATA/'run_v2'
OUT=ROOT/'Data/Paper/publication_route_choice_20260910'


def same(a,b):
    assert isclose(a,b,rel_tol=1e-9,abs_tol=1e-7),(a,b)


def complete_physical_network(instance,domain):
    """Enumerate all physically timed arcs independently, then dynamic-program
    reachability by increasing/decreasing time. No route-constructor functions.
    """
    edges=instance.physical_scenario.road.edges
    nodes={node for e in edges for node in (e.tail,e.head)}
    weather=_formal_v3_weather(instance.seed,256)
    physical_arcs=[]
    for edge in edges:
        if not edge.available:continue
        distance=sum(((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)**.5 for a,b in zip(edge.geometry.points,edge.geometry.points[1:]))
        speeds={k:edge.safe_speed_kmph/(60*(1+instance.road_beta_w*w.wind_speed+instance.road_beta_eta*w.uncertainty)) for k,w in weather.items()}
        for start in range(instance.rolling_state.current_layer,256):
            progress=0.
            for k in range(start,256):
                progress+=speeds[k]
                if progress+1e-12>=distance:
                    physical_arcs.append((edge.edge_id,edge.tail,start,edge.head,k+1));break
    expected_road=set();expected_wait=set();expected_terminal=set();expected_anchor=set()
    for vid,vehicle in domain.problem.rolling_state.vehicles.items():
        start=max(domain.problem.rolling_state.current_layer,vehicle.available_layer)
        end=min(256,start+floor(vehicle.remaining_duty_time+1e-9)) if vehicle.remaining_duty_time is not None else 256
        departing=defaultdict(list)
        for eid,tail,lo,head,hi in physical_arcs:
            if start<=lo<hi<=end:departing[lo].append((eid,tail,lo,head,hi))
        reached={(vehicle.node_id,start)}
        for k in range(start,end):
            reached.update((node,k+1) for node in nodes if (node,k) in reached)
            for eid,tail,lo,head,hi in departing[k]:
                if (tail,lo) in reached:reached.add((head,hi))
        returnable={('sink',k) for k in range(start,end+1)}
        for k in range(end-1,start-1,-1):
            returnable.update((node,k) for node in nodes if (node,k+1) in returnable)
            for eid,tail,lo,head,hi in departing[k]:
                if (head,hi) in returnable:returnable.add((tail,lo))
        if (vehicle.node_id,start) in returnable:expected_anchor.add((vid,'physical:'+vehicle.node_id,start))
        for k in range(start,end+1):
            if ('sink',k) in reached:expected_terminal.add((vid,k))
            if k==end:continue
            for node in nodes:
                if (node,k) in reached and (node,k+1) in returnable:expected_wait.add((vid,'physical:'+node,k))
            for eid,tail,lo,head,hi in departing[k]:
                if (tail,lo) in reached and (head,hi) in returnable:expected_road.add((vid,eid,lo,hi))
    observed_road=set();observed_wait=set();observed_terminal=set();observed_anchor=set()
    for arc in domain.problem.search_state.arcs:
        if arc.arc_type=='road':observed_road.add((arc.vehicle_id,domain.road_origins[arc.arc_id],arc.departure_layer,arc.arrival_layer))
        elif arc.arc_type=='wait':observed_wait.add((arc.vehicle_id,arc.tail,arc.departure_layer))
        elif arc.arc_type=='terminal':observed_terminal.add((arc.vehicle_id,arc.arrival_layer))
        elif arc.arc_type=='anchor':observed_anchor.add((arc.vehicle_id,arc.head,arc.arrival_layer))
        else:raise ValueError('Unrecognized arc class')
    assert observed_road==expected_road,'Missing or spurious physical road/time choices'
    assert observed_wait==expected_wait,'Missing or spurious wait choices'
    assert observed_terminal==expected_terminal,'Missing or spurious return opportunities'
    assert observed_anchor==expected_anchor,'Incorrect current-state anchors'
    return dict(roads=len(expected_road),waits=len(expected_wait),terminals=len(expected_terminal),anchors=len(expected_anchor))


def independent_conditional_cost(problem,selected_ids,weights):
    """Segment the route at required handling events and use heap shortest
    paths. This does not call the evaluated DAG/barrier route optimizer.
    """
    selected=[t for t in problem.search_state.tuples if t.tuple_id in set(selected_ids)]
    profiles={}
    for vid in sorted({t.vehicle_id for t in selected}):
        arcs=[a for a in problem.search_state.arcs if a.vehicle_id==vid]
        by_id={a.arc_id:a for a in arcs};graph=defaultdict(list)
        for a in arcs:
            tail=('source',-1) if a.tail=='source' else (a.tail,a.departure_layer)
            head=('return',a.arrival_layer) if a.head=='sink' else (a.head,a.arrival_layer)
            cost=a.secondary_cost+weights['time_weight']*a.duty_time
            assert cost>=0
            graph[tail].append((head,cost))
        def distances(start,target=None):
            values={start:0.};heap=[(0.,start)]
            while heap:
                cost,node=heappop(heap)
                if cost>values[node]+1e-10:continue
                if node==target:return values
                for head,weight in graph[node]:
                    if target is not None and head[1]>target[1]:continue
                    candidate=cost+weight
                    if candidate<values.get(head,float('inf'))-1e-10:
                        values[head]=candidate;heappush(heap,(candidate,head))
            return values
        required={k for t in selected if t.vehicle_id==vid for k in t.required_arc_ids}
        sequence=sorted((by_id[k] for k in required),key=lambda a:(a.departure_layer,a.arc_id))
        current=('source',-1);prefix=0.
        for a in sequence:
            target=(a.tail,a.departure_layer);segment=distances(current,target)
            assert target in segment
            prefix+=segment[target]+a.secondary_cost+weights['time_weight']*a.duty_time
            current=(a.head,a.arrival_layer)
        profile=distances(current)
        profiles[vid]={time:prefix+cost for (name,time),cost in profile.items() if name=='return'}
        assert profiles[vid]
    if not selected:return 0.
    completion=max(k for t in selected for _,k in t.completion_layers)
    best=float('inf')
    for cap in range(completion,257):
        total=weights['makespan_weight']*cap
        for profile in profiles.values():
            total+=min((cost for time,cost in profile.items() if time<=cap),default=float('inf'))
        best=min(best,total)
    return best


def main():
    contract=read(RUN/'contract.json');analysis=read(OUT/'analysis.json')
    assert read(RUN/'progress.json')==dict(status='complete',completed=72,total=72)
    assert digest(DATA/'contract.json')==contract['parent_contract_sha256']
    assert digest(ROOT/'Agent/Run/evaluate_publication_route_choice_v2.py')==contract['runner_sha256']
    assert digest(ROOT/'Agent/Run/publication_solver_observation.py')==contract['observer_sha256']
    assert digest(RUN/'contract.json')==analysis['contract_sha256']
    for name,expected in contract['domain_files'].items():assert digest(ROOT/name)==expected,name
    for name,expected in contract['source_hashes'].items():assert digest(ROOT/name)==expected,name
    for name,expected in contract['preserved_manuscript'].items():assert digest(ROOT/name)==expected,name
    loader=FormalV3InstanceLoader(MANIFEST);config=load_chdgrl_experiment_config(CONFIG)
    domains={};payloads={};counts=Counter();spatial=Counter();hashes={}
    for path in sorted((DATA/'domains').glob('*.json')):
        payload=read(path);key=payload['instance_id'];payloads[key]=payload;assert loader.split_for(key)=='train'
        domain=RouteChoiceDomain(**{k:decode(v) for k,v in payload['domain']['fields'].items()})
        assert signature(domain)==payload['domain_hash'];domains[key]=domain
        counts.update(complete_physical_network(loader(key),domain))
        for row in payload['structure']:
            counts['vehicle_domains']+=1
            spatial['with_spatial_branching' if row['spatial_branch_time_nodes'] else 'without_spatial_branching']+=1
        old_arcs=domain.fixed_problem.search_state.arcs
        degrees=Counter((a.vehicle_id,a.tail,a.departure_layer) for a in old_arcs)
        assert max(degrees.values())==1,'Old domain is not a fixed chain'
        old,mapped=payload['old_certified_reference'],payload['embedded_reference']
        for problem,witness in ((domain.fixed_problem,old),(domain.problem,mapped)):
            e,p,_=check_schedule(problem,witness,config.search.objective_weights);assert not e and not p,(key,e,p)
            counts['verified_schedules']+=1
        same(old['primary'],mapped['primary']);same(old['secondary'],mapped['secondary'])
        hashes[str(path.relative_to(ROOT))]=digest(path)
    assert sorted(domains)==contract['instances']
    rows=[read(p) for p in (RUN/'runs').glob('*.json')]
    assert {signature(r['job']) for r in rows}=={signature(j) for j in contract['jobs']}
    assert len(rows)==72
    for row in rows:
        assert row['contract_hash']==signature(contract)
        d=domains[row['job']['instance_id']];p=d.problem if row['job']['domain']=='route_choice' else d.fixed_problem
        assert signature(p)==row['problem_hash']
        assert row['known_feasible_embedding_available']
        status_names={0:'optimal',1:'feasible',2:'infeasible',3:'unbounded',4:'abnormal',6:'not_solved'}
        for stage in row['stages']:
            assert stage['status']==status_names.get(stage['raw_status'],'other')
        assert row['primary_status']==row['stages'][0]['status']
        assert row['secondary_status']==(row['stages'][1]['status'] if len(row['stages'])>1 else 'not_solved')
        if row['validation_ok']:
            e,f,_=check_schedule(p,row['snapshot'],config.search.objective_weights);assert not e and not f
            counts['verified_schedules']+=1
        else:
            counts['unsuccessful_solver_returns']+=1
            assert row['snapshot']['primary'] is None and row['snapshot']['secondary'] is None
        if row['primary_status']!=row['legacy_wrapper_primary_status']:
            counts['legacy_primary_status_disagreements']+=1
    gains=[];secondary=[]
    for key in domains:
        old=next(r for r in rows if r['job']==dict(instance_id=key,domain='fixed_tour'))
        new=next(r for r in rows if r['job']==dict(instance_id=key,domain='route_choice'))
        reported=next(p for p in analysis['pairs'] if p['instance_id']==key)
        certified=old['primary_status']==new['primary_status']=='optimal' and old['validation_ok'] and new['validation_ok']
        assert reported['primary_certified']==certified
        if certified:
            delta=old['snapshot']['primary']-new['snapshot']['primary'];assert delta>=-1e-7
            pp=100*delta/old['total_weight'];same(pp,reported['primary_improvement_pp']);gains.append(pp)
            if abs(delta)<1e-7 and old['secondary_status']==new['secondary_status']=='optimal':
                value=old['snapshot']['secondary']-new['snapshot']['secondary'];assert value>=-1e-7
                same(value,reported['secondary_improvement']);secondary.append(value)
        else:
            assert reported['primary_improvement_pp'] is None and reported['secondary_improvement'] is None
    assert len(gains)==analysis['primary_certified_pairs']
    assert sum(g>1e-7 for g in gains)==analysis['primary_wins']
    assert len(secondary)==analysis['secondary_certified_pairs']
    for kind in ('fixed_tour','route_choice'):
        subset=[r for r in rows if r['job']['domain']==kind]
        assert dict(Counter(r['primary_status'] for r in subset))==analysis['primary_statuses'][kind]
        same(median(r['whole_call_elapsed_s'] for r in subset),analysis['whole_call_median_s'][kind])
    if gains:same(median(gains),analysis['median_primary_improvement_pp']);same(mean(gains),analysis['mean_primary_improvement_pp'])
    if secondary:same(median(secondary),analysis['median_secondary_improvement'])
    review=read(OUT/'model_difference.json');witness=read(OUT/'revisit_witness.json')
    assert witness['primary_optima']=={'True':1.,'False':0.}
    for source in review['sources'].values():assert digest(Path(source['path']))==source['sha256']
    certificate=read(OUT/'common_tour_certificate.json')
    assert certificate['instances']==36 and certificate['primary_effect_identifiable'] is False
    assert digest(ROOT/'Agent/Run/diagnose_publication_common_tour.py')==certificate['script_sha256']
    for row in rows:
        if row['primary_status']=='optimal' and row['validation_ok']:
            certified=next(r for r in certificate['rows'] if r['instance_id']==row['job']['instance_id'])
            same(row['snapshot']['primary'],certified['expanded_primary_by_projection'])
    conditional=read(OUT/'conditional_route_analysis.json')
    cond_contract=read(DATA/'conditional_routes/contract.json')
    assert digest(DATA/'conditional_routes/contract.json')==conditional['contract_sha256']
    assert digest(ROOT/'Agent/Run/optimize_publication_fixed_services.py')==cond_contract['script_sha256']
    assert digest(CONFIG)==cond_contract['config_sha256']
    assert digest(OUT/'common_tour_certificate.json')==cond_contract['primary_certificate_sha256']
    assert {r['instance_id'] for r in conditional['rows']}==set(domains)
    conditional_gains=[];conditional_percent=[];mechanisms=[]
    for row in conditional['rows']:
        key=row['instance_id'];domain=domains[key];snapshot=row['snapshot']
        e,p,_=check_schedule(domain.problem,snapshot,config.search.objective_weights);assert not e and not p
        counts['verified_schedules']+=1
        stored=read(DATA/'conditional_routes/runs'/(signature(key)[:24]+'.json'))
        assert stored==row and row['contract_hash']==signature(cond_contract)
        old=payloads[key]['old_certified_reference']
        assert old['selected_tuple_ids']==snapshot['selected_tuple_ids']
        same(old['primary'],snapshot['primary']);same(old['secondary'],row['old_secondary'])
        cost=independent_conditional_cost(domain.problem,snapshot['selected_tuple_ids'],config.search.objective_weights)
        same(cost,row['details']['conditional_route_cost'])
        chosen=[t for t in domain.problem.search_state.tuples if t.tuple_id in set(snapshot['selected_tuple_ids'])]
        weights=config.search.objective_weights
        service_cost=sum(t.secondary_cost+weights['risk_weight']*t.risk_exposure+weights['energy_weight']*t.expected_energy for t in chosen)
        same(cost+service_cost,snapshot['secondary'])
        gain=old['secondary']-snapshot['secondary'];assert gain>=-1e-7
        same(gain,row['secondary_improvement']);same(100*gain/old['secondary'],row['secondary_improvement_percent'])
        old_path=[a for a in domain.fixed_problem.search_state.arcs if a.arc_id in set(old['selected_arc_ids'])]
        new_path=[a for a in domain.problem.search_state.arcs if a.arc_id in set(snapshot['selected_arc_ids'])]
        old_distance=sum(a.secondary_cost for a in old_path);new_distance=sum(a.secondary_cost for a in new_path)
        old_duty=sum(a.duty_time for a in old_path);new_duty=sum(a.duty_time for a in new_path)
        completion=[k for t in chosen for _,k in t.completion_layers]
        old_makespan=max(completion+[a.arrival_layer for a in old_path if a.head=='sink'],default=0)
        new_makespan=max(completion+[a.arrival_layer for a in new_path if a.head=='sink'],default=0)
        old_road_sequence=defaultdict(list);new_road_sequence=defaultdict(list)
        expanded_by_id={a.arc_id:a for a in domain.problem.search_state.arcs}
        for a in sorted(old_path,key=lambda a:(a.departure_layer,a.arc_id)):
            for aid in domain.arc_expansion[a.arc_id]:
                if expanded_by_id[aid].arc_type=='road':old_road_sequence[a.vehicle_id].append(domain.road_origins[aid])
        for a in sorted(new_path,key=lambda a:(a.departure_layer,a.arc_id)):
            if a.arc_type=='road':new_road_sequence[a.vehicle_id].append(domain.road_origins[a.arc_id])
        distance_gain=old_distance-new_distance
        duty_gain=weights['time_weight']*(old_duty-new_duty)
        makespan_gain=weights['makespan_weight']*(old_makespan-new_makespan)
        same(distance_gain+duty_gain+makespan_gain,gain)
        mechanisms.append(dict(instance_id=key,changed_physical_road_sequence=old_road_sequence!=new_road_sequence,
            distance_cost_reduction=distance_gain,duty_reduction=old_duty-new_duty,makespan_reduction=old_makespan-new_makespan,
            weighted_duty_reduction=duty_gain,weighted_makespan_reduction=makespan_gain,secondary_reduction=gain,
            old_physical_road_sequences=dict(old_road_sequence),new_physical_road_sequences=dict(new_road_sequence)))
        conditional_gains.append(gain);conditional_percent.append(row['secondary_improvement_percent'])
        counts['independent_conditional_optima']+=1
    assert sum(v>1e-7 for v in conditional_gains)==conditional['strict_secondary_improvements']
    same(median(conditional_gains),conditional['median_secondary_improvement'])
    same(median(conditional_percent),conditional['median_secondary_improvement_percent'])
    write(OUT/'conditional_route_mechanisms.json',dict(status='objective_components_independently_recomputed',
        instances=36,changed_road_sequence_instances=sum(r['changed_physical_road_sequence'] for r in mechanisms),
        positive_distance_reduction_instances=sum(r['distance_cost_reduction']>1e-7 for r in mechanisms),
        unchanged_route_distance_instances=sum(abs(r['distance_cost_reduction'])<1e-7 for r in mechanisms),
        median_distance_reduction=median(r['distance_cost_reduction'] for r in mechanisms),
        median_duty_reduction=median(r['duty_reduction'] for r in mechanisms),
        median_makespan_reduction=median(r['makespan_reduction'] for r in mechanisms),rows=mechanisms))
    result=dict(status='independent_model_and_result_checks_passed',publication_ready=False,
        counts=dict(counts),spatial_vehicle_domains=dict(spatial),paired_instances=36,solver_calls=72,
        domain_hashes=hashes,analysis_sha256=digest(OUT/'analysis.json'),model_difference_sha256=digest(OUT/'model_difference.json'),
        revisit_sha256=digest(OUT/'revisit_witness.json'),validator_sha256=digest(Path(__file__)),
        common_tour_certificate_sha256=digest(OUT/'common_tour_certificate.json'),
        conditional_analysis_sha256=digest(OUT/'conditional_route_analysis.json'),
        scientific_scope='Complete physical road/time alternatives verified within the static same-copy comparison. New copy generation, locked dynamic states, matched full paper1 quantitative comparison, learning and fresh test evidence remain open.')
    write(OUT/'validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='domain_hashes'},indent=2),flush=True)


if __name__=='__main__':main()
