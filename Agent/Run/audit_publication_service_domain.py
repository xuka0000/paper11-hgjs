"""Independent geometry/resource/route and scientific-summary verification.

Reads saved outputs, does not call the new constructor or either new MILP.
Selected copies are re-derived from physical inputs. Sampled copies receive
independent segmented shortest-path feasibility checks on the previously
verified complete physical graph. Counts are not misreported as exhaustive
validation of every implicit time pair.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from math import sqrt,ceil,isfinite,isclose
from statistics import median
import json
import xml.etree.ElementTree as ET
from Agent.Run.publication_corrected_training import read,write,digest,MANIFEST,CONFIG
from Agent.Run.publication_timing_comparison import signature
from Agent.Run.evaluate_publication_rolling_roads import decode
from Agent.Run.validate_publication_timing_comparison import check_schedule
from Agent.Run.audit_publication_route_choice_results import independent_conditional_cost
from Agent.Tool.Benchmark.formal_v3_loader import FormalV3InstanceLoader
from Agent.Tool.Training.chdgrl_config import load_chdgrl_experiment_config
from Mask.ServiceTuple import SearchTuple,SearchArc,SearchState,ServiceAdmissionChecks
from Mask.TupleFactory import EnergyMoments,build_energy_certificate

DATA=ROOT/'Data/Result/Diagnostic/publication_service_domain_20260910'
OUT=ROOT/'Data/Paper/publication_service_domain_20260910'
PARENT=ROOT/'Data/Result/Diagnostic/publication_route_choice_20260910'


def same(a,b):
    assert isclose(a,b,rel_tol=1e-9,abs_tol=1e-7),(a,b)


def restore_copy(payload):
    values=dict(payload)
    for name in ('task_ids','required_arc_ids','interval_keys','operation_keys'):values[name]=tuple(values[name])
    values['completion_layers']=tuple(tuple(row) for row in values['completion_layers'])
    values['admission_checks']=ServiceAdmissionChecks(**values['admission_checks'])
    return SearchTuple(**values)


def verify_copy(instance,copy):
    assert len(copy.task_ids)==1 and copy.task_ids[0] in instance.rolling_state.remaining_tasks
    tower=next(t for t in instance.physical_scenario.power.towers if int(t.tower_id)==copy.task_ids[0])
    uav=next(u for u in instance.physical_scenario.uavs if u.uav_id==copy.uav_id)
    stops={'physical:'+s.stop_id:s for s in instance.physical_scenario.stops}
    launch=stops[copy.launch_node];recovery=stops[copy.recovery_node]
    assert copy.vehicle_id==uav.owner_vehicle_id and copy.admitted and not copy.is_air_residual
    lengths=[sum((x-y)**2 for x,y in zip((a.x,a.y,a.z),(b.x,b.y,b.z)))**.5
             for a,b in ((launch.location,tower.location),(tower.location,recovery.location))]
    duration=sum(lengths)*60/uav.cruise_speed_kmph+tower.service_minutes
    offset=lengths[0]*60/uav.cruise_speed_kmph+tower.service_minutes
    elapsed=copy.recovery_layer-copy.launch_layer
    assert elapsed>=ceil(duration-1e-12)
    noise=.2*int.from_bytes(sha256(f'{instance.seed}|{tower.tower_id}|{uav.uav_id}|service-energy-v1'.encode()).digest()[:8],'big')/(2**64)
    mean=(.42*sum(lengths)+.16*tower.service_minutes+noise)*elapsed/duration
    std=.08*mean;bound=mean+sqrt(instance.energy_alpha/(1-instance.energy_alpha))*std
    same(mean,copy.expected_energy);same(std,copy.energy_std);same(bound,copy.reliable_energy)
    same(copy.battery_capacity,uav.battery_capacity);same(copy.reserve_ratio,uav.reserve_ratio)
    assert bound<=uav.battery_capacity*(1-uav.reserve_ratio)+1e-8
    completion=copy.launch_layer+ceil(offset-1e-12)
    assert copy.completion_layers==((int(tower.tower_id),completion),)
    same(copy.secondary_cost,duration);same(copy.risk_exposure,tower.risk*(1+.05*completion))
    identity=f'{tower.tower_id}|{uav.uav_id}|{launch.stop_id}|{recovery.stop_id}'
    expected='physical-service:'+sha256(identity.encode()).hexdigest()[:20]
    assert copy.physical_candidate_id==expected
    assert copy.tuple_id==f'service:{expected}:{copy.launch_layer}:{copy.recovery_layer}'
    p=instance.preparation_layers;r=instance.recovery_service_layers;t=instance.turnaround_layers
    assert copy.launch_layer-p>=max(instance.rolling_state.current_layer,instance.rolling_state.uavs[uav.uav_id].available_layer)
    intervals=tuple(f'uav:{uav.uav_id}:{k}' for k in range(copy.launch_layer-p,copy.recovery_layer+r+t))
    operations=tuple(f'op:{copy.vehicle_id}:{node}:{k}' for node,lo,hi in
        ((copy.launch_node,copy.launch_layer-p,copy.launch_layer),(copy.recovery_node,copy.recovery_layer,copy.recovery_layer+r)) for k in range(lo,hi))
    assert copy.interval_keys==intervals and copy.operation_keys==operations
    required=tuple(f'route_wait:{copy.vehicle_id}:{node.removeprefix("physical:")}:{k}' for node,lo,hi in
        ((copy.launch_node,copy.launch_layer-p,copy.launch_layer),(copy.recovery_node,copy.recovery_layer,copy.recovery_layer+r)) for k in range(lo,hi))
    assert copy.required_arc_ids==required


def verify_declared_copy(instance, copy):
    """Independently verify a rolling copy against its declared candidate."""
    assert len(copy.task_ids) == 1
    assert copy.task_ids[0] in instance.rolling_state.remaining_tasks
    candidate = next(
        item
        for item in instance.physical_scenario.candidates
        if item.candidate_id == copy.physical_candidate_id
    )
    tower = next(
        item
        for item in instance.physical_scenario.power.towers
        if int(item.tower_id) == copy.task_ids[0]
    )
    uav = next(
        item
        for item in instance.physical_scenario.uavs
        if item.uav_id == copy.uav_id
    )
    assert candidate.admitted
    assert candidate.task_ids == tuple(str(item) for item in copy.task_ids)
    assert candidate.vehicle_id == copy.vehicle_id == uav.owner_vehicle_id
    assert candidate.uav_id == copy.uav_id
    assert copy.launch_node == "physical:" + candidate.launch_stop_id
    assert copy.recovery_node == "physical:" + candidate.recovery_stop_id
    assert copy.admitted and not copy.is_air_residual
    duration = float(candidate.duration_minutes)
    completion_offset = float(candidate.completion_offsets_minutes[0][1])
    elapsed = copy.recovery_layer - copy.launch_layer
    assert elapsed >= ceil(duration - 1e-12)
    ratio = elapsed / duration
    certificate = build_energy_certificate(
        EnergyMoments(
            float(candidate.expected_energy) * ratio,
            float(candidate.energy_std) * ratio,
        ),
        instance.energy_alpha,
        uav.battery_capacity,
        uav.reserve_ratio,
    )
    assert certificate.admitted
    same(certificate.mean, copy.expected_energy)
    same(certificate.std, copy.energy_std)
    same(certificate.reliable_bound, copy.reliable_energy)
    same(copy.battery_capacity, uav.battery_capacity)
    same(copy.reserve_ratio, uav.reserve_ratio)
    assert copy.admission_certificate == certificate.certificate_id
    completion = copy.launch_layer + ceil(completion_offset - 1e-12)
    assert copy.completion_layers == ((int(tower.tower_id), completion),)
    same(copy.secondary_cost, duration)
    same(copy.risk_exposure, tower.risk * (1.0 + 0.05 * completion))
    assert copy.tuple_id == (
        f"service:{candidate.candidate_id}:"
        f"{copy.launch_layer}:{copy.recovery_layer}"
    )
    p = instance.preparation_layers
    r = instance.recovery_service_layers
    t = instance.turnaround_layers
    assert copy.launch_layer - p >= max(
        instance.rolling_state.current_layer,
        instance.rolling_state.uavs[uav.uav_id].available_layer,
    )
    intervals = tuple(
        f"uav:{uav.uav_id}:{layer}"
        for layer in range(
            copy.launch_layer - p,
            copy.recovery_layer + r + t,
        )
    )
    operations = tuple(
        f"op:{copy.vehicle_id}:{node}:{layer}"
        for node, lower, upper in (
            (copy.launch_node, copy.launch_layer - p, copy.launch_layer),
            (copy.recovery_node, copy.recovery_layer, copy.recovery_layer + r),
        )
        for layer in range(lower, upper)
    )
    assert copy.interval_keys == intervals
    assert copy.operation_keys == operations
    required = tuple(
        f"route_wait:{copy.vehicle_id}:"
        f"{node.removeprefix('physical:')}:{layer}"
        for node, lower, upper in (
            (copy.launch_node, copy.launch_layer - p, copy.launch_layer),
            (copy.recovery_node, copy.recovery_layer, copy.recovery_layer + r),
        )
        for layer in range(lower, upper)
    )
    assert copy.required_arc_ids == required


def materialized_problem(instance,parent,copies):
    capacities={v.vehicle_id:v.operation_capacity for v in instance.physical_scenario.vehicles}
    return replace(parent,search_state=SearchState(parent.search_state.arcs,tuple(copies)),
        operation_capacity={k:capacities[c.vehicle_id] for c in copies for k in c.operation_keys})


def independent_fixed_copy_count(instance,parent,allowed):
    # Enumerate physical stop/time pairs directly on the complete fixed path.
    # This does not read the new catalogue's templates or time-factor arrays.
    waits={(a.vehicle_id,a.tail,a.departure_layer) for a in parent.search_state.arcs if a.arc_id in allowed and a.arc_type=='wait'}
    stops={'physical:'+s.stop_id:s for s in instance.physical_scenario.stops}
    p=instance.preparation_layers;r=instance.recovery_service_layers;count=0
    for uav in instance.physical_scenario.uavs:
        vid=uav.owner_vehicle_id
        launches={(node,k+p) for v,node,k in waits if v==vid and k>=instance.rolling_state.uavs[uav.uav_id].available_layer
                  and all((vid,node,j) in waits for j in range(k,k+p))}
        recoveries={(node,k) for v,node,k in waits if v==vid and all((vid,node,j) in waits for j in range(k,k+r))}
        for tower in instance.physical_scenario.power.towers:
            noise=.2*int.from_bytes(sha256(f'{instance.seed}|{tower.tower_id}|{uav.uav_id}|service-energy-v1'.encode()).digest()[:8],'big')/(2**64)
            distances={node:sum((a-b)**2 for a,b in zip((stop.location.x,stop.location.y,stop.location.z),
                (tower.location.x,tower.location.y,tower.location.z)))**.5 for node,stop in stops.items()}
            for a,l in launches:
                for b,end in recoveries:
                    distance=distances[a]+distances[b]
                    duration=60*distance/uav.cruise_speed_kmph+tower.service_minutes
                    if end-l<ceil(duration-1e-12):continue
                    mean=(.42*distance+.16*tower.service_minutes+noise)*(end-l)/duration
                    bound=mean*(1+.08*sqrt(instance.energy_alpha/(1-instance.energy_alpha)))
                    if bound<=uav.battery_capacity*(1-uav.reserve_ratio):count+=1
    return count


def main():
    contract=read(DATA/'contract.json');refined_contract=read(DATA/'reference_refinement/contract.json')
    analysis=read(OUT/'analysis.json');refined=read(OUT/'refined_analysis.json')
    assert read(DATA/'progress.json')==read(DATA/'reference_refinement/progress.json')==dict(status='complete',completed=36,total=36)
    assert digest(DATA/'contract.json')==analysis['contract_sha256']==refined_contract['parent_contract_sha256']
    assert digest(DATA/'reference_refinement/contract.json')==refined['contract_sha256']
    for c in (contract,refined_contract):
        for name,expected in c['sources'].items():assert digest(ROOT/name)==expected,name
    assert digest(MANIFEST)==contract['manifest_sha256'] and digest(CONFIG)==contract['config_sha256']
    assert digest(OUT/'tests.xml')==contract['tests_sha256']
    assert digest(OUT/'fixed_timetable_tests.xml')==refined_contract['test_sha256']
    for name,number in (('tests.xml',8),('fixed_timetable_tests.xml',2)):
        suites=ET.parse(OUT/name).getroot().findall('.//testsuite')
        assert sum(int(s.attrib['tests']) for s in suites)==number
        assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    loader=FormalV3InstanceLoader(MANIFEST);weights=load_chdgrl_experiment_config(CONFIG).search.objective_weights
    counts=Counter();rows=[];hashes={}
    for key in contract['instances']:
        assert loader.split_for(key)=='train';instance=loader(key)
        original_path=DATA/'instances'/(signature(key)[:24]+'.json');new_path=DATA/'reference_refinement/instances'/(signature(key)[:24]+'.json')
        old=read(original_path);new=read(new_path);rows.append(new)
        assert old['arms']['fixed_timetable']['primary_optimal']==(old['fixed_timetable_solver']['stages'][0]['status']=='optimal')
        assert new['refinement']['witness']['primary_optimal']==(new['refinement']['status']=='optimal' or new['refinement']['witness']['snapshot']['primary']<1e-8)
        assert new['arms']['fixed_timetable']['primary_optimal']==(old['arms']['fixed_timetable']['primary_optimal'] or new['refinement']['witness']['primary_optimal'])
        assert old['contract_hash']==signature(contract) and new['contract_hash']==signature(refined_contract)
        assert digest(original_path)==new['original_sha256']==refined_contract['original_rows'][key]
        parent_path=PARENT/'domains'/(signature(key)[:24]+'.json');payload=read(parent_path)
        assert digest(parent_path)==old['parent_road_domain_sha256']
        parent=decode(payload['domain']['fields']['problem']);expansion=decode(payload['domain']['fields']['arc_expansion'])
        origins=decode(payload['domain']['fields']['road_origins']);arc_map={a.arc_id:a for a in parent.search_state.arcs}
        assert signature(parent.search_state.arcs)==old['full_road_signature']
        fixed_timetable={k for values in expansion.values() for k in values}
        refs={int(vid):set(edges) for vid,edges in old['reference_physical_edges'].items()}
        vehicle_nodes={vid:{'physical:'+n for e in instance.physical_scenario.road.edges if e.edge_id in edges for n in (e.tail,e.head)} for vid,edges in refs.items()}
        fixed_geometry={a.arc_id for a in parent.search_state.arcs if a.arc_type in ('anchor','terminal') or
            a.arc_type=='road' and origins[a.arc_id] in refs[a.vehicle_id] or a.arc_type=='wait' and a.tail in vehicle_nodes[a.vehicle_id]}
        assert signature(sorted(fixed_timetable))==old['allowed_arc_hashes']['fixed_timetable']
        assert signature(sorted(fixed_geometry))==old['allowed_arc_hashes']['fixed_path']
        complete_fixed_count=independent_fixed_copy_count(instance,parent,fixed_timetable)
        assert complete_fixed_count==new['refinement']['explicit_fixed_time_copies'],(key,complete_fixed_count,new['refinement']['explicit_fixed_time_copies'])
        assert new['refinement']['variables']==complete_fixed_count+len(instance.task_ids)
        counts['independently_enumerated_fixed_timetable_copies']+=complete_fixed_count
        for mode,record in [(k,v) for item in (old,new) for k,v in item['arms'].items()]+[('fixed_timetable',new['refinement']['witness'])]:
            copies=[restore_copy(c) for c in record['copies']]
            for copy in copies:verify_copy(instance,copy);counts['physical_copy_observations']+=1
            for a in record['arcs']:assert SearchArc(**a)==arc_map[a['arc_id']]
            selected=set(record['snapshot']['selected_arc_ids'])
            if mode=='fixed_timetable':assert selected<=fixed_timetable
            elif mode=='fixed_path_flexible_times':assert selected<=fixed_geometry
            problem=materialized_problem(instance,parent,copies)
            e,p,_=check_schedule(problem,record['snapshot'],weights,instance.preparation_layers,instance.recovery_service_layers,instance.turnaround_layers)
            assert not e and not p,(key,mode,e,p)
            assert {tuple(a) for a in record['assignments']}=={(c.physical_candidate_id,c.launch_layer,c.recovery_layer) for c in copies}
            if mode!='fixed_timetable':assert record['primary_optimal']==(record['snapshot']['primary']<1e-8)
            counts['schedule_observations']+=1
        for sampled in old['sampled_copies']:
            copy=restore_copy(sampled['copy']);verify_copy(instance,copy)
            problem=materialized_problem(instance,parent,[copy])
            cost=independent_conditional_cost(problem,[copy.tuple_id],weights)
            assert isfinite(cost)
            counts['sampled_copy_route_certificates']+=1
        counts['declared_templates']+=old['statistics']['retained_templates']
        counts['declared_implicit_copies']+=old['statistics']['implicit_feasible_time_copies']
        for path in (original_path,new_path):hashes[path.relative_to(ROOT).as_posix()]=digest(path)
        print(f'INDEPENDENT SERVICE DOMAIN {len(rows)}/36',flush=True)
    assert counts['declared_templates']==analysis['physical_templates']
    assert counts['declared_implicit_copies']==analysis['implicit_time_copies']
    for arm in contract['arms']:
        assert sum(r['arms'][arm]['primary_optimal'] for r in rows)==refined['primary_certified'][arm]
        assert sum(r['arms'][arm]['snapshot']['primary']<1e-8 for r in rows)==refined['zero_missed_instances'][arm]
    gains=[r['arms']['fixed_timetable']['snapshot']['primary']-r['arms']['fixed_path_flexible_times']['snapshot']['primary'] for r in rows]
    assert all(v>=-1e-7 for v in gains)
    same(median(gains),refined['median_fixed_to_flexible_primary_reduction'])
    assert sum(r['arms']['fixed_timetable']['primary_optimal'] and gain>1e-7 for r,gain in zip(rows,gains))==refined['certified_timing_gains']
    assert dict(Counter(r['refinement']['status'] for r in rows))==refined['refined_solver_statuses']
    mechanisms=read(OUT/'mechanism_validation.json')
    assert mechanisms['primary_optima']==dict(fixed_path_A=2.,fixed_path_B=1.,fixed_timetable_B=2.,free_roads=1.)
    assert mechanisms['implicit_time_copies']==mechanisms['explicitly_enumerated_copies']
    for name,expected in mechanisms['sources'].items():assert digest(ROOT/name)==expected
    result=dict(status='independent_geometry_resources_and_comparison_checks_passed',publication_ready=False,
        counts=dict(counts),scoped_tests_passed=10,instances=36,test_instances_used=0,
        analysis_sha256=digest(OUT/'analysis.json'),refined_analysis_sha256=digest(OUT/'refined_analysis.json'),
        mechanism_validation_sha256=digest(OUT/'mechanism_validation.json'),
        data_hashes=hashes,validator_sha256=digest(Path(__file__)),
        boundary='Complete-copy equivalence tested on the constructed small domain; all retained scheduling observations verified independently;24 deterministic sampled copies per native instance receive independent geometric and carrier-route feasibility checks. The implicit pair count is a constructor count, not a count of exhaustively inspected copies.')
    write(OUT/'validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='data_hashes'},indent=2),flush=True)


if __name__=='__main__':main()
