"""Controlled long operations on declared staged roads, using frozen road mechanics.

This is a synthetic mechanism panel. Its bypasses, service attributes and energy
coefficients are specified inputs, not observations of an operating fleet.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from math import sqrt,ceil
from random import Random
from types import SimpleNamespace

from Agent.Run.publication_rolling_roads import RoadMemory,build_rolling_problem,physical_entries,SegmentClosure
from Agent.Tool.Benchmark.benchmark_v3 import FormalV3Instance
from Agent.Tool.Evaluation.dynamic_evaluation import validate_dynamic_plan
from Agent.Tool.Evaluation.method_runner import run_method
from Env.Execution import record_plan,execute_due_events,advance_state,PlannedVehicleArcEvent
from Env.State import RollingState
from Mask.Geometry import Point3D,Polyline3D
from Mask.PowerNetwork import Tower,PowerLine,PowerNetwork
from Mask.Scenario import RoadNode,RoadEdge,RoadNetwork,CandidateStop,VehicleSpec,UavSpec,PhysicalServiceCandidate,ScenarioSpec,IrregularDualNetworkScenario
from Mask.ServiceTuple import SearchArc,SearchTuple,SearchState
from Mask.Vehicle import VehicleState
from Mask.UAV import UavState

LAYERS=tuple(range(1,250,8))
TERMINAL=256
SETTINGS=tuple((scale,height,fleet) for scale in (1.,1.05,1.1) for height in (1.5,4.) for fleet in (1,2))


def generate(split,index):
    if split not in ('validation','test') or not 0<=index<len(SETTINGS):raise ValueError('Unknown registered case')
    scale,height,fleet=SETTINGS[index];seed=(2026090900 if split=='validation' else 2026091000)+index
    rng=Random(seed);scale*=rng.uniform(.995,1.005)
    nodes={};edges=[];tour=[];stops=[];towers=[];candidates=[];originals=[];penalties={}
    def addnode(name,x,y=0.):nodes[name]=RoadNode(name,Point3D(x,y,0.))
    def addedge(name,tail,head):
        edges.append(RoadEdge(name,tail,head,Polyline3D((nodes[tail].location,nodes[head].location)),'controlled',60.,'uniform',True))
    for stage in range(1,9):
        launch='source' if stage==1 else f'L{stage}';x=(stage-1)*21*scale
        addnode(launch,x);addnode(f'M{stage}',x+10*scale);addnode(f'R{stage}',x+13*scale);addnode(f'D{stage}',x+11.5*scale,height*scale)
    addnode('sink',(7*21+21)*scale)
    for stage in range(1,9):
        launch='source' if stage==1 else f'L{stage}';middle=f'M{stage}';recovery=f'R{stage}';bypass=f'D{stage}'
        nextstop=f'L{stage+1}' if stage<8 else 'sink'
        for name,tail,head in [(f'approach{stage}',launch,middle),(f'direct{stage}',middle,recovery),(f'bypass_in{stage}',middle,bypass),(f'bypass_out{stage}',bypass,recovery),(f'gap{stage}',recovery,nextstop)]:addedge(name,tail,head)
        tour.extend((f'approach{stage}',f'direct{stage}',f'gap{stage}'))
        for vid in range(1,fleet+1):
            task=(vid-1)*8+stage;lid=nodes[launch].location;rid=nodes[recovery].location
            tower=Tower(str(task),Point3D((lid.x+rid.x)/2,rng.uniform(.6,1.4)*vid,.035),rng.uniform(.8,1.2),10.,1.,3.)
            towers.append(tower)
            flight=Polyline3D((lid,tower.location,rid)).length_3d
            duration=flight/2+tower.service_minutes
            energy=.42*flight+.16*tower.service_minutes;std=.08*energy;bound=energy+3*std
            candidate=PhysicalServiceCandidate(f'long:{vid}:{stage}',vid,vid,launch,recovery,(str(task),),duration,((str(task),duration),),energy,std,bound,bound<=32.)
            candidates.append(candidate);penalties[task]=10*tower.risk
            originals.append(SearchTuple(f'origin:{vid}:{stage}',vid,vid,(task,),launch,recovery,1,1+ceil(duration),admitted=candidate.admitted,physical_candidate_id=candidate.candidate_id))
    stops=tuple(CandidateStop(n.node_id,n.location,road_node_id=n.node_id) for n in nodes.values())
    vehicles=tuple(VehicleSpec(v,'source',255.,1,1.) for v in range(1,fleet+1))
    uavs=tuple(UavSpec(v,v,120.,40.,.2,('inspection',)) for v in range(1,fleet+1))
    lines=[]
    for vid in range(1,fleet+1):
        assigned=[t for t in towers if (int(t.tower_id)-1)//8+1==vid]
        lines.append(PowerLine(f'line{vid}',tuple(t.tower_id for t in assigned),Polyline3D(tuple(t.location for t in assigned))))
    spec=ScenarioSpec(split,'long','staged_bypass',seed,8*fleet,len(nodes),len(stops),fleet,fleet,256.)
    scenario=IrregularDualNetworkScenario(spec,RoadNetwork(tuple(nodes.values()),tuple(edges),'source'),PowerNetwork(tuple(towers),tuple(lines)),stops,vehicles,uavs,tuple(candidates),tuple((str(t),p) for t,p in penalties.items()))
    by_id={e.edge_id:e for e in edges};arcs=[]
    for vid in range(1,fleet+1):
        layer=1
        for eid in tour:
            e=by_id[eid];end=layer+ceil(e.geometry.length_3d-1e-12)
            arcs.append(SearchArc(f'original:{vid}:{eid}',vid,e.tail,e.head,layer,end,'road',e.geometry.length_3d,float(end-layer)));layer=end
    state=RollingState(0,1,frozenset(),frozenset(penalties),{v:VehicleState(v,'source',1,remaining_duty_time=255.) for v in range(1,fleet+1)},{v:UavState(v,v,1,remaining_energy=40.) for v in range(1,fleet+1)})
    topology=sha256(repr(sorted((e.tail,e.head) for e in edges)).encode()).hexdigest()
    return FormalV3Instance(f'long:{split}:{index:02d}',split,seed,'long','staged_bypass',topology,tuple(sorted(penalties)),penalties,{},SearchState(tuple(arcs),tuple(originals)),state,scenario,road_beta_w=0.,road_beta_eta=0.,max_duty_time=255.)


def scenarios(instance):
    # Register event clocks from geometry and fixed stop allowances alone.
    # No solver result, observed trajectory or realized performance enters this rule.
    edges={e.edge_id:e for e in instance.physical_scenario.road.edges};launch=3;events={}
    for stage in range(1,9):
        approach=ceil(edges[f'approach{stage}'].geometry.length_3d-1e-12)
        if stage in (3,5,7):
            layer=1+8*ceil(launch/8)
            assert launch<layer<launch+approach
            events[stage]=SegmentClosure(f'direct{stage}',layer)
        launch+=approach+ceil(edges[f'direct{stage}'].geometry.length_3d-1e-12)+ceil(edges[f'gap{stage}'].geometry.length_3d-1e-12)+4
    return {'stage3':(events[3],),'stage5':(events[5],),'stage7':(events[7],),
            'sequential':tuple(events[s] for s in (3,5,7)),
            'cut5':(events[5],SegmentClosure('bypass_in5',events[5].reveal_layer))}


def run(instance,method,config,closures=(),*,allow_detour=True,allow_new=True,initial_state=None,initial_memory=None,layers=LAYERS):
    if not layers or tuple(sorted(set(layers)))!=tuple(layers) or layers[-1]>=TERMINAL:raise ValueError('Invalid decision grid')
    if any(c.reveal_layer>=layers[0] and c.reveal_layer not in layers for c in closures):raise ValueError('Closure outside decision grid')
    state=initial_state or instance.rolling_state;memory=deepcopy(initial_memory) if initial_memory is not None else RoadMemory.initial(instance)
    records=[];pending=();failure=''
    for index,layer in enumerate(layers):
        if state.current_layer!=layer:raise ValueError('Decision grid and state disagree')
        active=tuple(c for c in closures if c.reveal_layer<=layer);before=deepcopy(memory)
        problem,routes=build_rolling_problem(instance,state,memory,active,allow_detour=allow_detour,allow_new=allow_new or not active)
        r=dict(layer=layer,state=state,closures=active,problem=problem,routes=routes,edge_ids=dict(memory.edge_ids),flight_origins=dict(memory.flight_origins),memory_before=before)
        if problem.safety_status!='feasible':failure=problem.safety_reason;r['failure']=failure;records.append(r);break
        if problem.search_state.tuples or problem.search_state.selected_arc_ids:
            trial=run_method(method,state=problem.search_state,task_ids=problem.task_ids,miss_penalty=problem.miss_penalty,validation_state=problem.rolling_state,operation_capacity=problem.operation_capacity,config=config)
            result=trial.result;r.update(result=result,config=trial.config)
            selected=replace(problem.search_state,selected_arc_ids=frozenset(result.selected_arc_ids),selected_tuple_ids=frozenset(result.selected_tuple_ids))
            check=validate_dynamic_plan(problem,selected)
            if not check.ok:failure='returned_plan:'+str(check.first_violation);r['failure']=failure;records.append(r);break
        else:selected=problem.search_state
        entries=physical_entries(problem,selected,routes)
        planned=record_plan(problem.rolling_state,problem.search_state.arcs,problem.search_state.tuples,selected.selected_arc_ids,selected.selected_tuple_ids)
        end=layers[index+1] if index+1<len(layers) else TERMINAL
        executed,pending=execute_due_events(planned,entries,end);state=advance_state(planned,executed)
        selected_vids={a.vehicle_id for a in problem.search_state.arcs if a.arc_id in selected.selected_arc_ids}
        pending_ids={e.arc_id for e in pending if isinstance(e,PlannedVehicleArcEvent)}
        for vid in selected_vids:memory.routes[vid]=tuple(a for a in routes[vid] if a.arc_id in pending_ids)
        r.update(selected=selected,entries=entries,planned_state=planned,execution=executed,pending=pending,state_after=state);records.append(r)
        if executed.safety_status!='feasible':failure=executed.safety_status;break
    return SimpleNamespace(state=state,pending=pending,records=records,failure=failure,feasible=not failure)
