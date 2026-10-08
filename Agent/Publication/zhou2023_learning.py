"""Official Zhou GCN with explicit paper11 graph and learning adaptations.

The archived model class is loaded unchanged. Both lexicographic objectives are
represented separately, with no scalar weighting introduced into optimization.
"""
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
from math import isfinite, sqrt
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import scipy.sparse as sp
import torch
from ortools.linear_solver import linear_solver_pb2

from Agent.Publication.physical_search_baselines import _pool_domain
from Agent.Publication.vehicle_lns_milp import solve_committed
from Agent.Run.publication_committed_service_domain import copy_terms
from Agent.Run.publication_plan_distance import plan_reference

ROOT = Path(__file__).resolve().parents[2]
MODEL_SOURCE = ROOT/'Data/Reference/external_baselines_20260913/AGH/Improvement_based/cplex/models.py'
MODEL_SHA = 'dd417fbdd2f83df34670432a377cd3399d9b811e2e55ace1acc7448ab6169ffc'


def official_model(**kwargs):
    if sha256(MODEL_SOURCE.read_bytes()).hexdigest() != MODEL_SHA:
        raise ValueError('Archived author GCN changed')
    spec = importlib.util.spec_from_file_location('paper11_pinned_zhou_gcn', MODEL_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GCN(**kwargs)


def new_policy():
    return official_model(c_dim=5, x_dim=4, t_dim=4, k_f_dim=2, emb_dim=64, hidden=128)


def sparse_tensor(matrix):
    matrix = matrix.tocoo()
    return torch.sparse_coo_tensor(
        torch.tensor(np.stack([matrix.row,matrix.col]),dtype=torch.long),
        torch.tensor(matrix.data,dtype=torch.float32), matrix.shape).coalesce()


def normalized_edges(matrix):
    rows = np.asarray(abs(matrix).sum(axis=1)).ravel()
    columns = np.asarray(abs(matrix).sum(axis=0)).ravel()
    forward = sp.diags(1/np.maximum(rows,1e-30)) @ matrix
    backward = (matrix @ sp.diags(1/np.maximum(columns,1e-30))).T
    return sparse_tensor(forward), sparse_tensor(backward)


class GraphCaptured(Exception):
    pass


@dataclass
class FormulationGraph:
    problem: object
    weights: dict
    proto: object
    names: tuple
    owner: dict
    primary: np.ndarray
    secondary: np.ndarray
    secondary_offset: float
    integer_ids: tuple
    continuous_ids: tuple
    group_ids: tuple
    constraints: torch.Tensor
    e_cv: torch.Tensor
    e_vc: torch.Tensor
    e_v_veh: torch.Tensor
    physical_matrix: object
    lower: np.ndarray
    upper: np.ndarray
    horizon: float
    task_groups: object = None

    def assignment(self, state):
        plan = self.problem.plan(state)
        domain = self.problem.domain
        arcs = set(state.selected_arc_ids)
        copies = set(state.selected_tuple_ids)
        active = {a.vehicle_id for a in plan.arcs}
        covered = {t for c in plan.copies for t in c.task_ids}
        maximum = max([0]+[a.arrival_layer for a in plan.arcs if a.head=='sink']+
                      [k for c in plan.copies for t,k in c.completion_layers if t in domain.context.state.remaining_tasks])
        values=[]
        certificates = self.task_groups.evaluate(plan)['groups'] if self.task_groups is not None else {}
        for name in self.names:
            if name.startswith('x:'): value=float(name[2:] in arcs)
            elif name.startswith('g:'): value=float(int(name[2:]) in active)
            elif name.startswith('copy:'): value=float(name[5:] in copies)
            elif name.startswith('miss:'): value=float(int(name[5:]) not in covered)
            elif name.startswith('group_miss:'):
                value=float(not certificates[name[len('group_miss:'):]]['complete'])
            elif name.startswith('group_window:'):
                gid,end=name[len('group_window:'):].rsplit(':',1)
                cert=certificates[gid]
                value=float(cert['complete'] and cert['time']==int(end))
            elif name=='makespan': value=float(maximum)
            elif name.startswith('busy:'):
                _,u,k=name.split(':');u,k=int(u),int(k)
                now=domain.context.state.current_layer
                value=float(sum(c.uav_id==u and max(now,c.launch_layer-domain.instance.preparation_layers)<=k
                     <c.recovery_layer+domain.instance.recovery_service_layers+domain.instance.turnaround_layers for c in plan.copies))
            else: raise ValueError('Unmapped common-domain variable '+name)
            values.append(value)
        values=np.asarray(values,dtype=np.float64)
        activity=self.physical_matrix @ values
        if np.any(activity<self.lower-1e-7) or np.any(activity>self.upper+1e-7):
            raise ValueError('Graph assignment violates actual MILP rows')
        for i,var in enumerate(self.proto.variable):
            if not var.lower_bound-1e-7 <= values[i] <= var.upper_bound+1e-7:
                raise ValueError('Graph assignment violates variable bounds')
        if self.task_groups is None:
            check=self.problem.check(state,self.weights)
        else:
            from Agent.Publication.component_objective import evaluate_plan
            check=evaluate_plan(domain,plan,self.weights,self.task_groups)
        if abs(self.primary @ values-check['primary'])>1e-6 or abs(self.secondary @ values+self.secondary_offset-check['secondary'])>1e-6:
            raise ValueError('Graph objective differs from full physical accounting')
        return values

    def tensors(self,current,best):
        cur,inc=self.assignment(current),self.assignment(best)
        costs=np.stack([self.primary,self.secondary],axis=1)
        costs=costs/np.maximum(abs(costs).max(axis=0),1e-30)
        def features(ids,normalizer):
            ids=np.asarray(ids,dtype=np.int64)
            return torch.tensor(np.column_stack([costs[ids],cur[ids]/normalizer,inc[ids]/normalizer]),dtype=torch.float32)
        x=features(self.integer_ids,1.)
        t=features(self.continuous_ids,self.horizon)
        group_cost=[]
        used=[]
        for group in self.group_ids:
            indices=[i for i,name in enumerate(self.names) if self.owner.get(name)==group]
            group_cost.append(sum(self.secondary[i]*cur[i] for i in indices))
            used.append(float(any(cur[i]>.5 for i in indices)))
        scale=max([abs(x) for x in group_cost]+[1e-30])
        kf=torch.tensor([[u,c/scale] for u,c in zip(used,group_cost)],dtype=torch.float32)
        result=(self.constraints,x,t,kf,self.e_cv,self.e_vc,self.e_v_veh)
        if any(not torch.isfinite(z.values() if z.is_sparse else z).all() for z in result):
            raise ValueError('Nonfinite graph feature')
        return result


def build_formulation_graph(problem,weights,*,task_groups=None,deadline=None):
    if deadline is not None:deadline.check()
    domain=_pool_domain(problem.domain,problem.full_copies.values())
    if domain.templates:
        raise ValueError('This graph adapter requires the shared explicit bounded catalogue')
    captured={}
    def observer(solver,variables,hints,owned,fixed):
        if hints is None or fixed:
            raise ValueError('Graph must use a verified state and the unrestricted common model')
        proto=linear_solver_pb2.MPModelProto()
        solver.ExportModelToProto(proto)
        captured.update(proto=proto,owned=owned,hints=np.asarray(hints))
        raise GraphCaptured()
    try:
        solve_committed(domain,NS(objective_weights=weights),incumbent=problem.plan(problem.state),
                        released_vehicle_ids=tuple(problem.domain.context.state.vehicles),model_observer=observer,
                        task_groups=task_groups,deadline=deadline)
    except GraphCaptured: pass
    proto=captured['proto'];names=tuple(v.name for v in proto.variable)
    ids={name:i for i,name in enumerate(names)}
    primary=np.asarray([v.objective_coefficient for v in proto.variable],dtype=np.float64)
    secondary=np.zeros(len(names));reference=plan_reference(domain)
    churn=weights['plan_change_weight']
    for arc in domain.problem.search_state.arcs:
        secondary[ids['x:'+arc.arc_id]]=arc.secondary_cost+weights['time_weight']*arc.duty_time+churn*reference.arc_coefficient(arc)
    for c in domain.specials:
        _,cost=copy_terms(domain.instance,c,domain.context.state,weights)
        secondary[ids['copy:'+c.tuple_id]]=cost+churn*reference.copy_coefficient(c)
    secondary[ids['makespan']]=weights['makespan_weight']
    offset=churn*(len(reference.old_arcs)+len(reference.old_services)) if reference.present else 0.
    integer=tuple(i for i,v in enumerate(proto.variable) if v.is_integer)
    continuous=tuple(i for i,v in enumerate(proto.variable) if not v.is_integer)
    permutation=integer+continuous
    norms=[np.linalg.norm(primary),np.linalg.norm(secondary)]
    raw_rows=[];raw_columns=[];raw_values=[]
    rows=[];columns=[];values=[];features=[]
    lower=[];upper=[]
    for index,c in enumerate(proto.constraint):
        if deadline is not None and index%128==0:deadline.check()
        lower.append(c.lower_bound);upper.append(c.upper_bound)
        raw_rows.extend([index]*len(c.var_index));raw_columns.extend(c.var_index);raw_values.extend(c.coefficient)
        equality=c.lower_bound==c.upper_bound
        # Convert each finite side to an equality or <= row without losing rows.
        sides=[(1.,c.upper_bound,True)] if equality else (
            ([(1.,c.upper_bound,False)] if isfinite(c.upper_bound) else [])+
            ([(-1.,-c.lower_bound,False)] if isfinite(c.lower_bound) else []))
        for sign,bias,eq in sides:
            coefficients=np.asarray(c.coefficient)*sign
            scale=abs(bias) if abs(bias)>1e-12 else 1.
            norm=np.linalg.norm(coefficients)
            cosine=[float(np.dot(coefficients,obj[list(c.var_index)])/(norm*n)) if norm*n>0 else 0.
                    for obj,n in zip([primary,secondary],norms)]
            row_id=len(features)
            features.append(cosine+[bias/scale,float(eq),float(not eq)])
            rows.extend([row_id]*len(c.var_index));columns.extend(c.var_index);values.extend(coefficients/scale)
    matrix=sp.coo_matrix((values,(rows,columns)),shape=(len(features),len(names))).tocsr()[:,permutation]
    forward,backward=normalized_edges(matrix)
    groups=tuple(sorted(problem.domain.context.state.vehicles))
    group_index={g:i for i,g in enumerate(groups)}
    vehicle_rows=[];vehicle_columns=[]
    for column,i in enumerate(permutation):
        if names[i] in captured['owned']:
            vehicle_rows.append(group_index[captured['owned'][names[i]]]);vehicle_columns.append(column)
    membership=sp.coo_matrix((np.ones(len(vehicle_rows)),(vehicle_rows,vehicle_columns)),shape=(len(groups),len(names))).tocsr()
    counts=np.asarray(membership.sum(axis=1)).ravel()
    if np.any(counts==0):raise ValueError('Vehicle group without variables')
    pooled=sparse_tensor(sp.diags(1/counts) @ membership)
    graph=FormulationGraph(problem,dict(weights),proto,names,captured['owned'],primary,secondary,offset,
        integer,continuous,groups,torch.tensor(features,dtype=torch.float32),forward,backward,pooled,
        sp.coo_matrix((raw_values,(raw_rows,raw_columns)),shape=(len(proto.constraint),len(names))).tocsr(),
        np.asarray(lower),np.asarray(upper),max(1.,proto.variable[ids['makespan']].upper_bound),task_groups)
    if not np.allclose(graph.assignment(problem.state),captured['hints'],rtol=0,atol=1e-9):
        raise ValueError('Independent graph assignment differs from verified solver hint')
    if deadline is not None:deadline.check()
    return graph


def sample_groups(logits,groups,degree,generator):
    from math import floor
    if logits.numel()!=len(groups) or not 0<degree<=1:
        raise ValueError('Invalid learned selection dimensions or degree')
    # Official forward_training.py samples softmax(logits), not sigmoid scores.
    probability=torch.softmax(logits.reshape(-1),dim=0)
    indices=torch.multinomial(probability,max(1,floor(degree*len(groups))),replacement=False,generator=generator)
    return tuple(groups[i] for i in indices.tolist())


def best_improving_teacher(initial, trials):
    usable=[t for t in trials if t['physically_valid'] and t['within_budget'] and tuple(t['objective'])<tuple(initial)]
    return min(usable,key=lambda t:tuple(t['objective'])) if usable else None


def supervised_update(model,optimizer,samples,batch_size=16):
    model.train();criterion=torch.nn.BCEWithLogitsLoss();losses=[];updates=0
    for start in range(0,len(samples),batch_size):
        batch=samples[start:start+batch_size]
        optimizer.zero_grad();total=None
        for tensors,label in batch:
            loss=criterion(model(*tensors).reshape(-1),label.float().reshape(-1))
            total=loss if total is None else total+loss
            losses.append(float(loss.detach()))
        (total/len(batch)).backward();optimizer.step();updates+=1
    return {'samples':len(samples),'optimizer_updates':updates,'mean_loss':sum(losses)/len(losses) if losses else None}
