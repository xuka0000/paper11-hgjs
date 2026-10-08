"""Physical task-order decoder for a disclosed inspection-SA adaptation.

Version2 additionally treats conflicting mandatory vehicle actions as a rejected
candidate, while all other unknown validation errors still stop execution.
The version1 source and its two smoke outcomes are unchanged.

This is an adapter component, not an admitted reproduction of Xia's TSHA.
The source's continuous projection and multi-task sortie decoder cannot be
silently substituted into paper11. This decoder uses exactly the supplied
finite service catalogue, carried-UAV ownership, positive handling, energy,
and full physical route checker. No MILP repair or learned score is used.
"""
from collections import Counter,defaultdict
from math import dist
from Agent.Publication.budget import BudgetExpired
from Agent.Run.publication_committed_service_domain import Plan,handling
from Agent.Run.optimize_publication_hamming_routes import optimize_committed_routes
from Agent.Run.publication_plan_distance import objective


class InspectionSequenceDecoderV2:
    def __init__(self,problem,weights,deadline,*,task_groups=None):
        if problem.domain.context.state.epoch or problem.domain.specials or problem.domain.groups:
            raise ValueError('Static decoder only: rolling commitments require a separate adapter')
        self.problem,self.weights,self.deadline=problem,weights,deadline
        self.task_groups=task_groups
        if task_groups is not None:task_groups.validate_domain(problem.domain)
        self.tasks=tuple(sorted(problem.domain.problem.task_ids))
        points={int(t.tower_id):(t.location.x,t.location.y,t.location.z)
                for t in problem.domain.instance.physical_scenario.power.towers}
        self.flight=[[dist(points[a],points[b]) for b in self.tasks] for a in self.tasks]
        self.pair_positions=[(a,b) for a in range(len(self.tasks)-1)
                             for b in range(a+2,len(self.tasks)-1)]
        self.by_task=defaultdict(list)
        for c in problem.full_copies.values():
            if len(c.task_ids)!=1 or c.is_air_residual:
                raise ValueError('Declared fresh single-task services required')
            self.by_task[c.task_ids[0]].append(c)
        self.capacity={v.vehicle_id:v.operation_capacity for v in problem.domain.instance.physical_scenario.vehicles}
        self.cache={};self.evaluations=0;self.physical_checks=0

    def initial_sequence(self):
        # Source projection order is not claimed: physical temporal order is
        # the explicit common-domain replacement, fixed before evaluation.
        return tuple(sorted(range(len(self.tasks)),key=lambda i:(
            min((c.launch_layer for c in self.by_task[self.tasks[i]]),default=10**9),self.tasks[i])))

    def evaluate(self,sequence):
        self.deadline.check()
        sequence=tuple(sequence)
        if sorted(sequence)!=list(range(len(self.tasks))):
            raise ValueError('Sequence must contain every known task exactly once')
        if sequence in self.cache:return self.cache[sequence]
        domain=self.problem.domain;instance=domain.instance;now=domain.context.state.current_layer
        plan=Plan();attempts=[]
        busy=Counter(domain.fixed_busy);ops=Counter(domain.fixed_operations)
        for index in sequence:
            task=self.tasks[index]
            for copy in sorted(self.by_task[task],key=lambda c:(c.launch_layer,c.recovery_layer,c.secondary_cost,c.tuple_id)):
                self.deadline.check()
                occupied=[(copy.uav_id,k) for k in range(max(now,copy.launch_layer-instance.preparation_layers),
                    copy.recovery_layer+instance.recovery_service_layers+instance.turnaround_layers)]
                operations=handling(instance,copy,now)
                if any(busy[k]>=1 for k in occupied) or any(ops[k]>=self.capacity[k[0]] for k in operations):
                    attempts.append((task,copy.tuple_id,'resource_conflict'));continue
                try:
                    trial,_=optimize_committed_routes(domain,plan.copies+(copy,),self.weights)
                except ValueError as exc:
                    # Only expected route infeasibility is a rejected move;
                    # checker/accounting defects must terminate the experiment.
                    if str(exc) not in ('No route for fixed continuation services','No common return-time cap',
                                      'Conflicting mandatory vehicle actions'):
                        raise
                    attempts.append((task,copy.tuple_id,str(exc)));continue
                self.deadline.check()
                self.physical_checks+=1;plan=trial;busy.update(occupied);ops.update(operations)
                attempts.append((task,copy.tuple_id,'accepted'));break
        values=objective(domain,plan,self.weights)
        if self.task_groups is not None:
            values=(self.task_groups.evaluate(plan)['primary'],values[1])
        result=dict(plan=plan,objective=values,attempts=attempts,
            selected_tasks=sorted(t for c in plan.copies for t in c.task_ids))
        self.deadline.check();self.cache[sequence]=result;self.evaluations+=1
        return result
