"""Source-corresponding TSHA search loop with an explicit decoder boundary.

The native scalar objective and native decoder can be injected for trace
parity. A paper11 decoder is a separately disclosed adaptation, not native
reproduction. No training, physical constraints or source implementation is
modified by this module.
"""
from math import isfinite
from random import Random
from Agent.Publication.xia2025_tsha import transform,acceptance_probability
from Agent.Publication.budget import BudgetExpired


def anneal(decoder,settings,deadline):
    current=best=decoder.initial_sequence()
    current_cost=decoder.cost(current)
    if not isfinite(current_cost):raise ValueError('Initial sequence is infeasible')
    initial=best_cost=current_cost
    rng=Random(settings.seed);weights=[1.]*4;history=set();total_uses=[0]*4
    temperature=settings.initial_temperature;stagnant=total_accepted=0
    trace=[];moves=[];stop='temperature_limit'
    try:
        while temperature>settings.temperature_stop and stagnant<settings.successive_cooldowns:
            before_best=best_cost;scores=[1.]*4;counts=[0]*4;infeasible=repeated=0
            for iteration in range(settings.iterations_per_temperature):
                deadline.check()
                operator=rng.choices(range(4),weights=weights,k=1)[0]
                candidate=transform(current,operator,decoder,rng);used=[operator]
                counts[operator]+=1;total_uses[operator]+=1;retries=0
                while candidate in history and retries<settings.history_retry_limit:
                    deadline.check();operator=rng.randrange(4)
                    candidate=transform(candidate,operator,decoder,rng);used.append(operator)
                    counts[operator]+=1;total_uses[operator]+=1;repeated+=1;retries+=1
                if candidate in history:stop='history_retry_limit';break
                candidate_cost=decoder.cost(candidate)
                old_cost=current_cost;probability=acceptance_probability(current_cost,candidate_cost,temperature)
                accepted=probability>rng.random()
                if not isfinite(candidate_cost):score=0.;infeasible+=1
                elif candidate_cost<current_cost:
                    score=1+(current_cost-candidate_cost)/current_cost
                elif accepted:score=1+(candidate_cost-current_cost)/candidate_cost
                else:score=1+(current_cost-candidate_cost)/candidate_cost
                for op in set(used):scores[op]+=score
                for op in range(4):
                    if counts[op]:weights[op]=settings.rho*weights[op]+(1-settings.rho)*scores[op]/counts[op]
                if accepted:current,current_cost=candidate,candidate_cost;total_accepted+=1
                if current_cost<best_cost-1e-9:best,best_cost=current,current_cost
                moves.append(dict(temperature=temperature,operators=used,
                    candidate_objective=candidate_cost if isfinite(candidate_cost) else None,
                    old_current_objective=old_cost,current_objective=current_cost,best_objective=best_cost,
                    accepted=accepted,acceptance_probability=probability,
                    accepted_sequence=list(current) if accepted else None))
            history.add(best)
            stagnant=stagnant+1 if abs(best_cost-before_best)<=1e-9 else 0
            trace.append(dict(temperature=temperature,best_return_time_seconds=best_cost,
                current_return_time_seconds=current_cost,weights=list(weights),operator_uses=counts,
                infeasible_candidates=infeasible,history_retries=repeated,stagnant_temperatures=stagnant))
            if stop=='history_retry_limit':break
            temperature*=settings.cooling_rate
    except BudgetExpired:stop='common_deadline'
    if stagnant>=settings.successive_cooldowns:stop='successive_cooldowns'
    return dict(initial_cost=initial,best_cost=best_cost,best_sequence=best,
        total_operator_uses=total_uses,accepted_moves=total_accepted,stop_reason=stop,trace=trace,moves=moves)
