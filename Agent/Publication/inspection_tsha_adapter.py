"""Two-stage inspection reference with explicitly non-native domain mapping.

This method is a source-mechanism adaptation, not a reproduction of the
published TSHA results. Native critical-road construction and the source
annealing controller are reused. Projection/multi-task decoding is replaced
by the declared paper11 single-task physical decoder. Full details and
native-controller parity must accompany every comparison.
"""
from dataclasses import replace
from time import perf_counter
from Agent.Publication.budget import Deadline,BudgetExpired
from Agent.Publication.xia2025_tsha import SearchSettings
from Agent.Publication.inspection_sa_kernel import anneal
from Agent.Publication.inspection_spatial_adapter import native_guided_roads
from Agent.Publication.inspection_sequence_decoder_v2 import InspectionSequenceDecoderV2
from Agent.Publication.component_objective import evaluate_plan


def run_inspection_tsha_adapter(problem,weights,*,seed=260919,total_seconds=15.,task_groups=None):
    if total_seconds<=1:raise ValueError('One-second final validation reserve required')
    start=perf_counter();deadline=Deadline(total_seconds-1.)
    shared=problem.plan(problem.state);best_plan=shared
    initial_check=evaluate_plan(problem.domain,shared,weights,task_groups)
    initial=best=(float(initial_check['primary']),float(initial_check['secondary']))
    stage={};search={};observations=[];stop='common_deadline'
    try:
        restricted,positions,stage=native_guided_roads(problem,deadline)
        physical=InspectionSequenceDecoderV2(restricted,weights,deadline,task_groups=task_groups)

        class AdaptedDecoder:
            flight=physical.flight
            pair_positions=physical.pair_positions

            def initial_sequence(self):
                return tuple(sorted(range(len(physical.tasks)),key=lambda i:(
                    positions.get(physical.tasks[i],float('inf')),physical.tasks[i])))

            def cost(self,sequence):
                nonlocal best,best_plan
                seen=tuple(sequence) in physical.cache
                result=physical.evaluate(sequence)
                values=tuple(result['objective'])
                if values<best:
                    best=values;best_plan=result['plan']
                # Nonlinear scalar is only an annealing guidance cost. Final
                # selection always uses the original lexicographic objective.
                scalar=1.+values[0]+values[1]/(1.+values[1])
                if not seen:
                    observations.append(dict(sequence=[physical.tasks[i] for i in sequence],
                        objective=values,guidance_cost=scalar,
                        selected_tuple_ids=sorted(c.tuple_id for c in result['plan'].copies),
                        selected_arc_ids=sorted(a.arc_id for a in result['plan'].arcs),
                        decoder_attempts=result['attempts'],observed_seconds=perf_counter()-start))
                return scalar

        search=anneal(AdaptedDecoder(),SearchSettings(rho=.5,seed=seed),deadline)
        stop=search['stop_reason']
    except BudgetExpired:pass
    final=replace(problem.state,selected_arc_ids=frozenset(a.arc_id for a in best_plan.arcs),
        selected_tuple_ids=frozenset(c.tuple_id for c in best_plan.copies))
    checked=evaluate_plan(problem.domain,problem.plan(final),weights,task_groups)
    assert max(abs(a-b) for a,b in zip(best,[checked['primary'],checked['secondary']]))<1e-6
    elapsed=perf_counter()-start
    return dict(method='inspection_tsha_mechanism_adaptation_v1' if task_groups is None else 'inspection_tsha_component_adaptation_v1',source_exact_reproduction=False,
        metric='weighted_missed_observations' if task_groups is None else 'weighted_incomplete_components',
        initial_objective=initial,objective=best,selected_arc_ids=sorted(final.selected_arc_ids),
        selected_tuple_ids=sorted(final.selected_tuple_ids),stage_one=stage,search=search,
        decoder_observations=observations,unique_sequences=len(observations),
        physical_check=checked,total_action_seconds=elapsed,within_common_budget=elapsed<=total_seconds,
        stop_reason=stop,seed=seed,
        objective_mapping='Annealing guidance1+U+C/(1+C); final best retained across all checked evaluations by original lexicographic(U,C), including shared incumbent. Not native return-time minimization.',
        source_parameter_mapping='Native T0=90,cooling=.98,150 iterations per temperature,11 stagnant temperatures,rho=.5; no outcome-based tuning.',
        limitations='Static adapted comparator. Native continuous rendezvous,multi-task sorties,homogeneous one-carrier assumptions and zero handling are not transplanted. Native route geometry constrains edge availability, not traversal order. No rolling admission.')
