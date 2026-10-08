"""Common checked objective for optional component-level baseline adaptations."""
from Agent.Run.publication_plan_distance import objective
from Agent.Run.check_publication_hamming_schedule import check_plan


def evaluate_plan(domain, plan, weights, task_groups=None):
    values = objective(domain, plan, weights)
    checked = check_plan(domain, plan, weights, values)
    if task_groups is None:
        return checked
    task_groups.validate_domain(domain)
    components = task_groups.evaluate(plan)
    return dict(checked, primary=components['primary'], observation_primary=values[0],
                metric='weighted_incomplete_components', components=components,
                completed_groups=[g for g, row in components['groups'].items() if row['complete']])
