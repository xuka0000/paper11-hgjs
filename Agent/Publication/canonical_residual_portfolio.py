"""Residual-time family portfolio for canonical repair candidates."""
from Agent.Publication.canonical_portfolio import family_portfolio_indices


def residual_portfolio_indices(scores, candidates):
    indices = family_portfolio_indices(scores, candidates)
    global_best = min(range(len(candidates)), key=lambda index: (
        -float(scores[index]), candidates[index]["candidate_id"]))
    if indices[0] != global_best:
        raise ValueError("The first family representative must be the global model choice")
    return indices


def residual_repair_budget(total_allowance_s, elapsed_s, remaining_candidates,
                           first_candidate=False):
    if total_allowance_s <= 0:
        raise ValueError("The total allowance must be positive")
    if remaining_candidates <= 0:
        raise ValueError("At least one candidate must remain")
    remaining_s = max(0.0, float(total_allowance_s) - float(elapsed_s))
    if first_candidate:
        return remaining_s
    return remaining_s / int(remaining_candidates)


__all__ = ["residual_portfolio_indices", "residual_repair_budget"]
