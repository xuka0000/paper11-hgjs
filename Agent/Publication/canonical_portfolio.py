"""Deterministic family-diverse portfolio for canonical repair candidates."""
from Agent.Publication.canonical_ranker import FAMILIES


def family_portfolio_indices(scores, candidates):
    if len(scores) != len(candidates):
        raise ValueError("Scores and candidates differ in length")
    selected = []
    for family in FAMILIES:
        eligible = [
            index for index, candidate in enumerate(candidates)
            if candidate["family"] == family]
        if not eligible:
            raise ValueError(f"Missing canonical candidate family: {family}")
        selected.append(min(eligible, key=lambda index: (
            -float(scores[index]), candidates[index]["candidate_id"])))
    return tuple(sorted(selected, key=lambda index: (
        -float(scores[index]), candidates[index]["candidate_id"])))


__all__ = ["family_portfolio_indices"]
