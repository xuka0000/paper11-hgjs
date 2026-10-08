"""One measured deadline for construction, decisions, repair and validation."""
from dataclasses import dataclass,field
from math import isfinite
from time import perf_counter


class BudgetExpired(RuntimeError):
    pass


@dataclass(frozen=True)
class Deadline:
    seconds: float
    started: float=field(default_factory=perf_counter)

    def __post_init__(self):
        if not isfinite(self.seconds) or self.seconds<=0:raise ValueError('Budget must be finite and positive')

    def remaining(self):return max(0.,self.started+self.seconds-perf_counter())
    def elapsed(self):return perf_counter()-self.started
    def check(self):
        if self.remaining()<=0:raise BudgetExpired('Complete execution budget exhausted')
    def allowance(self,requested,reserve=0.):
        self.check()
        return max(0.,min(float(requested),self.remaining()-reserve))
