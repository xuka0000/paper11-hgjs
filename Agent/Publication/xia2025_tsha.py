"""Xia et al. 2025 VAMDIRP-TSHA, with explicitly recorded interpretations.

See Data/Config/xia2025_reimplementation_spec_v1.json before attributing this
implementation. This is the article's STATIC ONE-VEHICLE problem, not paper11.
No physical_heuristic, local canonical neighborhoods or MILP repair is reused.
"""
from dataclasses import dataclass, asdict
from heapq import heappush, heappop
from math import hypot, exp, isfinite
from random import Random
from time import perf_counter


def distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def projection(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    denominator = dx * dx + dy * dy
    if denominator <= 0:
        raise ValueError('Zero-length road')
    fraction = max(0., min(1., ((point[0]-a[0])*dx + (point[1]-a[1])*dy)/denominator))
    foot = (a[0] + fraction*dx, a[1] + fraction*dy)
    return distance(point, foot), fraction, foot


@dataclass
class InspectionInstance:
    nodes: dict
    edges: tuple
    tasks: dict
    depot: str
    drone_count: int = 3
    endurance_seconds: float = 1500.
    service_seconds: float = 60.
    drone_speed_mps: float = 5.
    vehicle_speed_mps: float = 30.

    def __post_init__(self):
        self.nodes = {str(k): tuple(map(float, v)) for k, v in self.nodes.items()}
        self.tasks = {str(k): tuple(map(float, v)) for k, v in self.tasks.items()}
        self.depot = str(self.depot)
        self.edges = tuple(sorted({tuple(sorted(map(str, e))) for e in self.edges}))
        if self.depot not in self.nodes or not self.edges or not self.tasks:
            raise ValueError('Nonempty connected road network and tasks required')
        if self.drone_count < 1 or int(self.drone_count) != self.drone_count:
            raise ValueError('Positive integer drone count required')
        for value in [self.endurance_seconds, self.service_seconds, self.drone_speed_mps, self.vehicle_speed_mps]:
            if not isfinite(value) or value <= 0:
                raise ValueError('Finite positive times and speeds required')
        for point in list(self.nodes.values()) + list(self.tasks.values()):
            if len(point) != 2 or not all(isfinite(x) for x in point):
                raise ValueError('Finite planar coordinates required')
        for a, b in self.edges:
            if a not in self.nodes or b not in self.nodes or distance(self.nodes[a], self.nodes[b]) <= 0:
                raise ValueError('Invalid road edge')


@dataclass(frozen=True)
class SearchSettings:
    rho: float
    seed: int = 260913
    initial_temperature: float = 90.
    cooling_rate: float = 0.98
    iterations_per_temperature: int = 150
    successive_cooldowns: int = 11
    temperature_stop: float = 0.01
    history_retry_limit: int = 256

    def __post_init__(self):
        if not all(isfinite(x) for x in [self.rho, self.initial_temperature, self.cooling_rate, self.temperature_stop]):
            raise ValueError('Finite search parameters required')
        if not 0 < self.rho < 1 or not 0 < self.cooling_rate < 1:
            raise ValueError('rho and cooling_rate must be in (0,1)')
        if not 0 < self.temperature_stop < self.initial_temperature:
            raise ValueError('Invalid temperature range')
        if min(self.iterations_per_temperature, self.successive_cooldowns, self.history_retry_limit) < 1:
            raise ValueError('Positive search counts required')


def near_cluster(items):
    """Section 4.1 near/far clusters; explicit interpretation R10."""
    ordered = sorted(items, key=lambda x: (x[1], x[0]))
    if len(ordered) < 2 or ordered[0][1] == ordered[-1][1]:
        return {key for key, _ in ordered}
    prefix, squares = [0.], [0.]
    for _, value in ordered:
        prefix.append(prefix[-1] + value)
        squares.append(squares[-1] + value*value)
    candidates = []
    n = len(ordered)
    for cut in range(1, n):
        if ordered[cut-1][1] == ordered[cut][1]:
            continue
        error = squares[cut] - prefix[cut]**2/cut
        error += squares[n]-squares[cut] - (prefix[n]-prefix[cut])**2/(n-cut)
        candidates.append((error, cut))
    cut = min(candidates)[1]
    return {key for key, _ in ordered[:cut]}


def shortest_connection(instance, start, target, previously_used=frozenset()):
    """Section 4.1 connection priority, interpretation R11."""
    adjacency = {node: [] for node in instance.nodes}
    for a, b in instance.edges:
        length = distance(instance.nodes[a], instance.nodes[b])
        penalty = int((a, b) in previously_used)
        adjacency[a].append((b, penalty, length))
        adjacency[b].append((a, penalty, length))
    queue = [(0, 0., (start,), start)]
    best = {start: (0, 0., (start,))}
    while queue:
        repeats, length, path, node = heappop(queue)
        if (repeats, length, path) != best[node]:
            continue
        if node == target:
            return repeats, length, path
        for nxt, penalty, increment in adjacency[node]:
            value = (repeats+penalty, length+increment, path+(nxt,))
            if nxt not in best or value < best[nxt]:
                best[nxt] = value
                heappush(queue, (*value, nxt))
    raise ValueError('Disconnected road network')


def construct_vehicle_route(instance):
    """Algorithm 2, including source-required return to base."""
    radius = (instance.endurance_seconds-instance.service_seconds)*instance.drone_speed_mps/2
    covers, task_distances = {}, {}
    for edge in instance.edges:
        a, b = [instance.nodes[k] for k in edge]
        task_distances[edge] = {k: projection(p, a, b)[0] for k, p in instance.tasks.items()}
        covers[edge] = near_cluster([(k, d) for k, d in task_distances[edge].items() if d <= radius+1e-9])
    covered = set().union(*covers.values())
    for task in set(instance.tasks)-covered:
        edge = min(instance.edges, key=lambda e: (task_distances[e][task], e))
        if task_distances[edge][task] > radius+1e-9:
            raise ValueError('Task outside every endurance-feasible road: ' + task)
        covers[edge].add(task)
    costs = {e: sum(task_distances[e][t] for t in tasks)/len(tasks)
             for e, tasks in covers.items() if tasks}
    uncovered, critical = set(instance.tasks), []
    for edge in sorted(costs, key=lambda e: (costs[e], e)):
        if covers[edge] & uncovered:
            critical.append(edge)
            uncovered -= covers[edge]
        if not uncovered:
            break
    route, used, uncovered = [instance.depot], set(), set(instance.tasks)
    selected = []
    while uncovered:
        candidates = []
        for edge in critical:
            if not covers[edge] & uncovered:
                continue
            for start, end in [edge, edge[::-1]]:
                repeats, length, path = shortest_connection(instance, route[-1], start, used)
                candidates.append((repeats, length, edge, start, end, path))
        if not candidates:
            raise ValueError('Road cover construction failed')
        _, _, edge, start, end, path = min(candidates)
        extension = list(path[1:]) + [end]
        for node in extension:
            used.add(tuple(sorted((route[-1], node))))
            route.append(node)
        uncovered -= covers[edge]
        critical.remove(edge)
        selected.append(edge)
    _, _, closing = shortest_connection(instance, route[-1], instance.depot)
    route.extend(closing[1:])
    return tuple(route), {'coverage_radius_m': radius, 'selected_roads': selected,
                          'road_costs': [{'edge': e, 'cost': costs[e]} for e in sorted(costs)]}


class RouteDecoder:
    """Algorithms 4 and 5 on a frozen stage-one vehicle path."""
    def __init__(self, instance, vehicle_route):
        self.instance = instance
        self.vehicle_route = tuple(vehicle_route)
        self.tasks = tuple(sorted(instance.tasks))
        self.index = {k: i for i, k in enumerate(self.tasks)}
        self.coordinates = [instance.tasks[k] for k in self.tasks]
        self.prefix = [0.]
        for a, b in zip(vehicle_route, vehicle_route[1:]):
            self.prefix.append(self.prefix[-1] + distance(instance.nodes[a], instance.nodes[b]))
        self.feet, self.positions, self.offsets = [], [], []
        for task in self.coordinates:
            options = []
            for segment, (a, b) in enumerate(zip(vehicle_route, vehicle_route[1:])):
                d, fraction, foot = projection(task, instance.nodes[a], instance.nodes[b])
                position = self.prefix[segment] + fraction*(self.prefix[segment+1]-self.prefix[segment])
                options.append((d, position, segment, foot))
            d, position, _, foot = min(options)
            self.feet.append(foot)
            self.positions.append(position)
            self.offsets.append(d)
        self.unique_positions = sorted(set(round(s, 9) for s in self.positions))
        self.event_index = [self.unique_positions.index(round(s, 9)) for s in self.positions]
        self.flight = [[distance(a, b)/instance.drone_speed_mps for b in self.coordinates]
                       for a in self.coordinates]
        self.off_time = [d/instance.drone_speed_mps for d in self.offsets]
        self.pair_positions = [(a, b) for a in range(len(self.tasks)-1)
                               for b in range(a+2, len(self.tasks)-1)]

    def initial_sequence(self):
        """Section 4.2.1, earliest projection along the road tour."""
        return tuple(sorted(range(len(self.tasks)), key=lambda i: (self.positions[i], self.tasks[i])))

    def split_route(self, sequence):
        """Algorithm 4 with endurance correction R01, derived from eq. 10."""
        pieces, current, duration = [], [], 0.
        for task in sequence:
            single = 2*self.off_time[task] + self.instance.service_seconds
            if single > self.instance.endurance_seconds+1e-9:
                return None
            added = (self.flight[current[-1]][task] + self.off_time[task]
                     - self.off_time[current[-1]] + self.instance.service_seconds) if current else single
            if current and duration+added > self.instance.endurance_seconds+1e-9:
                pieces.append((tuple(current), duration))
                current, duration = [task], single
            else:
                current.append(task)
                duration += added
        if current:
            pieces.append((tuple(current), duration))
        return pieces

    def evaluate(self, sequence, materialize=False):
        pieces = self.split_route(sequence)
        if pieces is None:
            return float('inf'), None
        starts, ends = {}, {}
        for flight_id, (tasks, duration) in enumerate(pieces):
            start, end = self.event_index[tasks[0]], self.event_index[tasks[-1]]
            if end < start:
                return float('inf'), None
            starts.setdefault(start, []).append(flight_id)
            ends.setdefault(end, []).append(flight_id)
        available = list(range(self.instance.drone_count))
        records, active, visits = {}, {}, []
        clock, previous_s = 0., 0.
        for event in sorted(set(starts) | set(ends)):
            position = self.unique_positions[event]
            arrival = clock + (position-previous_s)/self.instance.vehicle_speed_mps
            clock = arrival
            ending = set(ends.get(event, []))
            launched, recovered = [], []

            def recover_ready():
                for fid in sorted(ending & set(active)):
                    if active[fid]['landing_time'] <= clock+1e-9:
                        rec = active.pop(fid)
                        rec['recovery_time'] = clock
                        heappush(available, rec['drone'])
                        ending.remove(fid)
                        recovered.append(fid)

            recover_ready()
            for fid in starts.get(event, []):
                if not available:
                    candidates = [active[k]['landing_time'] for k in ending if k in active]
                    if not candidates:
                        return float('inf'), None
                    clock = max(clock, min(candidates))
                    recover_ready()
                drone = heappop(available)
                tasks, duration = pieces[fid]
                rec = {'id': fid, 'drone': drone, 'tasks': [self.tasks[t] for t in tasks],
                       'launch_position_m': self.positions[tasks[0]],
                       'recovery_position_m': self.positions[tasks[-1]],
                       'launch_point': self.feet[tasks[0]], 'recovery_point': self.feet[tasks[-1]],
                       'launch_time': clock, 'landing_time': clock+duration,
                       'flight_and_service_seconds': duration}
                records[fid] = active[fid] = rec
                launched.append(fid)
                recover_ready()
            remaining = [active[k]['landing_time'] for k in ending if k in active]
            if remaining:
                clock = max(clock, max(remaining))
                recover_ready()
            if ending:
                return float('inf'), None
            if materialize:
                visits.append({'position_m': position, 'arrival_time': arrival,
                               'departure_time': clock, 'launches': launched,
                               'recoveries': recovered, 'drones_on_vehicle_at_departure': len(available)})
            previous_s = position
        if active or len(available) != self.instance.drone_count:
            return float('inf'), None
        makespan = clock + (self.prefix[-1]-previous_s)/self.instance.vehicle_speed_mps
        if not materialize:
            return makespan, None
        return makespan, {'vehicle_route': self.vehicle_route,
                          'sequence': [self.tasks[t] for t in sequence],
                          'road_length_m': self.prefix[-1], 'return_time_seconds': makespan,
                          'visits': visits, 'flights': [records[k] for k in sorted(records)]}


def transform(sequence, operator, decoder, rng):
    """The four transformations of Section 4.2.4.1."""
    seq = list(sequence)
    n = len(seq)
    if n < 2:
        return tuple(seq)
    if operator == 0:
        a, b = rng.sample(range(n), 2)
        seq[a], seq[b] = seq[b], seq[a]
    elif operator == 1:
        pairs = decoder.pair_positions
        if not pairs:
            return tuple(seq)
        a, b = rng.choice(pairs)
        seq[a:a+2], seq[b:b+2] = seq[b:b+2], seq[a:a+2]
    elif operator == 2:
        a, b = sorted(rng.sample(range(n), 2))
        seq[a:b+1] = seq[a:b+1][::-1]
    elif operator == 3:
        task = rng.choice(seq)
        nearest = min((x for x in seq if x != task), key=lambda x: (decoder.flight[task][x], x))
        seq.remove(task)
        seq.insert(seq.index(nearest)+1, task)
    else:
        raise ValueError('Unknown transformation')
    return tuple(seq)


def acceptance_probability(old, new, temperature):
    """Metropolis interpretation R02; infeasible moves remain rejected."""
    if not isfinite(new):
        return 0.
    if new <= old:
        return 1.
    return exp(-(new-old)/temperature)


def run_tsha(instance, settings):
    """Algorithms 1, 3 and 6, with the frozen interpretation specification."""
    tick = perf_counter()
    route, construction = construct_vehicle_route(instance)
    decoder = RouteDecoder(instance, route)
    current = best = decoder.initial_sequence()
    current_cost, _ = decoder.evaluate(current)
    initial_cost = best_cost = current_cost
    if not isfinite(current_cost):
        raise ValueError('Native initial sequence is infeasible; no substitute initializer used')
    rng = Random(settings.seed)
    weights, history = [1.]*4, set()
    total_uses, total_accepted = [0]*4, 0
    temperature, stagnant = settings.initial_temperature, 0
    trace, moves, stop = [], [], 'temperature_limit'
    while temperature > settings.temperature_stop and stagnant < settings.successive_cooldowns:
        before_best = best_cost
        scores, counts, infeasible, repeated = [1.]*4, [0]*4, 0, 0
        for iteration in range(settings.iterations_per_temperature):
            operator = rng.choices(range(4), weights=weights, k=1)[0]
            candidate = transform(current, operator, decoder, rng)
            used = [operator]
            counts[operator] += 1
            total_uses[operator] += 1
            retries_this_move = 0
            while candidate in history and retries_this_move < settings.history_retry_limit:
                operator = rng.randrange(4)
                candidate = transform(candidate, operator, decoder, rng)
                used.append(operator)
                counts[operator] += 1
                total_uses[operator] += 1
                repeated += 1
                retries_this_move += 1
            if candidate in history:
                stop = 'history_retry_limit'
                break
            candidate_cost, _ = decoder.evaluate(candidate)
            old_cost = current_cost
            probability = acceptance_probability(current_cost, candidate_cost, temperature)
            accepted = probability > rng.random()
            if not isfinite(candidate_cost):
                score = 0.
                infeasible += 1
            elif candidate_cost < current_cost:
                score = 1 + (current_cost-candidate_cost)/current_cost  # eq. 41
            elif accepted:
                score = 1 + (candidate_cost-current_cost)/candidate_cost  # eq. 42
            else:
                score = 1 + (current_cost-candidate_cost)/candidate_cost  # eq. 43
            for op in set(used):
                scores[op] += score
            for op in range(4):
                if counts[op]:
                    weights[op] = settings.rho*weights[op] + (1-settings.rho)*scores[op]/counts[op]
            if accepted:
                current, current_cost = candidate, candidate_cost
                total_accepted += 1
            if current_cost < best_cost-1e-9:
                best, best_cost = current, current_cost
            moves.append({'temperature': temperature, 'operators': used,
                          'candidate_objective': candidate_cost if isfinite(candidate_cost) else None,
                          'old_current_objective': old_cost, 'current_objective': current_cost,
                          'best_objective': best_cost, 'accepted': accepted,
                          'acceptance_probability': probability,
                          'accepted_sequence': list(current) if accepted else None})
        history.add(best)
        stagnant = stagnant+1 if abs(best_cost-before_best) <= 1e-9 else 0
        trace.append({'temperature': temperature, 'best_return_time_seconds': best_cost,
                      'current_return_time_seconds': current_cost, 'weights': list(weights),
                      'operator_uses': counts, 'infeasible_candidates': infeasible,
                      'history_retries': repeated, 'stagnant_temperatures': stagnant})
        if stop == 'history_retry_limit':
            break
        temperature *= settings.cooling_rate
    if stagnant >= settings.successive_cooldowns:
        stop = 'successive_cooldowns'
    recomputed, solution = decoder.evaluate(best, materialize=True)
    if abs(recomputed-best_cost) > 1e-7:
        raise ValueError('Saved best sequence and objective disagree')
    return {'method': 'VAMDIRP-TSHA explicit-interpretation reimplementation',
            'scope': 'native_static_one_vehicle', 'paper11_comparison_ready': False,
            'settings': asdict(settings), 'initial_return_time_seconds': initial_cost,
            'best_return_time_seconds': best_cost,
            'relative_reduction': (initial_cost-best_cost)/initial_cost,
            'runtime_seconds': perf_counter()-tick, 'stop_reason': stop,
            'total_operator_uses': total_uses, 'accepted_moves': total_accepted,
            'construction': construction, 'solution': solution, 'trace': trace, 'moves': moves}
