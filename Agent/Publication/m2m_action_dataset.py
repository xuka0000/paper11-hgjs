"""Frozen, resumable development labels for joint service actions.

Each state is a checked physical plan.  Each label replays the *entire* fixed
service set through the route DP and independent evaluator; a search rejection
is never used as a surrogate cost or feasibility certificate.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter

from Agent.Publication.budget import BudgetExpired, Deadline
from Agent.Publication.component_objective import evaluate_plan
from Agent.Publication.foundation_benchmark import WEIGHTS
from Agent.Publication.inspection_tsha_adapter import run_inspection_tsha_adapter
from Agent.Publication.joint_service_actions import ActionProposal, _PHYSICAL_REJECTIONS
from Agent.Publication.joint_service_search import (ActionContext, ConditionalHeuristic,
    EpsilonGreedy, SearchLimits, iter_proposals)
from Agent.Publication.m2m_public_learning import load_case
from Agent.Publication.many_to_many_model import (CompletionQuota, ManyToManyModel,
    ServiceAlternative, SharedResource)
from Agent.Run.optimize_publication_hamming_routes import optimize_committed_routes
from Agent.Run.publication_committed_service_domain import Plan

VERSION = 'm2m-action-dataset-v1'
TEMPLATES = ('empty', 'one_per_completed_component', 'full_warm',
             'prescribed_inspection_tsha_intermediate')
SOURCES = ('quota_guided', 'resource_guided', 'random')


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def _hash(value):
    return sha256(_canonical(value)).hexdigest()


def _write_json(path, value, *, compressed=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical(value)
    temp = path.with_name(path.name + '.tmp')
    if compressed:
        with temp.open('wb') as handle:
            with gzip.GzipFile(fileobj=handle, mode='wb', mtime=0) as stream:
                stream.write(raw)
    else:
        temp.write_bytes(raw)
    temp.replace(path)
    return sha256(path.read_bytes()).hexdigest()


def _read_checked(path, digest):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError('Stored dataset state hash mismatch: ' + str(path))
    return json.loads(gzip.decompress(raw))


def _write_envelope(path, payload):
    """Atomic JSON with a digest of the canonical payload."""
    _write_json(path, {'sha256': _hash(payload), 'payload': payload})


def _read_envelope(path):
    envelope = json.loads(Path(path).read_text(encoding='utf-8'))
    if _hash(envelope['payload']) != envelope['sha256']:
        raise ValueError('Journal payload hash mismatch: ' + str(path))
    return envelope['payload']


def _view_record(view):
    return {'version': view.version, 'now': view.now,
        'services': [asdict(s) for s in view.services.values()],
        'quotas': [{**asdict(q), 'task_ids': sorted(q.task_ids, key=str)} for q in view.quotas.values()],
        'resources': [asdict(r) for r in view.resources.values()],
        'route_requirements': view.route_requirements}


def context_from_record(record):
    """Reconstruct the input-only context used by all trained variants."""
    source = record['view']
    view = ManyToManyModel(
        [ServiceAlternative(s['service_id'], tuple(tuple(x) for x in s['completion_layers']),
                            s.get('vehicle_id'), s.get('uav_id')) for s in source['services']],
        [CompletionQuota(q['group_id'], frozenset(q['task_ids']), q['minimum'],
                         q['priority'], q['deadline']) for q in source['quotas']],
        [SharedResource(r['resource_id'], r['kind'], tuple(r['members']),
                        tuple(r['coefficients']), r['capacity'], r['fixed_load'])
         for r in source['resources']], now=source['now'],
        route_requirements=source.get('route_requirements'))
    incumbent = record['incumbent']
    return ActionContext(view, incumbent['selected_tuple_ids'], costs=incumbent['costs'],
                         immutable=incumbent['immutable'])


def iter_state_records(output):
    """Yield only hash-verified completed states from one case manifest."""
    output = Path(output)
    index = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    for row in index['states']:
        if row['status'] == 'completed':
            yield _read_checked(output / row['file'], row['sha256'])


def _fixed_plan(problem, groups, ids, seconds):
    started = perf_counter()
    try:
        deadline = Deadline(max(1e-3, seconds))
        copies = tuple(problem.full_copies[k] for k in sorted(ids))
        plan, route = optimize_committed_routes(problem.domain, copies, WEIGHTS, deadline=deadline)
        checked = evaluate_plan(problem.domain, plan, WEIGHTS, groups)
        deadline.check()
        return plan, checked, route, perf_counter()-started, None
    except BudgetExpired:
        return None, None, None, perf_counter()-started, 'timeout'
    except ValueError as exc:
        if str(exc).startswith(_PHYSICAL_REJECTIONS):
            return None, None, None, perf_counter()-started, str(exc)
        raise


def _state_plan(problem, groups, warm, template, seconds, seed):
    if template == 'empty':
        return Plan(), {'source': 'empty'}
    if template == 'full_warm':
        return warm, {'source': 'frozen_warm'}
    if template == 'one_per_completed_component':
        completed = evaluate_plan(problem.domain, warm, WEIGHTS, groups)['completed_groups']
        chosen = set()
        for group in groups.groups:
            if group.group_id not in completed:
                continue
            eligible = [c for c in warm.copies if any(t in group.task_ids and k <= group.deadline_layer
                        for t, k in c.completion_layers)]
            if not eligible:
                return None, {'reason': 'completed_group_has_no_warm_service', 'group_id': group.group_id}
            chosen.add(min(eligible, key=lambda c: c.tuple_id).tuple_id)
        plan, check, _, _, error = _fixed_plan(problem, groups, chosen, seconds)
        return plan, {'source': 'warm_one_per_completed_component', 'reason': error,
                      'selected_count': len(chosen)}
    if template == 'prescribed_inspection_tsha_intermediate':
        # Fixed seed and half of the 60 s comparator budget.  This is a
        # predeclared intermediate, not selected from a future best trace.
        result = run_inspection_tsha_adapter(problem, WEIGHTS, seed=seed,
                    total_seconds=min(30., max(2., seconds)), task_groups=groups)
        plan = problem.plan(replace(problem.state,
            selected_arc_ids=frozenset(result['selected_arc_ids']),
            selected_tuple_ids=frozenset(result['selected_tuple_ids'])))
        return plan, {'source': 'inspection_tsha_half_budget_best', 'seed': seed,
                      'budget_seconds': min(30., max(2., seconds)),
                      'actual_seconds': result['total_action_seconds'],
                      'stop_reason': result['stop_reason']}
    raise ValueError('Unknown state template')


class _ResourceHeuristic(ConditionalHeuristic):
    def score(self, context, partial, extensions, descriptors):
        return [5*d['primary_gain'] + d['quota_progress']
                - .1*d['direct_cost_delta'] + .02*len(d['affected_resources'])
                - .1*d['overload'] - .001*d['action_size'] for d in descriptors]


def _proposals(context, seed, seconds):
    """Balanced 8/8/8 nominal slots; unavailable slots stay visible."""
    policies = {'quota_guided': ConditionalHeuristic(),
                'resource_guided': _ResourceHeuristic(),
                'random': EpsilonGreedy(seed, epsilon=1.)}
    found, missing, seen = [], [], set()
    per_source_seconds = max(1e-3, seconds / len(SOURCES))
    source_order = list(SOURCES)
    Random(seed).shuffle(source_order)
    for source in source_order:
        policy = policies[source]
        deadline = Deadline(per_source_seconds)
        candidates = []
        exhausted = False
        try:
            for action, descriptor in iter_proposals(context, policy, deadline,
                    SearchLimits(terminals=32), excluded=seen):
                if action in seen:
                    continue
                candidates.append((source, action, descriptor))
        except BudgetExpired:
            exhausted = True
        # Reserve representation for exchanges and multistep additions.  The
        # order inside each stratum remains the input-only source ranking.
        joint = [x for x in candidates if len(x[1].add) >= 2 and x[1].release]
        release = [x for x in candidates if x[1].release and x not in joint]
        multi = [x for x in candidates if len(x[1].add) >= 2 and x not in joint]
        single = [x for x in candidates if x not in joint and x not in release and x not in multi]
        chosen = []
        for pool, quota in ((joint, 2), (release, 2), (multi, 2), (single, 2)):
            for x in pool[:quota]:
                if x[1] not in seen:
                    chosen.append(x)
                    seen.add(x[1])
        for x in candidates:
            if len(chosen) >= 8:
                break
            if x[1] not in seen:
                chosen.append(x)
                seen.add(x[1])
        found.extend(chosen)
        count = len(chosen)
        missing.extend({'source': source, 'slot': j+1,
                        'reason': 'proposal_generation_budget_censored' if exhausted else 'candidate_unavailable'}
                       for j in range(count, 8))
    # Round-robin order gives all sources equal access to a bounded state
    # label budget even when individual checks take their full allowance.
    buckets = {source: [row for row in found if row[0] == source] for source in source_order}
    balanced = [bucket[j] for j in range(8) for bucket in buckets.values() if j < len(bucket)]
    return balanced, missing


def _finish_intent(output, template, intent, *, resume_interrupted, pulse=None):
    """Commit each immutable label; never recalculate a previously attempted slot."""
    labels_dir = output / (template + '.labels')
    labels_dir.mkdir(exist_ok=True)
    proposals = []
    for i, proposal in enumerate(intent['proposals']):
        path = labels_dir / f'{i:02d}.json'
        if path.exists():
            saved = _read_envelope(path)
            if saved['proposal_sha256'] != _hash(proposal):
                raise ValueError('Label file belongs to a different proposed action')
            label = saved['label']
        elif resume_interrupted:
            label = _budget_censored_label()
            label['reason_detail'] = 'interrupted_after_intent_before_label_commit'
            label['rejection_reason'] = 'censored_interrupted'
            _write_envelope(path, {'proposal_sha256': _hash(proposal), 'label': label})
        else:
            raise RuntimeError('Interrupted state has uncommitted labels; require explicit resume_interrupted receipt')
        proposals.append({**proposal, 'label': label})
        if pulse is not None:
            pulse({'case_id': intent['base']['case_id'], 'template': template,
                   'label_count': i+1, 'label_status': label['status']})
    record = {**intent['base'], 'proposals': proposals,
              'checked_proposals': sum(x['label']['status']=='checked' for x in proposals),
              'continuations': _prefix_labels(proposals)}
    return record


def _label(problem, groups, incumbent, proposal, seconds):
    base = evaluate_plan(problem.domain, incumbent, WEIGHTS, groups)
    selected = {c.tuple_id for c in incumbent.copies}
    proposed = (selected-set(proposal.release)) | set(proposal.add)
    started = perf_counter()
    plan, checked, route, runtime, error = _fixed_plan(problem, groups, proposed, seconds)
    if plan is not None and ({c.tuple_id for c in plan.copies} != proposed
                             or len(plan.copies) != len(proposed)):
        raise RuntimeError('Fixed-service route solver changed requested service set')
    label = {'status': 'censored' if error == 'timeout' else 'infeasible' if error else 'checked',
             'physical_feasibility': None if error == 'timeout' else error is None,
             'gain': None if error else base['primary']-checked['primary'],
             'cost': None if error else checked['secondary'],
             'runtime_seconds': runtime, 'rejection_reason': error,
             'raw_plan': None if plan is None else {'selected_tuple_ids': sorted(c.tuple_id for c in plan.copies),
                  'selected_arc_ids': sorted(a.arc_id for a in plan.arcs)},
             'checker_version': 'component_objective.evaluate_plan+check_publication_hamming_schedule',
             'route_solver_version': 'optimize_committed_routes',
             'route': route}
    return label


def _budget_censored_label():
    return {'status': 'censored', 'physical_feasibility': None, 'gain': None,
            'cost': None, 'runtime_seconds': 0., 'rejection_reason': 'state_budget_exhausted',
            'raw_plan': None, 'checker_version': 'component_objective.evaluate_plan+check_publication_hamming_schedule',
            'route_solver_version': 'optimize_committed_routes', 'route': None}


def _prefix_labels(proposals):
    """Only observed, checked terminal actions can supervise partial prefixes."""
    prefixes = {}
    for row in proposals:
        label = row['label']
        if label['status'] != 'checked':
            continue
        add, release = row['add'], row['release']
        operations = [('add', x) for x in add] + [('release', x) for x in release]
        for n in range(1, len(operations)):
            key = tuple(operations[:n])
            entry = prefixes.setdefault(key, {'add': [x for k,x in key if k=='add'],
                'release': [x for k,x in key if k=='release'],
                'label': {'status': 'observed_terminal_continuation',
                          'physical_feasibility': None, 'gain': None, 'cost': None,
                          'checked_completions': 0}})
            prefix_label = entry['label']
            prefix_label['checked_completions'] += 1
            score = (label['gain'], -label['cost'])
            previous = (prefix_label['gain'], -prefix_label['cost']) if prefix_label['gain'] is not None else None
            if previous is None or score > previous:
                prefix_label['gain'], prefix_label['cost'] = label['gain'], label['cost']
    return list(prefixes.values())


def generate_case(batch: Path, manifest: dict, output: Path, seed=1701,
                  proposal_seconds=10., state_seconds=60., *, templates=TEMPLATES,
                  pulse=None, resume_interrupted=False):
    """Generate one frozen train/validation case, atomically and resumably.

    `output` is a case directory.  Existing states are hash checked before
    reuse.  A changed case hash, seed, budget or code version refuses resume.
    """
    if manifest.get('split') not in ('train', 'validation') or manifest.get('final_test'):
        raise ValueError('Action labels allow only frozen train/validation cases')
    if proposal_seconds <= 0 or state_seconds <= 0:
        raise ValueError('Budgets must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'version': VERSION, 'case_id': manifest['case_id'], 'split': manifest['split'],
                'pickle_sha256': manifest['pickle_sha256'], 'seed': seed,
                'proposal_seconds': proposal_seconds, 'state_seconds': state_seconds,
                'templates': list(templates),
                'dataset_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                'checker_sha256': sha256(Path(evaluate_plan.__code__.co_filename).read_bytes()).hexdigest(),
                'route_solver_sha256': sha256(Path(optimize_committed_routes.__code__.co_filename).read_bytes()).hexdigest()}
    index_path = output / 'manifest.json'
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding='utf-8'))
        if index['identity'] != identity:
            raise ValueError('Cannot resume with changed dataset identity')
        for row in index['states']:
            if row['status'] == 'completed':
                _read_checked(output / row['file'], row['sha256'])
    else:
        index = {'identity': identity, 'states': [], 'nominal_states': len(templates),
                 'completed_states': 0, 'skipped_states': 0}
    problem, groups, _ = load_case(Path(batch), manifest, 'warm')
    warm = problem.plan(problem.state)
    view = ManyToManyModel.from_problem(problem, groups)
    previous_failures = [r for r in index['states'] if r['status'] == 'failed']
    if previous_failures:
        if not resume_interrupted:
            raise RuntimeError('Prior failed state requires explicit diagnosed resume')
        index.setdefault('failures', []).extend(previous_failures)
        index['states'] = [r for r in index['states'] if r['status'] != 'failed']
        _write_json(index_path, index)
    existing = {r['template']: r for r in index['states']}
    seen_hashes = {r['state_hash'] for r in index['states'] if r.get('state_hash')}
    for template in templates:
        if template in existing:
            continue
        intent_path = output / (template + '.intent.json')
        if intent_path.exists():
            intent = _read_envelope(intent_path)
            if (intent['identity'] != identity or intent['base']['template'] != template):
                raise ValueError('Interrupted intent identity mismatch')
            record = _finish_intent(output, template, intent,
                                    resume_interrupted=resume_interrupted, pulse=pulse)
            name = template + '.json.gz'
            digest = _write_json(output / name, record, compressed=True)
            row = {'template': template, 'status': 'completed', 'state_hash': record['state_hash'],
                   'file': name, 'sha256': digest, 'proposals': len(record['proposals'])}
            seen_hashes.add(record['state_hash'])
            index['states'].append(row)
            index['completed_states'] = sum(r['status']=='completed' for r in index['states'])
            index['skipped_states'] = sum(r['status']=='skipped' for r in index['states'])
            _write_json(index_path, index)
            continue
        try:
            plan, provenance = _state_plan(problem, groups, warm, template, state_seconds, seed)
            if plan is None:
                raise ValueError(provenance.get('reason', 'state_unavailable'))
            checked = evaluate_plan(problem.domain, plan, WEIGHTS, groups)
            arcs = sorted(a.arc_id for a in plan.arcs)
            services = sorted(c.tuple_id for c in plan.copies)
            state_hash = _hash({'case_sha256': manifest['pickle_sha256'], 'arcs': arcs, 'services': services})
            if state_hash in seen_hashes:
                raise ValueError('duplicate_actual_arcs_and_services')
            costs = {k: c.secondary_cost + WEIGHTS['risk_weight']*c.risk_exposure
                     + WEIGHTS['energy_weight']*c.expected_energy for k,c in problem.full_copies.items()}
            context = ActionContext(view, services, costs=costs,
                immutable=set(services) & set(problem.domain.context.state.locked_tuple_ids))
            state_deadline = Deadline(state_seconds)
            state_seed = seed + int(_hash(manifest['case_id'])[:8], 16) + len(index['states'])
            available, missing = _proposals(context, state_seed,
                                             min(state_deadline.remaining() * .25, proposal_seconds * 3))
            base = {'schema_version': VERSION, 'case_id': manifest['case_id'],
                'split': manifest['split'], 'state_id': manifest['case_id'] + ':' + template,
                'template': template, 'state_hash': state_hash, 'case_sha256': manifest['pickle_sha256'],
                'provenance': provenance, 'view': _view_record(view),
                'incumbent': {'selected_tuple_ids': services, 'selected_arc_ids': arcs,
                    'objective': [checked['primary'], checked['secondary']], 'costs': costs,
                    'immutable': sorted(context.immutable)},
                'unavailable_proposal_slots': missing, 'nominal_proposals': 24}
            intent = {'identity': identity, 'base': base,
                      'proposals': [{'source': source, 'add': list(action.add),
                          'release': list(action.release), 'descriptor': descriptor}
                          for source, action, descriptor in available]}
            _write_envelope(intent_path, intent)
            labels_dir = output / (template + '.labels')
            labels_dir.mkdir(exist_ok=True)
            for i, (source, action, descriptor) in enumerate(available):
                remaining = state_deadline.remaining()
                label = (_label(problem, groups, plan, action, min(proposal_seconds, remaining))
                         if remaining > 0 else _budget_censored_label())
                _write_envelope(labels_dir / f'{i:02d}.json',
                                {'proposal_sha256': _hash(intent['proposals'][i]), 'label': label})
                if pulse is not None:
                    pulse({'case_id': manifest['case_id'], 'template': template,
                           'label_count': i+1, 'label_status': label['status']})
            record = _finish_intent(output, template, intent, resume_interrupted=False)
            name = template + '.json.gz'
            digest = _write_json(output / name, record, compressed=True)
            seen_hashes.add(state_hash)
            row = {'template': template, 'status': 'completed', 'state_hash': state_hash,
                   'file': name, 'sha256': digest, 'proposals': len(record['proposals'])}
        except BudgetExpired as exc:
            row = {'template': template, 'status': 'skipped', 'reason': 'state_budget_exhausted: '+str(exc)}
        except ValueError as exc:
            if (str(exc) not in ('duplicate_actual_arcs_and_services',
                                 'completed_group_has_no_warm_service', 'timeout')
                    and not str(exc).startswith(_PHYSICAL_REJECTIONS)):
                index['states'].append({'template': template, 'status': 'failed',
                    'error_type': type(exc).__name__, 'reason': str(exc)})
                _write_json(index_path, index)
                raise
            row = {'template': template, 'status': 'skipped', 'reason': str(exc)}
        except Exception as exc:
            index['states'].append({'template': template, 'status': 'failed',
                'error_type': type(exc).__name__, 'reason': str(exc)})
            _write_json(index_path, index)
            raise
        index['states'].append(row)
        index['completed_states'] = sum(r['status']=='completed' for r in index['states'])
        index['skipped_states'] = sum(r['status']=='skipped' for r in index['states'])
        _write_json(index_path, index)
        if pulse is not None:
            pulse({'case_id': manifest['case_id'], 'template': template,
                   'state_status': row['status'], 'completed_states': index['completed_states'],
                   'skipped_states': index['skipped_states']})
    return index
