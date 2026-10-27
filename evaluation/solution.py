"""score и quality_gate для ДЗ 6+7."""
import json
import math


def _parse_output(raw):
    if not raw:
        return None, False
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, False
    if not isinstance(parsed, dict) or list(parsed.keys()) != ['answer']:
        return None, False
    if not isinstance(parsed['answer'], str):
        return None, False
    return parsed, True


def score(rows, outputs):
    if not rows or not outputs:
        raise ValueError('Пустой набор rows или outputs')
    row_ids = [r['id'] for r in rows]
    out_ids = [o['id'] for o in outputs]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError('Дубли id в rows')
    if len(out_ids) != len(set(out_ids)):
        raise ValueError('Дубли id в outputs')
    if set(row_ids) != set(out_ids):
        raise ValueError('Несовпадающие id rows/outputs')
    by_id = {o['id']: o for o in outputs}

    per_item = []
    slice_acc = {}

    for row in rows:
        out = by_id[row['id']]
        raw = out.get('raw', '') or ''
        error = out.get('error')
        parsed, is_json = _parse_output(raw) if not error else (None, False)
        valid = is_json and parsed['answer'] in row['choices']
        correct = valid and parsed['answer'] == row['expected']
        per_item.append(dict(id=row['id'], raw=raw, parsed=parsed, valid=valid,
                             correct=correct, error=error, slices=row['slices']))
        for s in row['slices']:
            st = slice_acc.setdefault(s, dict(n=0, correct=0, valid=0, errors=0))
            st['n'] += 1
            st['correct'] += int(correct)
            st['valid'] += int(valid)
            st['errors'] += int(bool(error))

    n = len(rows)
    correct_n = sum(1 for x in per_item if x['correct'])
    valid_n = sum(1 for x in per_item if x['valid'])
    errors_n = sum(1 for x in per_item if x['error'])

    overall = dict(n=n, correct=correct_n, valid=valid_n, errors=errors_n,
                   accuracy=correct_n / n, format_rate=valid_n / n)
    slices = {}
    for s, st in slice_acc.items():
        slices[s] = dict(n=st['n'], correct=st['correct'], valid=st['valid'],
                         errors=st['errors'],
                         accuracy=st['correct'] / st['n'],
                         format_rate=st['valid'] / st['n'])
    return dict(overall=overall, slices=slices, per_item=per_item)


def _finite(value, name):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name}: не конечное число')


def _check_report(report, tag, require_behavioral_pass=False):
    if not isinstance(report, dict):
        raise ValueError(f'{tag}: не словарь')
    if report.get('status') != 'complete':
        raise ValueError(f'{tag}: status != complete')
    if not report.get('model'):
        raise ValueError(f'{tag}: не указана модель')
    overall = report.get('overall')
    if not overall:
        raise ValueError(f'{tag}: нет overall')
    n = overall.get('n', 0)
    if n < 50:
        raise ValueError(f'{tag}: golden < 50 (n={n})')
    _finite(overall.get('accuracy'), f'{tag}.accuracy')
    _finite(overall.get('format_rate'), f'{tag}.format_rate')
    correct = overall.get('correct', 0)
    valid = overall.get('valid', 0)
    errors = overall.get('errors', 0)
    if correct > valid or valid > n or errors < 0:
        raise ValueError(f'{tag}: counts некорректны')
    if abs(overall['accuracy'] - correct / n) > 1e-9:
        raise ValueError(f'{tag}: accuracy != correct/n')
    if abs(overall['format_rate'] - valid / n) > 1e-9:
        raise ValueError(f'{tag}: format_rate != valid/n')
    if errors > 0:
        raise ValueError(f'{tag}: есть ошибки генерации ({errors})')
    per_item = report.get('per_item') or []
    if len(per_item) != n:
        raise ValueError(f'{tag}: per_item != n')
    ids = [x['id'] for x in per_item]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{tag}: дубли id в per_item')
    slices = report.get('slices') or {}
    if len(slices) < 3:
        raise ValueError(f'{tag}: меньше 3 срезов')
    for s, st in slices.items():
        if st['n'] <= 0:
            raise ValueError(f'{tag}: пустой срез {s}')
        _finite(st.get('accuracy'), f'{tag}.slices.{s}.accuracy')
        if abs(st['accuracy'] - st['correct'] / st['n']) > 1e-9:
            raise ValueError(f'{tag}: accuracy среза {s} != counts')
    beh = report.get('behavioral') or {}
    bo = beh.get('overall') or {}
    if bo.get('n', 0) < 3:
        raise ValueError(f'{tag}: behavioral < 3')
    if require_behavioral_pass and bo.get('correct') != bo.get('n'):
        raise ValueError(f'{tag}: не все behavioral прошли ({bo.get("correct")}/{bo.get("n")})')


def quality_gate(baseline, candidate, critical_slices):
    for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
        if key not in baseline or key not in candidate:
            raise ValueError(f'Нет {key}')
        if baseline[key] != candidate[key]:
            raise ValueError(f'Несовместимый {key}')
    _check_report(baseline, 'baseline', require_behavioral_pass=False)
    _check_report(candidate, 'candidate', require_behavioral_pass=True)

    base_ids = {x['id'] for x in baseline['per_item']}
    cand_ids = {x['id'] for x in candidate['per_item']}
    if base_ids != cand_ids:
        raise ValueError('Разные id в per_item')
    if set(baseline['slices']) != set(candidate['slices']):
        raise ValueError('Разные срезы')
    for s in baseline['slices']:
        if baseline['slices'][s]['n'] != candidate['slices'][s]['n']:
            raise ValueError(f'Размер среза {s} не совпадает')

    if candidate['overall']['accuracy'] < baseline['overall']['accuracy']:
        raise ValueError('accuracy кандидата ниже baseline')
    if candidate['overall']['format_rate'] < baseline['overall']['format_rate']:
        raise ValueError('format_rate кандидата ниже baseline')

    for s in critical_slices:
        if s not in baseline['slices']:
            raise ValueError(f'Критический срез {s} отсутствует в baseline')
        if s not in candidate['slices']:
            raise ValueError(f'Критический срез {s} отсутствует в candidate')
        if candidate['slices'][s]['accuracy'] < baseline['slices'][s]['accuracy']:
            raise ValueError(f'Критический срез {s} регрессировал: '
                             f'{candidate["slices"][s]["accuracy"]} < '
                             f'{baseline["slices"][s]["accuracy"]}')
    return dict(passed=True, reason='Все проверки пройдены')