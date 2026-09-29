"""Reproduce the eight-ticket synthetic evaluation; not a general benchmark."""
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from winnow_ollama import decide


def main():
    fixture = json.loads((ROOT / 'examples' / 'evaluation.json').read_text(encoding='utf-8'))
    rows = []
    for name, state, department, refund in fixture['cases']:
        result = decide({'state': state, 'questions': fixture['questions']})
        answers = result['answers']
        checks = [answers['department']['choice'] == department, (answers['refund']['noul'] >= 0.5) == refund]
        row = {'case': name, 'checks': checks, 'result': result}
        rows.append(row)
        print(json.dumps({'case': name, 'checks': checks, 'seconds': result['adapter']['elapsed_seconds']}), flush=True)
    summary = {'correct': sum(sum(r['checks']) for r in rows), 'total': len(rows)*2,
               'warm_median_seconds': statistics.median(r['result']['adapter']['elapsed_seconds'] for r in rows[1:]),
               'note': 'First request excluded from warm median. Urgency has no correctness label.', 'rows': rows}
    (ROOT / 'test-results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{summary['correct']}/{summary['total']} checks; warm median {summary['warm_median_seconds']:.3f} s")
    if summary['correct'] != summary['total']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
