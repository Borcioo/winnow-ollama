"""Run visual consistency experiments on a local model, or render a saved report offline."""
import argparse
import datetime
import hashlib
import json
import pathlib
import random
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from winnow_ollama import decide
from consistency_suite import build_scenarios, finalize


def get(base, route):
    with urllib.request.urlopen(base.rstrip('/')+route, timeout=10) as response:
        return json.load(response)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def render(data, target):
    # Prevent a model/user string from terminating the JSON script element.
    encoded = json.dumps(finalize(data), ensure_ascii=False, allow_nan=False)
    encoded = encoded.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    encoded = encoded.replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    template = (ROOT/'report'/'template.html').read_text(encoding='utf-8')
    if template.count('__REPORT_DATA__') != 1:
        raise ValueError('Template must contain exactly one data placeholder')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace('__REPORT_DATA__', encoded), encoding='utf-8')


def gpu_model(runtime, name):
    model = next((m for m in runtime['models'] if m['name'] in (name, name+':latest')), None)
    return model if model and model.get('size_vram', 0)>0 and model['size_vram']>=model['size'] else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260930, help='Scenario scheduling seed; not token sampling')
    parser.add_argument('--model', default='winnow-e4b')
    parser.add_argument('--url', default='http://127.0.0.1:11434')
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--suite', choices=['all', 'trolley', 'risk', 'intent'], default='all')
    parser.add_argument('--output', type=pathlib.Path, default=ROOT/'reports'/'consistency.html')
    parser.add_argument('--render', type=pathlib.Path, help='Render saved JSON without calling the model')
    args = parser.parse_args()
    if args.render:
        render(json.loads(args.render.read_text(encoding='utf-8-sig')), args.output)
        print(args.output.resolve())
        return
    if not 2 <= args.repeats <= 1000:
        parser.error('--repeats must be between 2 and 1000')
    cases = [c for c in build_scenarios() if args.suite=='all' or c['suite']==args.suite]
    version = get(args.url, '/api/version')['version']
    print('Warming model; excluded from trial metrics...', flush=True)
    warm_start = time.perf_counter()
    decide({'state': cases[0]['state'], 'questions': {'decision': cases[0]['question']}}, args.url, args.model)
    runtime = get(args.url, '/api/ps')
    if args.require_gpu and not gpu_model(runtime, args.model):
        raise RuntimeError('Model is not fully on GPU; inspect ollama ps before continuing')
    try:
        revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    try:
        dirty = bool(subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    fixture_hash = hashlib.sha256(json.dumps(cases, ensure_ascii=False).encode()).hexdigest()
    data = {'meta': {'model': args.model, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'repeats': args.repeats, 'seed': args.seed, 'version': version, 'runtime': runtime,
        'git_head': revision, 'git_dirty': dirty, 'fixture_sha256': fixture_hash,
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('scripts/visual_tests.py', 'scripts/consistency_suite.py', 'winnow_ollama.py')},
        'adapter_sha256': hashlib.sha256((ROOT/'winnow_ollama.py').read_bytes()).hexdigest(),
        'warmup_seconds': time.perf_counter()-warm_start, 'complete': False,
        'methodology': 'Repeated deterministic argmax decisions, not independent random samples. '
            'One request per scenario per round; seeded shuffled order within rounds. No chat history. '
            'Model/cache remain warm. Scheduling seed is not a sampling seed; adapter token seed stays 42. '
            'Moral/preference scenarios have no correct label. Facts and policy changes are not invariance tests. '
            'Accuracy is descriptive over successful labeled trials; errors are reported separately. '
            'Latency excludes initial warmup but includes HTTP; percentiles use linear interpolation.'},
        'scenarios': cases}
    output_json = args.output.with_suffix('.json')
    rng = random.Random(args.seed)
    started = time.perf_counter()
    try:
        for repeat in range(1, args.repeats+1):
            order = list(cases)
            rng.shuffle(order)
            for case in order:
                start = time.perf_counter()
                trial = {'repeat': repeat, 'choice': None, 'probabilities': {},
                         'elapsed_seconds': None, 'load_seconds': None, 'error': None}
                try:
                    result = decide({'state':case['state'], 'questions':{'decision':case['question']}}, args.url, args.model)
                    answer = result['answers']['decision']
                    detail = result['adapter']['details'][0]
                    trial.update(choice=answer['choice'], probabilities=answer['probabilities'],
                        elapsed_seconds=result['adapter']['elapsed_seconds'],
                        load_seconds=(detail['load_duration'] or 0)/1e9)
                except Exception as exc:
                    trial.update(error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter()-start)
                case['trials'].append(trial)
            finalize(data)
            write_json(output_json, data)
            print(f'Round {repeat}/{args.repeats}: {data["overall"]["calls"]} calls, '
                  f'{data["overall"]["errors"]} errors; {time.perf_counter()-started:.1f} s', flush=True)
        data['meta']['runtime_end'] = get(args.url, '/api/ps')
        data['meta']['gpu_at_end'] = bool(gpu_model(data['meta']['runtime_end'], args.model))
        data['meta']['complete'] = True
    finally:
        data['meta']['wall_seconds'] = time.perf_counter()-started
        write_json(output_json, finalize(data))
        if (ROOT/'report'/'template.html').exists():
            render(data, args.output)
    print(json.dumps(data['overall']), flush=True)
    print(args.output.resolve(), flush=True)
    if data['overall']['errors'] or (args.require_gpu and not data['meta'].get('gpu_at_end')):
        raise SystemExit(1)


if __name__=='__main__':
    main()
