"""Run a real decision and report Ollama GPU residency. No model downloads."""
import argparse
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from winnow_ollama import decide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-gpu', action='store_true', help='Fail if the model is not fully resident on GPU')
    parser.add_argument('--model', default='winnow-e4b')
    args = parser.parse_args()
    request = json.loads((ROOT / 'examples' / 'refund.json').read_text(encoding='utf-8-sig'))
    cold = decide(request, model=args.model)
    warm = decide(request, model=args.model)
    with urllib.request.urlopen('http://127.0.0.1:11434/api/ps', timeout=10) as response:
        models = json.load(response)['models']
    model = next((m for m in models if m['name'] in (args.model, args.model + ':latest')), None)
    fully_gpu = bool(model and model.get('size_vram', 0) > 0 and model['size_vram'] >= model['size'])
    answers = warm['answers']
    correct = answers['department']['choice'] == 'billing' and answers['refund']['noul'] >= 0.5
    print(json.dumps({'correct': correct, 'fully_on_gpu': fully_gpu, 'runtime': model,
                      'first_seconds': cold['adapter']['elapsed_seconds'],
                      'warm_seconds': warm['adapter']['elapsed_seconds'], 'answers': answers}, ensure_ascii=False, indent=2))
    if not correct or (args.require_gpu and not fully_gpu):
        raise SystemExit(1)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
