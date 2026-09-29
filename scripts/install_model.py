"""Download pinned original Winnow-E4B Q8 weights, verify SHA-256 and import into Ollama."""
import argparse
import hashlib
import pathlib
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request

REVISION = '734302fe5fbfeb3f21a7ece62653c9539be4aaf3'
SHA256 = '840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2'
URL = f'https://huggingface.co/EldanRing/Winnow-E4B/resolve/{REVISION}/gguf/Winnow-E4B-Q8_0.gguf'


def verify(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise ValueError(f'SHA-256 mismatch for {path}; file has not been imported')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=pathlib.Path, help='Reuse an existing original Q8 GGUF (or Ollaya blob)')
    parser.add_argument('--model', default='winnow-e4b', help='New Ollama model name (default: winnow-e4b)')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9._:-]*', args.model):
        parser.error('Use a simple lowercase Ollama model name')
    ollama = shutil.which('ollama')
    if not ollama:
        parser.error('Install Ollama and open a new terminal first: https://ollama.com/download')
    # Do not download 8 GB only to discover the server is unavailable.
    try:
        with urllib.request.urlopen('http://127.0.0.1:11434/api/version', timeout=5) as response:
            print('Ollama:', response.read().decode(), flush=True)
    except urllib.error.URLError:
        parser.error('Start Ollama on localhost:11434 first')
    if args.gguf:
        path = args.gguf.resolve(strict=True)
    else:
        path = pathlib.Path(__file__).resolve().parents[1] / 'models' / 'Winnow-E4B-Q8_0.gguf'
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            partial = path.with_suffix('.gguf.part')
            print('Downloading original Q8 weights (8.01 GB). Allow room for the download and Ollama import.', flush=True)
            try:
                with urllib.request.urlopen(URL, timeout=120) as response, partial.open('wb') as target:
                    copied = 0
                    next_update = 512 * 1024 * 1024
                    while chunk := response.read(8 * 1024 * 1024):
                        target.write(chunk)
                        copied += len(chunk)
                        if copied >= next_update:
                            print(f'{copied / 1e9:.2f} GB downloaded', flush=True)
                            next_update += 512 * 1024 * 1024
                verify(partial)
                partial.replace(path)
            except Exception:
                print('Download failed. Re-run to restart; partial data is not imported.', flush=True)
                raise
    print('Verifying the original SHA-256...', flush=True)
    verify(path)
    # Quoted absolute path supports spaces; no shell interpolation is used.
    if any(char in str(path) for char in ('"', '\n', '\r')):
        raise ValueError('Unsupported character in GGUF path')
    with tempfile.TemporaryDirectory(prefix='winnow-ollama-') as directory:
        modelfile = pathlib.Path(directory) / 'Modelfile'
        modelfile.write_text(f'FROM "{path.as_posix()}"\nPARAMETER num_ctx 8192\nPARAMETER temperature 0\n', encoding='utf-8')
        subprocess.run([ollama, 'create', args.model, '-f', str(modelfile)], check=True)
    print(f'Ready: {args.model}. Run the GPU smoke test described in README.md.', flush=True)


if __name__ == '__main__':
    main()
