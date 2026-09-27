"""Выполняет выбранные ноутбуки; сохраняет также ноутбук с ошибкой для диагностики."""
import argparse
import json
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('names', nargs='*')
args = parser.parse_args()
runtime = root / '.runtime'
kernel = runtime / 'kernels' / 'avito'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': 'Avito local', 'language': 'python'}))
os.environ['JUPYTER_PATH'] = str(runtime)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime)
os.environ.setdefault('MPLCONFIGDIR', str(runtime / 'matplotlib'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
paths = [root / 'notebooks' / name for name in args.names] if args.names else sorted((root / 'notebooks').glob('*.ipynb'))
for path in paths:
    print('RUN', path.name, flush=True)
    notebook = nbformat.read(path, as_version=4)
    def save_progress(**kwargs):
        nbformat.write(notebook, path)
        print('  cell', kwargs['cell_index'], 'done', flush=True)
    client = NotebookClient(notebook, timeout=None, kernel_name='avito',
                            on_cell_executed=save_progress, resources={'metadata': {'path': str(root)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print('DONE', path.name, flush=True)
