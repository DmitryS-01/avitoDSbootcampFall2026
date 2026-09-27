"""Однократная подготовка весов; ноутбуки используют только локальную модель."""
from pathlib import Path
import json
from huggingface_hub import HfApi, snapshot_download

root = Path(__file__).resolve().parents[1]
model = 'intfloat/multilingual-e5-small'
revision = HfApi().model_info(model).sha
path = snapshot_download(model, revision=revision, local_dir=root / 'outputs/models/multilingual-e5-small',
                         allow_patterns=['*.json', '*.safetensors', 'sentencepiece.bpe.model', '*.txt', 'README.md'],
                         ignore_patterns=['onnx/*', 'openvino/*'], max_workers=3)
(root / 'outputs/models/dense_manifest.json').write_text(json.dumps({'model': model, 'revision': revision, 'path': 'outputs/models/multilingual-e5-small'}, indent=2))
print(path)
