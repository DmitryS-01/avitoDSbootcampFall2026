"""Загрузка и одна нормализация для обучения и инференса."""
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QUERY_COLS = ['search_query', 'search_location_id', 'search_is_delivery_search',
              'search_infm_params_text', 'search_category']


def normalize(value):
    if pd.isna(value):
        return ''
    text = unicodedata.normalize('NFKC', str(value)).lower().replace('ё', 'е')
    return ' '.join(re.findall(r'[\w]+', text))


def data_dir():
    path = Path(os.environ.get('AVITO_DATA_DIR', ROOT / 'dataset'))
    if not (path / 'train.parquet').exists():
        raise FileNotFoundError(f'Положите три Parquet в {path} или задайте AVITO_DATA_DIR')
    return path


def load_raw():
    return tuple(pd.read_parquet(data_dir() / f'{name}.parquet')
                 for name in ['train', 'benchmark_queries', 'benchmark_items'])


def query_keys(frame):
    """Полный контекст сохраняет различие фильтров; cold split группируется по тексту."""
    values = frame[QUERY_COLS].copy()
    for col in QUERY_COLS:
        values[col] = values[col].map(normalize) if col in ['search_query', 'search_infm_params_text'] else values[col].astype('string').fillna('<NA>')
    return values.apply(lambda row: json.dumps(row.tolist(), ensure_ascii=False, separators=(',', ':')), axis=1)


def prepare_queries(frame):
    result = frame[QUERY_COLS + (['query_id'] if 'query_id' in frame else [])].copy()
    result['text_key'] = result.search_query.map(normalize)
    result['context_key'] = query_keys(result)
    if 'query_id' not in result:
        result['query_id'] = result.context_key.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:24])
    result['query_text'] = result.text_key + ' ' + result.search_infm_params_text.map(normalize)
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
