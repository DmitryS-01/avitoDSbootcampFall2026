"""Метрика и строгий контракт сдачи. Пустая выдача разрешена условием."""
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


def recall_table(predictions, truth, ks=(1, 10, 20, 50, 100)):
    rows = []
    for query_id, relevant in truth.items():
        relevant = set(relevant)
        if not relevant:
            raise ValueError(f'Пустая разметка: {query_id}')
        ranked = predictions.get(query_id, [])
        if len(ranked) != len(set(ranked)):
            raise ValueError(f'Повторы в выдаче: {query_id}')
        rows.append({'query_id': query_id, 'n_relevant': len(relevant),
                     **{f'recall@{k}': len(set(ranked[:k]) & relevant) / len(relevant) for k in ks}})
    if not rows:
        raise ValueError('Нет запросов для оценки')
    return pd.DataFrame(rows)


def bootstrap_ci(values, seed=143, repeats=2000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(repeats)]
    return np.quantile(means, [.025, .975]).tolist()


def validate_submission(path, query_ids, item_ids):
    expected = list(query_ids)
    if len(set(expected)) != len(expected) or any(not isinstance(q, str) or len(q) != 16 for q in expected):
        raise ValueError('Некорректные query_id в исходном benchmark')
    allowed = set(item_ids)
    if any(not isinstance(i, str) or re.fullmatch('[0-9a-f]{16}', i) is None for i in allowed):
        raise ValueError('Некорректные item_id в исходном корпусе')
    with open(path, encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != ['query_id', 'answer']:
            raise ValueError('Ожидаются ровно query_id,answer в UTF-8 без BOM')
        rows = list(reader)
    seen = set()
    sizes = []
    for line, row in enumerate(rows, 2):
        if len(row) != 2:
            raise ValueError(f'Строка {line}: не две колонки')
        query_id, answer = row
        if query_id not in expected or query_id in seen:
            raise ValueError(f'Строка {line}: лишний или повторный query_id')
        seen.add(query_id)
        items = answer.split(' ') if answer else []
        if len(items) > 50 or len(set(items)) != len(items):
            raise ValueError(f'Строка {line}: >50 или повторные item_id')
        if any(re.fullmatch('[0-9a-f]{16}', item) is None or item not in allowed for item in items):
            raise ValueError(f'Строка {line}: неправильный item_id или разделитель')
        sizes.append(len(items))
    if seen != set(expected):
        raise ValueError(f'Пропущено {len(set(expected) - seen)} query_id')
    return {'rows': len(rows), 'min_candidates': min(sizes), 'max_candidates': max(sizes), 'valid': True}


def save_submission(predictions, queries, items, path):
    if set(predictions) != set(queries.query_id):
        raise ValueError('Набор query_id предсказаний не совпадает с benchmark')
    frame = pd.DataFrame({'query_id': queries.query_id,
                          'answer': [' '.join(predictions[q]) for q in queries.query_id]})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8')
    return validate_submission(path, queries.query_id.tolist(), items.item_id.tolist())
