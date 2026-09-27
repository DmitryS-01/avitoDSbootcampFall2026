"""Одинаковое построение признаков fusion при обучении и финальном инференсе."""
import numpy as np
import pandas as pd
from .retrieval import rrf
from .features import pair_features


def candidate_features(candidates, queries, items, numeric, counts, weights,
                       query_vectors=None, item_vectors=None, max_candidates=400, centers=None):
    pool = rrf(candidates, weights, constant=30, top_k=max_candidates)
    # Схема фиксируется по названиям, а не по случайно встретившимся в batch каналам.
    keys = pool[['query_id','pos']]
    selected = candidates.merge(keys,on=['query_id','pos'],how='inner')
    rank = selected.pivot_table(index=['query_id','pos'],columns='source',values='rank',aggfunc='min',observed=True)
    rank = (1/(30+rank)).fillna(0).add_prefix('rank_')
    score = selected.pivot_table(index=['query_id','pos'],columns='source',values='score',aggfunc='max',observed=True).fillna(0).add_prefix('score_')
    source_features = rank.join(score).reset_index()
    source_features['n_sources'] = (rank>0).sum(axis=1).to_numpy()
    if centers is None:
        centers = items.groupby('item_location_id')[['item_latitude','item_longitude']].median()
    # Arrow chunked strings при многократном take могут копировать весь буфер.
    # Оставляем только нужные поля и один раз материализуем строки для быстрых gather.
    pair_items = items[['item_id','title_text','params_text','description_text',
                        'item_location_id','item_category_id','item_rating','item_latitude','item_longitude']].copy()
    for col in ['item_id','title_text','params_text','description_text']:
        pair_items[col] = pair_items[col].astype(object)
    query_map = queries.set_index('query_id')
    frames=[]
    for query_id,group in pool.groupby('query_id',sort=False):
        q=query_map.loc[query_id]
        positions=group.pos.to_numpy(dtype=int)
        features=pair_features(q,positions,pair_items,numeric,counts,centers)
        features['query_id']=query_id
        features['pos']=positions
        features['rrf']=group.rrf.to_numpy()
        if query_vectors is not None and item_vectors is not None:
            features['dense_cosine'] = np.asarray(item_vectors[positions]) @ query_vectors[query_id]
        frames.append(features)
    result=pd.concat(frames,ignore_index=True).merge(source_features,on=['query_id','pos'],how='left')
    return result


def feature_columns(frame):
    return [c for c in frame if c not in ['query_id','pos','target','regime','split','item_id']]


def model_matrix(frame,columns):
    # Источник может не вернуть ни одного кандидата для batch: missing rank/score = 0.
    result=frame.reindex(columns=columns).copy()
    for col in columns:
        if col.startswith(('rank_','score_')) or col=='n_sources': result[col]=result[col].fillna(0)
    return result.fillna(-1).astype('float32')
