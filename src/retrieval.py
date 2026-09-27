"""Общие индексы и операции над кандидатами для 02–06."""
from collections import defaultdict
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def top_indices(scores, k=100, allowed=None):
    scores = np.asarray(scores).ravel()
    positions = np.arange(len(scores)) if allowed is None else np.asarray(allowed)
    if not len(positions):
        return np.empty(0, dtype=np.int32)
    values = scores[positions]
    valid = np.isfinite(values) & (values > 0)
    positions, values = positions[valid], values[valid]
    if len(positions) > k:
        # Все ties на границе проходят во второй этап: результат не зависит от argpartition.
        threshold = np.partition(values, -k)[-k]
        keep = values >= threshold
        positions, values = positions[keep], values[keep]
    order = np.lexsort((positions, -values))[:k]
    return positions[order].astype(np.int32)


def sparse_top(row, k=100, allowed_mask=None):
    row = row.tocsr()
    idx, values = row.indices, row.data
    keep = values > 0
    if allowed_mask is not None:
        keep &= allowed_mask[idx]
    idx, values = idx[keep], values[keep]
    if len(idx) > k:
        threshold = np.partition(values, -k)[-k]
        keep = values >= threshold
        idx, values = idx[keep], values[keep]
    order = np.lexsort((idx, -values))[:k]
    return idx[order], values[order]


class LexicalIndex:
    def __init__(self):
        self.word = CountVectorizer(min_df=2, max_features=160000, ngram_range=(1,2), dtype=np.float32)
        self.title = CountVectorizer(min_df=2, max_features=100000, ngram_range=(1,1), dtype=np.float32)
        self.char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=3,
                                    max_features=180000, sublinear_tf=True, dtype=np.float32)

    @staticmethod
    def bm25_matrix(counts, k1=1.2, b=.75):
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log1p((counts.shape[0] - df + .5) / (df + .5))
        result = counts.copy().tocsr()
        norm = k1 * (1-b + b*lengths/max(float(lengths.mean()), 1))
        result.data = result.data * (k1+1) / (result.data + np.repeat(norm, np.diff(result.indptr)))
        return result.multiply(idf.astype(np.float32)).tocsr()

    def fit(self, items):
        counts = self.word.fit_transform(items.lexical_text).tocsr()
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.word_matrix = self.tfidf.fit_transform(counts).astype(np.float32)
        self.bm25 = self.bm25_matrix(counts)
        del counts
        title_counts = self.title.fit_transform(items.title_text).tocsr()
        self.overlap = title_counts.copy()
        self.overlap.data[:] = 1
        self.title_bm25 = self.bm25_matrix(title_counts)
        self.char_matrix = self.char.fit_transform(items.title_text).tocsr()
        return self

    def search(self, queries, items, k=120, methods=None, progress=True):
        methods = methods or ['overlap','word','char','bm25','bm25_title']
        qword = self.word.transform(queries.query_text)
        qtitle = self.title.transform(queries.text_key)
        qtitle.data[:] = 1
        qword_binary = qword.copy(); qword_binary.data[:] = 1
        pairs = {
            'overlap': (qtitle, self.overlap),
            'word': (self.tfidf.transform(qword), self.word_matrix),
            'char': (self.char.transform(queries.text_key), self.char_matrix),
            'bm25': (qword_binary, self.bm25),
            'bm25_title': (qtitle, self.title_bm25),
        }
        locations = items.item_location_id.to_numpy()
        rows = []
        for method in methods:
            qm, im = pairs[method]
            start = time.monotonic()
            for batch in range(0, len(queries), 32):
                scores = (qm[batch:batch+32] @ im.T).tocsr()
                for offset in range(scores.shape[0]):
                    q = queries.iloc[batch+offset]
                    row = scores.getrow(offset)
                    for suffix, mask in [('', None), ('_local', locations == q.search_location_id)]:
                        pos, values = sparse_top(row, k, mask)
                        rows.extend((q.query_id, int(p), method+suffix, rank+1, float(s))
                                    for rank, (p,s) in enumerate(zip(pos,values)))
            if progress: print(f'{method}: {len(queries)} queries, {time.monotonic()-start:.1f}s', flush=True)
        return compact_candidates(pd.DataFrame(rows, columns=['query_id','pos','source','rank','score']))


class HistoryIndex:
    def fit(self, pairs, items, label_catalog=None):
        self.items = items[['item_id','item_location_id','item_category_id','item_microcat_id']].copy()
        self.position = dict(zip(items.item_id, items.index))
        self.pop = pairs.item_id.value_counts().to_dict()
        counts = np.array([self.pop.get(i,0) for i in items.item_id], dtype=np.float32)
        self.pop_scores = counts
        self.context = self._hist(pairs, 'context_key')
        self.text = self._hist(pairs, 'text_key')
        unique = pairs[pairs.text_key.isin(self.text)].drop_duplicates('text_key').sort_values('text_key')
        self.hist_texts = unique.text_key.tolist()
        self.query_vectorizer = TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1,max_features=100000,dtype=np.float32)
        self.query_matrix = self.query_vectorizer.fit_transform(self.hist_texts)
        meta = (items if label_catalog is None else label_catalog).set_index('item_id').item_microcat_id
        train = pairs[pairs.item_id.isin(meta.index)].copy()
        # Naive Bayes предсказывает тип услуги, а не 500 тысяч классов item_id.
        self.nb_vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=80000,dtype=np.float32)
        x = self.nb_vectorizer.fit_transform(train.query_text)
        self.nb = MultinomialNB(alpha=.2).fit(x, train.item_id.map(meta).astype(str))
        self.microcat_positions = {str(cat): np.asarray(idx,dtype=np.int32)
                                   for cat,idx in items.groupby('item_microcat_id').groups.items()}
        return self

    def _hist(self, pairs, key):
        counts = pairs.groupby([key,'item_id']).size().rename('count').reset_index()
        out = defaultdict(list)
        for row in counts.itertuples(index=False):
            p = self.position.get(row.item_id)
            if p is not None: out[getattr(row,key)].append((p, float(row.count)))
        return {key: sorted(values,key=lambda x:(-x[1],x[0])) for key,values in out.items()}

    def search(self, queries, k=120):
        rows=[]
        locations=self.items.item_location_id.to_numpy()
        categories=self.items.item_category_id.to_numpy()
        near = (self.query_vectorizer.transform(queries.text_key) @ self.query_matrix.T).tocsr()
        nb_probs = self.nb.predict_proba(self.nb_vectorizer.transform(queries.query_text))
        for j,q in enumerate(queries.itertuples(index=False)):
            local = np.flatnonzero(locations==q.search_location_id)
            category = np.flatnonzero(categories==q.search_category)
            channels = {
                'popularity': [(int(p),self.pop_scores[p]) for p in top_indices(self.pop_scores,k)],
                'popularity_local': [(int(p),self.pop_scores[p]) for p in top_indices(self.pop_scores,k,local)],
                'popularity_category': [(int(p),self.pop_scores[p]) for p in top_indices(self.pop_scores,k,category)],
                'history': self.text.get(q.text_key,[])[:k],
                'history_context': self.context.get(q.context_key,[])[:k],
                'history_local': [(p,s) for p,s in self.text.get(q.text_key,[]) if locations[p]==q.search_location_id][:k],
            }
            npos, nsim = sparse_top(near.getrow(j), 8)
            similar=defaultdict(float)
            for p,s in zip(npos,nsim):
                # Exact history — отдельный канал; не удваиваем её этим источником.
                if self.hist_texts[p] == q.text_key or s < .2: continue
                for item,count in self.text.get(self.hist_texts[p],[])[:200]:
                    similar[item] += float(s)**3 * np.log1p(count)
            channels['similar_history'] = sorted(similar.items(),key=lambda x:(-x[1],x[0]))[:k]
            channels['similar_history_local'] = sorted(((p,s) for p,s in similar.items() if locations[p]==q.search_location_id),key=lambda x:(-x[1],x[0]))[:k]
            cats = np.argsort(-nb_probs[j])[:3]
            nb_scores={}
            for catpos in cats:
                pos=self.microcat_positions.get(str(self.nb.classes_[catpos]),np.array([],dtype=int))
                for p in top_indices(self.pop_scores+1,k,pos):
                    nb_scores[int(p)] = float(nb_probs[j,catpos]) * float(np.log1p(self.pop_scores[p])+1)
            channels['naive_bayes'] = sorted(nb_scores.items(),key=lambda x:(-x[1],x[0]))[:k]
            channels['naive_bayes_local'] = []
            for catpos in cats:
                pos=self.microcat_positions.get(str(self.nb.classes_[catpos]),np.array([],dtype=int))
                pos=pos[locations[pos]==q.search_location_id]
                for p in top_indices(self.pop_scores+1,k,pos):
                    channels['naive_bayes_local'].append((int(p),float(nb_probs[j,catpos])*(np.log1p(self.pop_scores[p])+1)))
            channels['naive_bayes_local'].sort(key=lambda x:(-x[1],x[0]))
            for source,values in channels.items():
                rows.extend((q.query_id,p,source,rank+1,float(s)) for rank,(p,s) in enumerate(values[:k]))
        return compact_candidates(pd.DataFrame(rows,columns=['query_id','pos','source','rank','score']))


def rrf(candidates, weights=None, constant=30, top_k=100):
    frame = candidates.copy()
    weights = weights if weights is not None else {s:1. for s in frame.source.unique()}
    frame['rrf'] = frame.source.map(weights).fillna(0) / (constant+frame['rank'])
    frame = frame[frame.rrf.gt(0)].groupby(['query_id','pos'],as_index=False).rrf.sum()
    return frame.sort_values(['query_id','rrf','pos'],ascending=[True,False,True]).groupby('query_id',sort=False).head(top_k)


def predictions_from_frame(frame, items, score=None, k=100):
    if score is not None:
        frame=frame.sort_values(['query_id',score,'pos'],ascending=[True,False,True])
    frame=frame.drop_duplicates(['query_id','pos']).groupby('query_id',sort=False).head(k)
    ids=items.item_id.to_numpy()
    frame=frame.assign(item_id=ids[frame.pos.to_numpy(dtype=int)])
    return frame.groupby('query_id',sort=False).item_id.agg(list).to_dict()


def dense_search(query_vectors, item_vectors, queries, items, k=120, batch_size=16):
    rows=[]
    locations=items.item_location_id.to_numpy()
    for start in range(0,len(queries),batch_size):
        scores=np.asarray(query_vectors[start:start+batch_size],dtype=np.float32) @ np.asarray(item_vectors,dtype=np.float32).T
        for offset, values in enumerate(scores):
            q=queries.iloc[start+offset]
            for source,allowed in [('dense',None),('dense_local',np.flatnonzero(locations==q.search_location_id))]:
                positions=top_indices(values,k,allowed)
                rows.extend((q.query_id,int(p),source,rank+1,float(values[p])) for rank,p in enumerate(positions))
    return compact_candidates(pd.DataFrame(rows,columns=['query_id','pos','source','rank','score']))


def compact_candidates(frame):
    return frame.astype({'query_id':'string','pos':'int32','source':'category','rank':'int16','score':'float32'})
