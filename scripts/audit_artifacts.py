"""Проверяет связь разбиений, кандидатов и индексов после прогона."""
import json
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]
cache=root/'outputs/cache'
pairs=pd.read_parquet(cache/'split_pairs.parquet',columns=['context_key','text_key','item_id','split','regime','query_id'])
eval_q=pd.read_parquet(cache/'eval_queries.parquet')
base=pairs[pairs.split.eq('base')]
held=pairs[pairs.split.ne('base')]
assert not set(base.context_key)&set(held.context_key)
assert not set(base.text_key)&set(held.loc[held.regime.eq('cold'),'text_key'])
assert eval_q.query_id.is_unique
truth=json.loads((cache/'truth.json').read_text())
expected=held[held.query_id.isin(eval_q.query_id)].groupby('query_id').item_id.agg(lambda s:sorted(set(s))).to_dict()
assert truth==expected
items=pd.read_parquet(cache/'items.parquet',columns=['item_id'])
assert items.item_id.is_unique
allowed=set(items.item_id)
assert all(set(v)<=allowed for v in truth.values())
for a,b in [('rank_train','dev'),('rank_train','test'),('dev','test')]:
    assert not set(eval_q.loc[eval_q.split.eq(a),'text_key'])&set(eval_q.loc[eval_q.split.eq(b),'text_key'])
summary={}
for path in sorted(cache.glob('candidates_*.parquet')):
    c=pd.read_parquet(path)
    assert set(c.query_id)<=set(eval_q.query_id),path.name
    assert c.pos.between(0,len(items)-1).all(),path.name
    assert not c.duplicated(['query_id','source','pos']).any(),path.name
    assert c['rank'].gt(0).all(),path.name
    summary[path.name]={'rows':len(c),'queries':c.query_id.nunique(),'sources':c.source.nunique()}
(root/'outputs/validation/artifact_audit.json').write_text(json.dumps({'valid':True,'queries':len(eval_q),'items':len(items),'candidate_files':summary},indent=2))
print('Split, truth and candidate artifact invariants: OK')
