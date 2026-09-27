import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from src.retrieval import LexicalIndex, sparse_top, top_indices, rrf
from src.validation import recall_table, validate_submission
from src.data import normalize, prepare_queries


class Contracts(unittest.TestCase):
    def test_macro_recall_counts_missing_queries(self):
        truth={'a':['x','y'],'b':['z'],'c':['v']}
        result=recall_table({'a':['x'],'b':['z']},truth,ks=(50,))
        self.assertAlmostEqual(result['recall@50'].mean(),.5)
        with self.assertRaises(ValueError): recall_table({'a':['x','x']},{'a':['x']})

    def test_deterministic_ties_and_local_top(self):
        scores=np.array([1.,3.,3.,0.,2.])
        self.assertEqual(top_indices(scores,2).tolist(),[1,2])
        self.assertEqual(top_indices(scores,2,[0,3,4]).tolist(),[4,0])
        idx,val=sparse_top(sparse.csr_matrix(scores.reshape(1,-1)),2,np.array([1,0,0,1,1],bool))
        self.assertEqual(idx.tolist(),[4,0])
        self.assertEqual(top_indices(np.zeros(4)).tolist(),[])

    def test_bm25_matches_hand_formula(self):
        counts=sparse.csr_matrix([[2.,0.],[1.,1.],[0.,3.]],dtype=np.float32)
        got=LexicalIndex.bm25_matrix(counts).toarray()
        lengths=np.array([2.,2.,3.]); avg=lengths.mean()
        expected=2*2.2/(2+1.2*(.25+.75*lengths[0]/avg))*np.log1p((3-2+.5)/(2+.5))
        self.assertAlmostEqual(float(got[0,0]),float(expected),places=6)
        self.assertEqual(got[0,1],0)

    def test_rrf_deduplicates_items(self):
        c=pd.DataFrame([['q',0,'a',1,1.],['q',0,'b',2,1.],['q',1,'a',2,.5]],columns=['query_id','pos','source','rank','score'])
        result=rrf(c,{'a':1,'b':1},top_k=2)
        self.assertEqual(result.pos.tolist(),[0,1])
        self.assertEqual(len(result),2)

    def test_normalization_and_context(self):
        self.assertEqual(normalize('  Ёлка\nРЕМОНТ  12 '),'елка ремонт 12')
        q=pd.DataFrame({'search_query':['Ёлка','елка'],'search_location_id':[1,2],
                        'search_is_delivery_search':[0,0],'search_infm_params_text':['',''],'search_category':[114,114]})
        p=prepare_queries(q)
        self.assertEqual(p.text_key.nunique(),1)
        self.assertEqual(p.context_key.nunique(),2)

    def test_price_sentinel_is_not_free(self):
        from src.features import item_features
        frame=pd.DataFrame({'item_price':[-1.,0.,100.], 'item_rating_reviews_count':[0.,1.,10.],
            'item_rating':[float('nan'),5.,4.], 'title_text':['a']*3, 'params_text':['']*3,
            'description_text':['']*3, 'item_is_phone_hidden':[False]*3,
            'item_is_message_forbidden':[False]*3, 'item_microcat_id':[1]*3,
            'item_latitude':[55.]*3, 'item_longitude':[37.]*3})
        features=item_features(frame)
        self.assertEqual(features.item_price_unknown.tolist(),[1.,0.,0.])
        self.assertTrue(pd.isna(features.loc[0,'item_price_log1p']))
        self.assertEqual(features.loc[1,'item_price_log1p'],0.)
        self.assertTrue(pd.isna(features.loc[0,'price_vs_microcat_log']))

    def test_inference_absent_channel_has_zero_rank(self):
        from src.fusion import model_matrix
        frame=pd.DataFrame({'rating':[float('nan')],'rank_word':[.1]})
        result=model_matrix(frame,['rating','rank_word','rank_dense'])
        self.assertEqual(result.loc[0,'rating'],-1.)
        self.assertEqual(result.loc[0,'rank_dense'],0.)

    def test_csv_contract_and_leading_zero(self):
        q=['00WuFMaXSFZBxSzT']; ids=['0123456789abcdef','1111111111111111']
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'answer.csv'
            p.write_text('query_id,answer\n'+q[0]+','+ids[0]+'\n',encoding='utf-8')
            self.assertTrue(validate_submission(p,q,ids)['valid'])
            for bad in [ids[0]+' '+ids[0],ids[0].upper(),ids[0]+'  '+ids[1],'not_an_item_id___']:
                p.write_text('query_id,answer\n'+q[0]+','+bad+'\n',encoding='utf-8')
                with self.assertRaises(ValueError): validate_submission(p,q,ids)
            p.write_text('query_id,answer\n',encoding='utf-8')
            with self.assertRaises(ValueError): validate_submission(p,q,ids)
            p.write_text('query_id,answer\n'+q[0]+',\n',encoding='utf-8')
            self.assertEqual(validate_submission(p,q,ids)['min_candidates'],0)

if __name__=='__main__': unittest.main()
