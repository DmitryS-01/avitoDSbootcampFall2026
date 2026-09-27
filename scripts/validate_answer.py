import argparse
import sys
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from src.validation import validate_submission
from src.data import data_dir
parser=argparse.ArgumentParser()
parser.add_argument('answer',nargs='?',default=str(root/'outputs/submissions/answer.csv'))
args=parser.parse_args()
queries=pd.read_parquet(data_dir()/'benchmark_queries.parquet',columns=['query_id'])
items=pd.read_parquet(data_dir()/'benchmark_items.parquet',columns=['item_id'])
print(validate_submission(args.answer,queries.query_id.tolist(),items.item_id.tolist()))
