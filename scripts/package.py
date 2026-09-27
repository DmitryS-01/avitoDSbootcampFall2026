"""Переносимый архив: исходники, результаты, данные и offline-веса; без временных индексов."""
import argparse
import json
import sys
import zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from src.data import data_dir, sha256_file
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=root.parent/'avito_candidate_generation.zip')
args=parser.parse_args()
excluded={'.git','.venv','.runtime','__pycache__','.cache'}
files=[]
for path in sorted(root.rglob('*')):
    if not path.is_file():continue
    rel=path.relative_to(root)
    if any(p in excluded for p in rel.parts) or rel.parts[:2]==('outputs','cache') or path.suffix in ['.zip','.pyc']:continue
    if rel.parts[0]=='dataset' and path.suffix=='.parquet':continue
    files.append((path,Path('avito_candidate_generation')/rel))
for name in ['train.parquet','benchmark_queries.parquet','benchmark_items.parquet']:
    files.append((data_dir()/name,Path('avito_candidate_generation/dataset')/name))
args.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as archive:
    for source,target in files:archive.write(source,str(target))
with zipfile.ZipFile(args.output) as archive:
    bad=archive.testzip()
    if bad:raise ValueError(f'Ошибка CRC: {bad}')
result={'path':str(args.output),'files':len(files),'bytes':args.output.stat().st_size,'sha256':sha256_file(args.output),'crc_valid':True,'datasets_included':True,'local_model_included':True,'regenerable_cache_included':False}
args.output.with_suffix('.manifest.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
