"""Статическая проверка синтаксиса и изоляции исследовательских namespace."""
import ast
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
prefixes=['eda_','fe_','bl_','hy_','de_','fu_','su_']
results=[]
for path,prefix in zip(sorted((root/'notebooks').glob('*.ipynb')),prefixes):
    nb=json.loads(path.read_text())
    violations=[]
    for number,cell in enumerate(nb['cells']):
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        tree=ast.parse(source)
        for node in tree.body:
            targets=[]
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):targets=[node.name]
            if isinstance(node,ast.Assign):
                targets=[n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)]
            if isinstance(node,(ast.For,ast.With)):
                targets=[n.id for n in ast.walk(node.target) if isinstance(n,ast.Name)] if isinstance(node,ast.For) and isinstance(node.target,ast.Name) else []
            violations.extend((number,name) for name in targets if not name.startswith(prefix))
    assert not violations,(path.name,violations)
    results.append({'notebook':path.name,'syntax_valid':True,'prefix':prefix,'top_level_names_isolated':True})
(root/'outputs/validation/notebook_checks.json').write_text(json.dumps(results,indent=2))
print('7 notebooks: valid syntax, separate research namespaces')
