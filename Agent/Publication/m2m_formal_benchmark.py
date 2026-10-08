"""Declared24-component scale adapter, preserving the sealed development generator.

The source guard is the only changed AST node. All geometry, quota, resource,
time construction and full catalogue functions retain their frozen bytecode.
Formal access is controlled by m2m_formal_campaign; this module also serves the
one separately declared development-scale physical check.
"""
import ast
import hashlib
import inspect
from pathlib import Path
from types import FunctionType
from Agent.Publication import component_benchmark as component
from Agent.Publication import foundation_benchmark as foundation

SOURCE_SHA='b89023ac34976767cca6bc486f020123d5c6b92c674a74200bc4258edfa0a94b'


def _scenario():
    if hashlib.sha256(Path(component.__file__).read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError('Frozen geometry source changed; scale adapter requires review')
    tree=ast.parse(inspect.getsource(component.scenario));guard=tree.body[0].body[0]
    compare=guard.test.values[0]
    if (not isinstance(guard,ast.If) or not isinstance(compare,ast.Compare)
        or not isinstance(compare.left,ast.Name) or compare.left.id!='components'
        or not isinstance(compare.ops[0],ast.NotIn) or ast.literal_eval(compare.comparators[0])!=(6,12)):
        raise ValueError('Unexpected source scale guard')
    compare.comparators[0].elts.append(ast.Constant(value=24))
    namespace=dict(vars(component));exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace)
    return namespace['scenario']


_namespace=dict(vars(foundation));_namespace['original_scenario']=_scenario()
for _name in ['instance_model','domain_for','quota_tour','prepare_base']:
    _original=getattr(foundation,_name)
    _copy=FunctionType(_original.__code__,_namespace,_original.__name__,_original.__defaults__,_original.__closure__)
    _copy.__kwdefaults__=_original.__kwdefaults__;_namespace[_name]=_copy
instance_model=_namespace['instance_model']
prepare_base=_namespace['prepare_base']
