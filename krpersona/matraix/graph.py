"""Reading the MatrAIx graph (persona/synthesis/graph/full_dag.json)."""
import json
from functools import lru_cache
from krpersona import paths


@lru_cache(maxsize=1)
def _values():
    return {nd['id']: nd['values'] for nd in json.load(open(paths.graph_path()))['nodes']}


def graph_values():
    """attribute id -> list of MatrAIx category labels (a fresh copy each call)"""
    return {k: list(v) for k, v in _values().items()}
