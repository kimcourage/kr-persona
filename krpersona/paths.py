"""Locations used by every module.

Directories are configured in config/paths.toml (template: config/paths.example.toml) or by the
environment variables KRP_DATA, KRP_ARTIFACTS, KRP_MATRAIX and KRP_BUNDLE:
  data_dir       raw microdata, never committed (file names in config/data_files.toml); needed only to re-estimate or validate
  artifacts_dir  everything the pipeline writes (models, synthetic census base, masks, samples, results); default <repo>/artifacts
  matraix_dir    MatrAIx checkout (commit 3633d8d); always needed
  bundle_dir     generation inputs shipped with the repository (models, masks, synthetic census base, census aggregates); default <repo>/bundle
Generation reads its inputs with inp(): the file under artifacts_dir if it exists, otherwise the one in bundle_dir.
So a user without the raw data can generate personas from the bundle alone.
"""
import os, sys, tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / 'mapping' / '변수대응표.xlsx'          # attribute mapping workbook (step 1), tracked in git
MATRAIX_COMMIT = '3633d8d'


def _config():
    p = Path(os.environ.get('KRP_CONFIG', ROOT / 'config' / 'paths.toml'))
    return tomllib.load(open(p, 'rb')) if p.exists() else {}


_C = _config()
_FILES = tomllib.load(open(ROOT / 'config' / 'data_files.toml', 'rb'))


def _dir(key, env):
    v = os.environ.get(env) or _C.get(key)
    if not v:
        raise RuntimeError(f'{key} is not set: copy config/paths.example.toml to config/paths.toml or set {env}')
    return Path(v).expanduser()


def data(*parts):
    """path under the raw-data directory"""
    return _dir('data_dir', 'KRP_DATA').joinpath(*parts)


def raw(source, key):
    """raw file of a source as listed in config/data_files.toml, e.g. raw('census2020', 'person')"""
    return data(_FILES[source][key])


def _opt_dir(key, env, default):
    v = os.environ.get(env) or _C.get(key)
    return Path(v).expanduser() if v else default


def art(*parts, mkdir=False):
    """path under the artifacts directory; mkdir=True creates the parent directory (for files to be written)"""
    p = _opt_dir('artifacts_dir', 'KRP_ARTIFACTS', ROOT / 'artifacts').joinpath(*parts)
    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def bundle(*parts):
    """path under the bundle directory (generation inputs shipped with the repository)"""
    return _opt_dir('bundle_dir', 'KRP_BUNDLE', ROOT / 'bundle').joinpath(*parts)


def inp(*parts):
    """generation input: artifacts_dir/<parts> if that file exists (maintainer with re-estimated models), else bundle_dir/<parts>"""
    p = art(*parts)
    return p if p.is_file() else bundle(*parts)


def matraix(*parts):
    return _dir('matraix_dir', 'KRP_MATRAIX').joinpath(*parts)


def use_matraix():
    """put the MatrAIx checkout on sys.path (for `import persona...`)"""
    p = str(matraix())
    if p not in sys.path:
        sys.path.insert(0, p)


def graph_path():
    return matraix('persona', 'synthesis', 'graph', 'full_dag.json')


def dims_path():
    return matraix('persona', 'schema', 'dimensions.json')
