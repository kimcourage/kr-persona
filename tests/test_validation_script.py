"""Check validation command inputs without fitting models or reading microdata."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('history', [False, True])
def test_validation_uses_current_samples_even_when_history_exists(tmp_path, history):
    artifacts = tmp_path / 'artifacts with spaces'
    samples = artifacts / 'samples'
    names = ['val_on', 'out_on']
    for name in names:
        target = samples / name
        target.mkdir(parents=True)
        (target / 'personas.parquet').touch()
        (target / 'internal.parquet').touch()
    if history:
        old = samples / 'history'
        old.mkdir()
        for name in ['before_fix', 'round1', 'round2', 'round3']:
            (old / f'personas_{name}.parquet').touch()
    # Keep the executable path free of spaces; artifact paths above exercise quoting.
    fake = tmp_path / 'fake_python'
    fake.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-c':
    print(os.environ['KRP_ARTIFACTS'])
else:
    with open(os.environ['CALLS'], 'a') as f:
        f.write(json.dumps(args, ensure_ascii=False) + '\\n')
    for arg in args:
        if '=' in arg and arg.endswith('.parquet'):
            assert Path(arg.split('=', 1)[1]).is_file(), arg
''')
    fake.chmod(0o755)
    calls = tmp_path / 'calls.jsonl'
    env = dict(os.environ, PY=str(fake), KRP_ARTIFACTS=str(artifacts), CALLS=str(calls))
    subprocess.run(['bash', str(ROOT / 'scripts' / '07_validate.sh')], env=env,
                   cwd=tmp_path, capture_output=True, text=True, check=True)
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    language = next(c for c in commands if 'krpersona.validate.language_check' in c)
    rules = [c for c in commands if 'krpersona.validate.rules_summary' in c]
    assert len(rules) == 4
    assert len([a for a in language if '=' in a]) == 2
    for command in rules:
        inputs = [a for a in command if '=' in a]
        assert len(inputs) == 2
        assert inputs[0].startswith('검증 표본(80% 모형)=')
        assert inputs[1].startswith('최종 표본(100% 모형)=')
