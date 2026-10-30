"""Publication checks for a folder and a new Git repository."""
import importlib.util
from pathlib import Path
import subprocess

SPEC = importlib.util.spec_from_file_location('check_release', Path(__file__).parents[1] / 'scripts' / 'check_release.py')
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_clean_folder(tmp_path):
    (tmp_path / 'README.md').write_text('# Example\n')
    assert release.check(tmp_path) == []


def test_private_files_in_folder(tmp_path):
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'paths.toml').write_text('')
    (tmp_path / 'private sample.parquet').write_bytes(b'PAR1')
    (tmp_path / '.DS_Store').write_bytes(b'')
    assert len(release.check(tmp_path)) == 3


def test_local_path_and_placeholder(tmp_path):
    local = '/' + '/'.join(['Users', 'example', 'data'])
    (tmp_path / 'notes with spaces.md').write_text(local + '\n' + '〔' + '갱신〕')
    assert len(release.check(tmp_path)) == 2


def test_git_ignores_local_files_but_checks_tracked_ones(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / '.gitignore').write_text('*.parquet\n')
    sample = tmp_path / 'private sample.parquet'
    sample.write_bytes(b'PAR1')
    assert release.check(tmp_path) == []
    subprocess.run(['git', '-C', str(tmp_path), 'add', '-f', sample.name], check=True)
    assert len(release.check(tmp_path)) == 1


def test_nested_folder_does_not_scan_parent_repository(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    child = tmp_path / 'public'
    child.mkdir()
    (tmp_path / 'private.parquet').write_bytes(b'PAR1')
    (child / 'README.md').write_text('# Public\n')
    assert release.check(child) == []
