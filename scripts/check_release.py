"""Check files that will be published, without third-party dependencies."""
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from krpersona.generate.resources import BUNDLE_FILES, check_bundle
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'build', 'dist'}
PRIVATE_SUFFIXES = {'.parquet', '.pkl', '.pickle', '.npz', '.sav', '.dta', '.sas7bdat',
                    '.feather', '.pt', '.zip', '.gz', '.pem', '.key'}
LOCAL_PATH = re.compile(r'(?:/(?:Users|home)/[^/\s\"\'<>]+/|/(?:mnt/user-data|tmp/claude)/|[A-Za-z]:\\Users\\)')


def publication_files(root):
    """Use Git's publication list only when root itself is the repository root."""
    try:
        top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root,
                             capture_output=True, text=True, check=True).stdout.strip()
        if Path(top).resolve() == root.resolve():
            listed = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                                    cwd=root, capture_output=True, check=True).stdout
            return sorted({root / name.decode('utf-8') for name in listed.split(b'\0') if name})
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return sorted(p for p in root.rglob('*')
                  if not any(part in SKIP_DIRS for part in p.relative_to(root).parts)
                  and (p.is_file() or p.is_symlink()))


def text_content(path):
    if path.suffix == '.xlsx':
        with zipfile.ZipFile(path) as workbook:
            return '\n'.join(workbook.read(name).decode('utf-8') for name in workbook.namelist()
                             if name.endswith('.xml'))
    try:
        return path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        return ''


def check(root=ROOT):
    problems = []
    bundle = root / 'bundle'
    bundle_valid = False
    if bundle.exists():
        try:
            check_bundle(bundle)
            bundle_valid = True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f'bundle: 파일 구성 또는 무결성 오류 ({exc})')
    for path in publication_files(root):
        rel = path.relative_to(root)
        bundle_name = rel.as_posix().removeprefix('bundle/')
        is_bundle = rel.parts[0] == 'bundle' and bundle_name in BUNDLE_FILES
        if rel.parts[0] == 'bundle' and bundle_name not in BUNDLE_FILES | {'MANIFEST.json'}:
            problems.append(f'{rel}: 생성 묶음의 지정 목록에 없는 파일')
        if path.is_symlink():
            problems.append(f'{rel}: 외부 파일을 가리킬 수 있는 심볼릭 링크')
            continue
        if not path.exists():  # A tracked file deleted from the working tree.
            continue
        if ((path.suffix.lower() in PRIVATE_SUFFIXES and not (is_bundle and bundle_valid)) or rel.parts[0] in {'data', 'artifacts'}
                or rel.as_posix() == 'config/paths.toml'
                or (path.name.startswith('.env') and path.name != '.env.example')):
            problems.append(f'{rel}: 원자료·실행 산출물 또는 개인 설정')
        if path.name == '.DS_Store':
            problems.append(f'{rel}: 운영체제 메타데이터')
        if path.stat().st_size > 2_000_000 and not (is_bundle and bundle_valid):
            problems.append(f'{rel}: 2 MB 초과 파일 — 공개 대상인지 확인 필요')
        try:
            content = text_content(path)
        except (OSError, zipfile.BadZipFile) as exc:
            problems.append(f'{rel}: 파일을 읽을 수 없음 ({exc})')
            continue
        if LOCAL_PATH.search(content):
            problems.append(f'{rel}: 개인 작업 환경의 절대 경로')
        if path.suffix == '.md' and '〔갱신' in content:
            problems.append(f'{rel}: 미완성 문서 표시')
    return problems


if __name__ == '__main__':
    issues = check()
    if issues:
        print('\n'.join(issues))
        sys.exit(1)
    print('공개 파일 구성 및 무결성 점검 통과 (자료 공개 허용 여부는 별도 검토)')
