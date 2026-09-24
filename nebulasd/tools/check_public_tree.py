"""Check public documentation links, English-only text and directory READMEs."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SKIP = {'__pycache__', '.pytest_cache', 'build', '.git', '.venv'}
SUFFIXES = {'.py', '.sh', '.md', '.toml', '.json', '.txt', '.cpp', '.h', '.cu', '.cuh'}


def maintained(path):
    return not any(part in SKIP or part.endswith('.egg-info') for part in path.relative_to(ROOT).parts)


def main():
    issues = []
    files = [p for base in ('nebulasd', 'swiftLLM') for p in (ROOT / base).rglob('*')
             if p.is_file() and maintained(p)]
    files += [p for p in ROOT.iterdir() if p.is_file() and p.suffix == '.md']
    for p in files:
        if p.suffix not in SUFFIXES: continue
        text = p.read_text()
        if re.search(r'[\u3400-\u9fff]', text): issues.append(f'Non-English text: {p.relative_to(ROOT)}')
        if re.search(r'/(?:home|Users)/[A-Za-z0-9_.-]+/', text):
            issues.append(f'Hardcoded user-home path: {p.relative_to(ROOT)}')
        if p.suffix != '.md': continue
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if '://' in target or target.startswith(('#', 'mailto:')): continue
            target = target.split('#', 1)[0]
            if target and not (p.parent / target).exists():
                issues.append(f'Broken link: {p.relative_to(ROOT)} -> {target}')
    for p in (ROOT / 'nebulasd').rglob('*'):
        if p.is_dir() and maintained(p) and any(f.is_file() and f.suffix in SUFFIXES for f in p.iterdir()):
            if not (p / 'README.md').is_file(): issues.append(f'Missing README: {p.relative_to(ROOT)}')
    if issues: raise SystemExit('\n'.join(issues))
    print(f'PASS: {len(files)} public files; English-only text, local Markdown links and directory READMEs')


if __name__ == '__main__': main()
