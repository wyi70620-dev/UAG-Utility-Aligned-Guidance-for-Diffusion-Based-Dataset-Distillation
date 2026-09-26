"""Fetch the pinned upstream code; never downloads weights or runs experiments."""
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
versions = json.loads((root / 'vendor/versions.json').read_text())
for name, spec in versions.items():
    dest = root / 'vendor' / name
    if dest.exists():
        current = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
        if current != spec['commit']:
            raise RuntimeError('Existing upstream version differs; preserve it and choose a clean directory: ' + str(dest))
        print('Already present:', name)
        continue
    subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', spec['url'], str(dest)], check=True)
    subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', spec['commit']], check=True)
    print('Prepared:', name, spec['commit'])
