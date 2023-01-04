import os
import subprocess
from pathlib import Path

subprocess.run(['python', 'clean.py'])
subprocess.run(['python', 'generate.py'])

os.chdir('./src')

p = Path('.')
for f in p.glob('*.py'):
    rslt = subprocess.run(['python', f.name], capture_output=True, text=True)
    ac = 'AC' in rslt.stdout
    wa = 'WA' in rslt.stdout
    if ac and not wa:
        print(f.stem, ':', 'PASSED')
    elif not ac and wa:
        print(f.stem, ':', 'FAILED')
    else:
        raise RuntimeError
