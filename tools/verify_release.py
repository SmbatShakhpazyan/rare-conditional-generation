"""Verify published file bytes without executing research or upstream code."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'RELEASE_MANIFEST.json').read_text())
errors=[]
for name,entry in m['files'].items():
 p=root/name
 if not p.is_file() or p.is_symlink():errors.append(name+':missing_or_symlink');continue
 if p.stat().st_size!=entry['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:errors.append(name+':changed')
print(json.dumps({'status':'PASSED' if not errors else 'FAILED','verified_files':len(m['files']),'errors':errors}))
raise SystemExit(bool(errors))
