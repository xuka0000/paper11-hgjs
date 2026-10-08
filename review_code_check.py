"""Read-only source integrity/syntax check. Does not run experiments."""
import ast, hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parent
m=json.loads((root/'review_manifest.json').read_text(encoding='utf-8'))
for f in m['files']:
 p=root/f['path'];b=p.read_bytes()
 assert hashlib.sha256(b).hexdigest()==f['public_sha256'],f['path']
 if p.suffix=='.py':ast.parse(b.decode('utf-8-sig'),filename=f['path'])
print('Verified',len(m['files']),'exported files; scientific results are not certified.')
