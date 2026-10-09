import hashlib
import json
import zipfile
from pathlib import Path

B=Path(__file__).resolve().parents[1]
files=json.loads((B/'FILE_MANIFEST.json').read_text())
actual={p.relative_to(B).as_posix() for p in B.rglob('*') if p.is_file() and '.git' not in p.parts}
declared={r['path'] for r in files}|{'FILE_MANIFEST.json'}
assert actual==declared,{'unlisted_files':sorted(actual-declared),'missing_files':sorted(declared-actual)}
for r in files:
    p=B/r['path'];assert p.is_file(),r['path'];assert len(p.read_bytes())==r['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['path']
with zipfile.ZipFile(B/'inputs/prediction_packages.zip') as z:
    members=json.loads((B/'inputs/archive_members.json').read_text());assert len(members)==len(z.namelist())
    for r in members:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256'],r['path']
    forbidden=[n for n in z.namelist() if Path(n).name in {'source.png','source_crop.png','truth_part.png','object_mask_crop.png'} or '/parts_crop/' in n or 'LOCAL_ONLY' in n]
    assert not forbidden,forbidden
assert not list(B.rglob('*LOCAL_ONLY*'))
print(json.dumps({'status':'PASS','files_checked':len(files),'archive_members_checked':len(members),'source_photos_and_reference_mask_pixels_excluded':True},indent=2))
