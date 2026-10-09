"""Extract generated outputs and optionally add separately supplied local inputs."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
B=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=B/'inputs/runtime');p.add_argument('--local-runtime',type=Path);p.add_argument('--extract',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args();root=a.data_root.resolve()
    if a.extract:
        with zipfile.ZipFile(B/'inputs/prediction_packages.zip') as z:
            for member in z.infolist():
                target=(root/member.filename).resolve()
                if not target.is_relative_to(root):raise ValueError('Unsafe archive member')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(member))
    required=json.loads((B/'inputs/omitted_input_requirements.json').read_text())
    if a.local_runtime:
        if root==a.local_runtime.resolve():raise ValueError('Replay root must differ from archival runtime')
        for r in required:
            src=a.local_runtime/r['path'];dst=root/r['path']
            if not src.is_file() or sha(src)!=r['sha256']:raise ValueError('Missing or hash-mismatched locally supplied input: '+r['path'])
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    missing=[r['path'] for r in required if not (root/r['path']).is_file()]
    bad=[r['path'] for r in required if (root/r['path']).is_file() and sha(root/r['path'])!=r['sha256']]
    generated=json.loads((B/'inputs/archive_members.json').read_text());genmissing=[r['path'] for r in generated if not (root/r['path']).is_file()];genbad=[r['path'] for r in generated if (root/r['path']).is_file() and sha(root/r['path'])!=r['sha256']]
    report={'generated_files':len(generated),'generated_missing':len(genmissing),'generated_hash_mismatches':genbad,'external_files':len(required),'external_missing':len(missing),'external_hash_mismatches':bad,'complete_image_replay_inputs':not(missing or bad or genmissing or genbad),'first_missing_external':missing[:5]}
    print(json.dumps(report,indent=2))
    if bad or genbad:raise SystemExit(2)
if __name__=='__main__':main()
