from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gate_ablation import inputs,load_previous_package,pg,Image,np,read_json,write_csv,write_json,SNAPSHOT
from gate_ablation import _load_candidates
import hashlib
import importlib.util
spec=importlib.util.spec_from_file_location('hpid_split.physical_groups_aug20',Path(__file__).parent/'physical_groups_da7d236.py')
historic=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=historic
spec.loader.exec_module(historic)

def worker(item):
    p=Path(item['package'])
    inst,records=load_previous_package(p)
    cand=_load_candidates(p)
    image=Image.open(p/'source.png').convert('RGB')
    code=historic if item['split']=='test226' else pg
    res=code.build_physical_groups(inst,records,candidates=cand,image=image)
    stored=np.asarray(Image.open(p/'group_id_map.tiff'))
    old=read_json(p/'groups.json')
    return {'split':item['split'],'case_id':item['case_id'],'exact_map':bool(np.array_equal(stored,res.group_map)),
            'changed_pixels':int(np.count_nonzero(stored!=res.group_map)),
            'stored_groups':len(old),'rerun_groups':len(res.groups),
            'exact_semantics':[(g.group_index,g.group_id,g.semantic_name) for g in res.groups]==[(g['group_index'],g['group_id'],g['semantic_name']) for g in old],
            'stored_sha256':hashlib.sha256(stored.tobytes()).hexdigest(),
            'rerun_sha256':hashlib.sha256(res.group_map.tobytes()).hexdigest()}

if __name__=='__main__':
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(worker,i) for i in inputs()]
        for n,f in enumerate(as_completed(fs),1):
            r=f.result();rows.append(r)
            print(n,r['split'],r['case_id'],r['exact_map'],flush=True)
    rows.sort(key=lambda r:(r['split'],r['case_id']))
    write_csv(Path(__file__).parent/'gate_baseline_audit_historical.csv',rows)
    write_json(Path(__file__).parent/'gate_baseline_audit_historical.json',{
        'source':str(SNAPSHOT),'source_sha256':hashlib.sha256((SNAPSHOT/'src/hpid_split/physical_groups.py').read_bytes()).hexdigest(),
        'splits':{s:{'n':len([r for r in rows if r['split']==s]),'exact_maps':sum(r['exact_map'] for r in rows if r['split']==s),'exact_semantics':sum(r['exact_semantics'] for r in rows if r['split']==s)} for s in ('test226','holdout42')}})
