"""Probe counterfactual label ordering under specified Python worker hash seeds."""
from pathlib import Path
import sys,os,argparse
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gate_ablation import inputs,worker,write_json
import gate_ablation as gate
import numpy as np

def probe(item):
    module=gate.HISTORICAL_GROUP_MODULE if item['split']=='test226' else gate.CURRENT_GROUP_MODULE
    original=module.build_physical_groups
    maps=[]
    def wrapped(*args,**kwargs):
        r=original(*args,**kwargs);maps.append(r.group_map.copy());return r
    module.build_physical_groups=wrapped
    try:results=worker(item)
    finally:module.build_physical_groups=original
    np.savez_compressed(Path(__file__).parent/('gate_hashseed_'+os.environ['GATE_PROBE_TAG']+'_'+item['case_id']+'.npz'),**dict(zip(gate.VARIANTS,maps)))
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--tag',required=True)
    args=p.parse_args();os.environ['PYTHONHASHSEED']=str(args.seed);os.environ['GATE_PROBE_TAG']=args.tag
    selected=[i for i in inputs() if i['split']=='test226' and i['case_id'] in {'calculator__01','scarf__03'}]
    with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(probe,selected))
    write_json(Path(__file__).parent/('gate_hashseed_'+args.tag+'.json'),{'seed':args.seed,'rows':[r for res in results for r in res[0]]})
