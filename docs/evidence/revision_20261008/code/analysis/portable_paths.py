from pathlib import Path
import os
BUNDLE_ROOT=Path(__file__).resolve().parents[2]
DATA_ROOT=Path(os.environ.get('HPID_DATA_ROOT',BUNDLE_ROOT/'inputs/runtime')).resolve()
SOURCE_ROOT=BUNDLE_ROOT/'sources/frozen_be54300'
PROJECT_ROOT=BUNDLE_ROOT/'sources/project'
OUTPUT_DIR=Path(os.environ.get('HPID_RESULTS_DIR',BUNDLE_ROOT/'replay_results')).resolve()
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('HPID_HISTORICAL_GROUP_SOURCE',str(BUNDLE_ROOT/'sources/historical_da7d236/physical_groups.py'))
def resolve_data(value):
    if isinstance(value,dict):return {k:resolve_data(v) for k,v in value.items()}
    if isinstance(value,list):return [resolve_data(v) for v in value]
    if not isinstance(value,str):return value
    replacements={'${RUNTIME_ROOT}':DATA_ROOT,'${PROJECT_ROOT}':PROJECT_ROOT/'hpid_split','${REVISION_ROOT}':BUNDLE_ROOT,'${WORKSPACE_ROOT}':PROJECT_ROOT,'${BUNDLE_ROOT}':BUNDLE_ROOT}
    for marker,path in replacements.items():
        if marker in value:value=value.replace(marker,str(path)).replace('\\','/')
    return value
