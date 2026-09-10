import os, json
from pathlib import Path
os.chdir(r'c:\workspace\costimiser\training')
import cloudpickle
import numpy as np, pandas as pd
import sys
sys.path.insert(0, str(Path.cwd().parent))

nb = json.load(open('explore_differencing_experiments.ipynb', encoding='utf-8'))
def cell_src(marker):
    for c in nb['cells']:
        if c['cell_type']=='code' and marker in ''.join(c['source']):
            return ''.join(c['source'])
    raise RuntimeError(marker)
g={'__name__':'__main__'}
exec(cell_src('class TurnupFilter'), g)
exec(cell_src('class StateBuilder'), g)
exec(cell_src('Predict y = (state + beta'), g)
TurnupFilter=g['TurnupFilter']; load_state_and_prod=g['load_state_and_prod']

EWM_HALFLIFE='6h'
DEPLOY=Path('deploy'); DEPLOY.mkdir(exist_ok=True)
for f in DEPLOY.glob('*'): f.unlink()
print('cleared deploy/')

cloudpickle.dump(TurnupFilter(halflife=EWM_HALFLIFE), open(DEPLOY/'turnup_filter.pkl','wb'))

models_cfg = [
    ('experiments/differencing/Steam_power_corrected/260827160848', {}),
    ('experiments/differencing/MBS_SCT_CD/260827155414', {}),
]
manifest={'ewm_halflife':EWM_HALFLIFE,'required_columns':[],'models':[]}
req=set()
for path,kw in models_cfg:
    b,p = load_state_and_prod(Path(path), use_full_model=True,
                              scale=kw.get('scale',1.0),
                              multiplier=kw.get('multiplier'), divisor=kw.get('divisor'))
    bundle={'y_column':b.y_column,'state_col':b.state_col,'lag_days':b.lag_days,
            'ewm_halflife':b.ewm_halflife,'feat_list':b.feat_list,
            'state_builder':b,'production_model':p}
    pkl=DEPLOY/f'{b.y_column}_bundle.pkl'
    cloudpickle.dump(bundle, open(pkl,'wb'))
    req.update(b.feat_list); req.add(b.y_column)
    manifest['models'].append({'y_column':b.y_column,'state_col':b.state_col,
        'lag_days':b.lag_days,'ewm_halflife':b.ewm_halflife,'pkl':pkl.name})
    print('wrote', pkl.name)
manifest['required_columns']=sorted(req)
json.dump(manifest, open(DEPLOY/'manifest.json','w'), indent=2)

# round-trip: engine simulation, Wedge_Time index + reel-id selection
from utility import _feature_engineering, setpoint_df
from sklearn.metrics import r2_score
turnup = pd.read_parquet('../data/costimier_turnup.parquet')
turnup = _feature_engineering(turnup, setpoint_df, steam_null=False, clip=True)
turnup = turnup.set_index('Wedge_Time').sort_index()
turnup = turnup[turnup.grammage.isin([115,120,100,90,85,110])]
man=json.load(open(DEPLOY/'manifest.json'))
tf2=cloudpickle.load(open(DEPLOY/'turnup_filter.pkl','rb'))
filtered=tf2.filter_all(turnup, man['required_columns'])
for m in man['models']:
    bd=cloudpickle.load(open(DEPLOY/m['pkl'],'rb'))
    sb,pm=bd['state_builder'],bd['production_model']
    fm=sb.append_state(filtered.copy())
    ev=fm[fm[bd['state_col']].notna()]
    yhat=pm.predict(ev); ytrue=turnup[m['y_column']].reindex(ev.index).values.astype(float)
    ok=~np.isnan(yhat)&~np.isnan(ytrue)
    print(f"  round-trip {m['y_column']}: R2 vs raw = {r2_score(ytrue[ok],yhat[ok]):.4f} "
          f"(index={ev.index.name}, n={ok.sum()})")
print('models:', [m['y_column'] for m in man['models']], '| req cols:', len(man['required_columns']))
