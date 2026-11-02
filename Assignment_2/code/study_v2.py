"""Reproducible nested chronological study; all paths supplied by caller."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json, time, platform, itertools, random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sklearn

FEATURES=['Temp','Salinity','Oxygen_Saturation','pH','DIC','Chlorophyll','NH4','NO2.NO3','PO4','SiO4']
NEURAL=['shallow_mlp','residual_mlp','ae_ridge','dae_ridge']
UNITS='10^6 cells mL^-1'

def dump(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(obj,indent=2,allow_nan=False)); temp.replace(path)

def load_data(path):
    df=pd.read_csv(path,sep='\t').sort_values('YearDay',kind='stable').reset_index(drop=True)
    assert len(df)==151 and np.isfinite(df[FEATURES+['Bacteria_abundance']].values).all()
    return df,df[FEATURES].to_numpy(float),df.Bacteria_abundance.to_numpy(float)/1e6

def make_splits(n,config):
    return [{'train':tr.tolist(),'gap':list(range(tr[-1]+1,va[0])),'validation':va.tolist()} for tr,va in TimeSeriesSplit(**config).split(np.arange(n))]

class ResidualMLP(nn.Module):
    def __init__(self,c):
        super().__init__(); w=c['width']; self.input=nn.Linear(10,w)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(w,w),nn.Tanh(),nn.Dropout(c['dropout']),nn.Linear(w,w)) for _ in range(c['blocks'])]); self.output=nn.Linear(w,1)
    def forward(self,x):
        h=torch.tanh(self.input(x))
        for block in self.blocks: h=h+0.5*block(h)
        return self.output(torch.tanh(h))

class Autoencoder(nn.Module):
    def __init__(self,c):
        super().__init__()
        def stack(dims):
            layers=[]
            for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
                layers.append(nn.Linear(a,b))
                if i<len(dims)-2: layers.append(nn.Tanh())
            return nn.Sequential(*layers)
        self.encoder=stack([10]+c['encoder_hidden']+[c['latent_dimension']])
        self.decoder=stack([c['latent_dimension']]+list(reversed(c['encoder_hidden']))+[10])
    def forward(self,x): return self.decoder(self.encoder(x))

def build_model(c):
    if c['family']=='shallow_mlp': return nn.Sequential(nn.Linear(10,c['width']),nn.Tanh(),nn.Dropout(c['dropout']),nn.Linear(c['width'],1))
    if c['family']=='residual_mlp': return ResidualMLP(c)
    return Autoencoder(c)

def setup(seed):
    torch.set_num_threads(1); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False

def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)
    elif str(device)=='mps': torch.mps.synchronize()

def metrics(y,p):
    return dict(rmse=float(np.sqrt(mean_squared_error(y,p))),mae=float(mean_absolute_error(y,p)),r2=float(r2_score(y,p)),negative_predictions=int(np.sum(np.asarray(p)<0)))

def neural_fit(X_train,y_train,X_valid,y_valid,candidate,seed,epochs,device,alphas,checkpoint_dir=None):
    setup(seed); sync(device); start=time.perf_counter(); c=candidate
    sx=StandardScaler().fit(X_train); sy=StandardScaler().fit(np.asarray(y_train).reshape(-1,1))
    xt=torch.tensor(sx.transform(X_train),dtype=torch.float32,device=device); xv=torch.tensor(sx.transform(X_valid),dtype=torch.float32,device=device)
    yt=torch.tensor(sy.transform(np.asarray(y_train).reshape(-1,1)),dtype=torch.float32,device=device)
    model=build_model(c).to(device); assert sum(p.numel() for p in model.parameters())==c['neural_parameter_count']
    opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay']); ae=c['family'] in ['ae_ridge','dae_ridge']
    rows=[]; curves=[]; predictions={}
    for epoch in range(1,max(epochs)+1):
        model.train(); opt.zero_grad(set_to_none=True)
        inp=xt+torch.randn_like(xt)*c.get('noise_std_standardized',0) if c.get('noise_std_standardized',0)>0 else xt
        loss=nn.functional.mse_loss(model(inp),xt if ae else yt)
        if not torch.isfinite(loss): raise FloatingPointError((c['id'],seed,epoch))
        loss.backward(); opt.step(); curves.append({'epoch':epoch,'train_loss':float(loss.detach().cpu())})
        if epoch not in epochs: continue
        model.eval()
        with torch.no_grad():
            a=model(xt).cpu().numpy(); b=model(xv).cpu().numpy()
            zt=model.encoder(xt).cpu().numpy() if ae else None; zv=model.encoder(xv).cpu().numpy() if ae else None
        checkpoint={'candidate':c,'seed':seed,'epoch':epoch,'model_state':{k:v.cpu() for k,v in model.state_dict().items()},'input_scaler':sx,'target_scaler':sy,'ridge_heads':{}}
        for alpha in (alphas if ae else [None]):
            trrec=varec=None
            if ae:
                sz=StandardScaler().fit(zt); ridge=Ridge(alpha=alpha).fit(sz.transform(zt),y_train)
                pt=ridge.predict(sz.transform(zt)); pv=ridge.predict(sz.transform(zv)); trrec=float(np.mean((a-sx.transform(X_train))**2)); varec=float(np.mean((b-sx.transform(X_valid))**2)); checkpoint['ridge_heads'][alpha]=(sz,ridge)
            else: pt=sy.inverse_transform(a).ravel(); pv=sy.inverse_transform(b).ravel()
            assert np.isfinite(pv).all()
            rows.append(dict(epoch=epoch,alpha=alpha,train_rmse=metrics(y_train,pt)['rmse'],rmse=metrics(y_valid,pv)['rmse'],train_reconstruction_mse=trrec,validation_reconstruction_mse=varec))
            predictions[f'{epoch}:{alpha}']=pv.tolist()
        if checkpoint_dir:
            path=Path(checkpoint_dir); path.mkdir(parents=True,exist_ok=True); torch.save(checkpoint,path/f'epoch_{epoch}.pt')
    sync(device)
    return dict(candidate=c,seed=seed,device=device,elapsed_seconds=time.perf_counter()-start,rows=rows,curves=curves,predictions=predictions)

def traditional_candidates(cfg):
    out=[dict(id='mean',family='mean'),dict(id='temperature_ols',family='temperature_ols')]
    out += [dict(id=f'ridge_{a}',family='ridge',alpha=a) for a in cfg['ridge_alphas']]
    out += [dict(id=f'pca_{k}_{a}',family='pca_ridge',components=k,alpha=a) for k in cfg['pca_components'] for a in cfg['ridge_alphas']]
    rf=cfg['random_forest']
    out += [dict(id=f'random_forest_{i:02d}',family='random_forest',n_estimators=rf['n_estimators'],max_depth=d,min_samples_leaf=l,max_features=f) for i,(d,l,f) in enumerate(itertools.product(rf['max_depth'],rf['min_samples_leaf'],rf['max_features']))]
    return out

def traditional_fit(X,y,V,v,c,seed):
    setup(seed); start=time.perf_counter(); fam=c['family']; trrec=varec=None; bundle={}
    if fam=='mean': pt=np.repeat(y.mean(),len(y)); pv=np.repeat(y.mean(),len(v)); bundle['mean']=y.mean()
    elif fam=='random_forest':
        model=RandomForestRegressor(**{k:c[k] for k in ['n_estimators','max_depth','min_samples_leaf','max_features']},random_state=seed,n_jobs=1).fit(X,y*1e6); pt=model.predict(X)/1e6; pv=model.predict(V)/1e6; bundle['model']=model
    else:
        sx=StandardScaler().fit(X); a=sx.transform(X); b=sx.transform(V); bundle['input_scaler']=sx
        if fam=='temperature_ols': a=X[:,[0]]; b=V[:,[0]]; model=LinearRegression()
        else:
            model=Ridge(alpha=c['alpha'])
            if fam=='pca_ridge':
                pca=PCA(n_components=c['components'],svd_solver='full').fit(a); za=pca.transform(a); zb=pca.transform(b)
                trrec=float(np.mean((a-pca.inverse_transform(za))**2)); varec=float(np.mean((b-pca.inverse_transform(zb))**2)); sz=StandardScaler().fit(za); a=sz.transform(za); b=sz.transform(zb); bundle.update(pca=pca,latent_scaler=sz)
        model.fit(a,y); pt=model.predict(a); pv=model.predict(b); bundle['model']=model
    return dict(candidate=c,seed=seed,device='cpu',elapsed_seconds=time.perf_counter()-start,rows=[dict(epoch=0,alpha=c.get('alpha'),train_rmse=metrics(y,pt)['rmse'],rmse=metrics(v,pv)['rmse'],train_reconstruction_mse=trrec,validation_reconstruction_mse=varec)],curves=[],predictions={f"0:{c.get('alpha')}":pv.tolist()}),bundle

def execute_job(job):
    path=Path(job['result_path'])
    if path.exists(): return json.loads(path.read_text())
    c=job['candidate']; args=[np.asarray(job[k]) for k in ['X','y','V','v']]
    if c['family'] in NEURAL: result=neural_fit(*args,c,job['seed'],job['epochs'],job['device'],job['alphas'],job['checkpoint_dir'])
    else:
        result,bundle=traditional_fit(*args,c,job['seed'])
        if job['checkpoint_dir']:
            p=Path(job['checkpoint_dir']); p.mkdir(parents=True,exist_ok=True); torch.save(bundle,p/'pipeline.pt')
    result.update(scope=job['scope'],inner_fold=job['inner_fold'],phase=job['phase']); dump(path,result); return result

def select(rows,candidates):
    df=pd.DataFrame(rows); df['alpha']=df.alpha.fillna(-1)
    scores=df.groupby(['family','candidate_id','epoch','alpha','inner_fold'],dropna=False).rmse.mean().groupby(level=[0,1,2,3]).mean().reset_index(name='selection_rmse')
    lookup={c['id']:c for c in candidates}; chosen={}
    def tie(r):
        c=lookup[r.candidate_id]
        return (c.get('neural_parameter_count',0),r.epoch,-c.get('weight_decay',0),c.get('latent_dimension',0),(-r.alpha if c['family'] in ['ae_ridge','dae_ridge'] else 0),c.get('max_depth') or 100000,-c.get('min_samples_leaf',0),c.get('max_features',0),c['id'])
    for fam,g in scores.groupby('family'):
        best=g[g.selection_rmse<=g.selection_rmse.min()+1e-12]; r=min(best.itertuples(),key=tie); c=dict(lookup[r.candidate_id]); c.update(epoch=int(r.epoch),alpha=None if r.alpha==-1 else float(r.alpha),selection_rmse=float(r.selection_rmse)); chosen[fam]=c
    return chosen,scores

def export_evaluation(output, predictions):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); df=pd.DataFrame(predictions); df.to_csv(output/'predictions.csv',index=False); records=[]
    for (scope,fam,seed),g in df.groupby(['scope','family','seed']): records.append(dict(scope=scope,family=fam,seed=seed,**metrics(g.observed,g.predicted)))
    for (scope,fam),g in df.groupby(['scope','family']):
        ens=g.groupby('row_index')[['observed','predicted']].mean(); records.append(dict(scope=scope,family=fam,seed='ensemble',**metrics(ens.observed,ens.predicted)))
    m=pd.DataFrame(records); m.to_csv(output/'metrics.csv',index=False); summary=[]
    for label,subset in [('mean_individual',m[m.seed!='ensemble']),('ensemble',m[m.seed=='ensemble'])]:
        means=subset.groupby(['family','scope'])[['rmse','mae','r2']].mean()
        for fam,g in means.groupby(level=0):
            for metric in ['rmse','mae','r2']: summary.append(dict(family=fam,metric=metric,mean=g[metric].mean(),fold_sd=g[metric].std(ddof=1) if len(g)>1 else 0.,aggregation=label))
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False)

def run_study(data_path,config_path,output_path,devices=('cuda:0','cuda:1')):
    root=Path(output_path); root.mkdir(parents=True,exist_ok=True); cfg=json.loads(Path(config_path).read_text()); df,X,y=load_data(data_path)
    candidates=cfg['candidates']+traditional_candidates(cfg); splits=make_splits(120,cfg['outer_cv']); scopes={f'outer_{i+1}':s for i,s in enumerate(splits)}; scopes['final']={'train':list(range(120)),'gap':[],'validation':list(range(120,151))}
    for s in scopes.values(): s['inner']=make_splits(len(s['train']),cfg['inner_cv'])
    dump(root/'splits.json',scopes); dump(root/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,devices=list(devices),cuda=torch.version.cuda,gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],dtype='float32',tf32=False,amp=False,deterministic=True))
    pools=[ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) for _ in devices]; all_pred=[]; all_rows=[]; selections={}; count={'neural':0,'random_forest':0}; started=time.time()
    def submit(jobs):
        futures=[pools[i%len(pools)].submit(execute_job,{**j,'device':devices[i%len(devices)]}) for i,j in enumerate(jobs)]
        results=[]
        for f in futures:
            r=f.result(); results.append(r)
            if r['candidate']['family'] in NEURAL: count['neural']+=1
            elif r['candidate']['family']=='random_forest': count['random_forest']+=1
            dump(root/'progress.json',dict(counts=count,elapsed_seconds=time.time()-started,last_scope=r['scope'],last_candidate=r['candidate']['id'],last_phase=r['phase']))
        return results
    def job(c,seed,scope,fold,tr,va,phase):
        name=f"{scope}/{phase}/{fold}/{c['id']}/seed_{seed}"
        return dict(candidate=c,seed=seed,scope=scope,inner_fold=fold,phase=phase,X=X[tr].tolist(),y=y[tr].tolist(),V=X[va].tolist(),v=y[va].tolist(),epochs=cfg['checkpoint_epochs'] if phase=='search' else [c['epoch']],alphas=cfg['ridge_alphas'] if phase=='search' else [c.get('alpha')],result_path=str(root/'search'/name/'result.json'),checkpoint_dir=str(root/'checkpoints'/name))
    try:
        for scope,s in scopes.items():
            tr=np.array(s['train']); jobs=[]
            for fold,split in enumerate(s['inner'],1):
                for c in candidates:
                    for seed in (cfg['selection_seeds'] if c['family'] in NEURAL+['random_forest'] else [0]): jobs.append(job(c,seed,scope,fold,tr[split['train']],tr[split['validation']],'search'))
            results=submit(jobs); rows=[]
            for r in results:
                for row in r['rows']: rows.append(dict(scope=scope,inner_fold=r['inner_fold'],family=r['candidate']['family'],candidate_id=r['candidate']['id'],seed=r['seed'],**row))
            chosen,scores=select(rows,candidates); selections[scope]=chosen; all_rows+=rows
            dest=root/('historical_holdout' if scope=='final' else 'formal'); dest.mkdir(exist_ok=True)
            pd.DataFrame([r for r in all_rows if (r['scope']=='final')==(scope=='final')]).to_csv(dest/'candidate_scores.csv',index=False)
            scores.insert(0,'scope',scope); scorepath=dest/'selection_scores.csv'
            if scope!='outer_1' and scope!='final' and scorepath.exists(): scores=pd.concat([pd.read_csv(scorepath),scores],ignore_index=True)
            scores.to_csv(scorepath,index=False); dump(dest/'selected_configurations.json',{k:v for k,v in selections.items() if (k=='final')==(scope=='final')})
            if scope=='final': dump(root/'final_configurations.json',dict(fixed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),configurations=chosen,historical_evaluated=False))
            jobs=[]
            for c in chosen.values():
                for seed in (cfg['evaluation_seeds'] if c['family'] in NEURAL+['random_forest'] else [0]): jobs.append(job(c,seed,scope,0,tr,np.array(s['validation']),'refit'))
            for r in submit(jobs):
                c=r['candidate']; key=f"{c['epoch']}:{c.get('alpha')}"; p=r['predictions'][key]
                for ix,pred in zip(s['validation'],p): all_pred.append(dict(scope=scope,family=c['family'],seed=r['seed'],row_index=ix,SampleID=str(df.iloc[ix].SampleID),YearDay=int(df.iloc[ix].YearDay),observed=float(y[ix]),predicted=pred,units=UNITS))
            export_evaluation(dest,[r for r in all_pred if (r['scope']=='final')==(scope=='final')])
        assert count=={'neural':1888,'random_forest':472},count
        dump(root/'completion.json',dict(status='complete',counts=count,elapsed_seconds=time.time()-started))
    finally:
        for pool in pools: pool.shutdown(wait=True)
    return root
