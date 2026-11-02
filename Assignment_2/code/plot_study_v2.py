"""Report figures from exported study evidence; no fitting or result selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ORDER=['mean','temperature_ols','ridge','random_forest','shallow_mlp','residual_mlp','pca_ridge','ae_ridge','dae_ridge']
LABELS=dict(zip(ORDER,['Training mean','Temperature OLS','Ridge','Random forest','Shallow MLP','Residual MLP','PCA + Ridge','AE + Ridge','DAE + Ridge']))
COLORS=dict(zip(ORDER,['#888888','#BB7744','#555555','#009E73','#56B4E9','#0072B2','#777777','#E69F00','#D55E00']))
UNIT=r'$10^6$ cells mL$^{-1}$'

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'axes.axisbelow':True,'grid.color':'#999999','grid.alpha':.22,'grid.linewidth':.45,'savefig.dpi':300,'pdf.fonttype':42})

def save(fig,out,name):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'): fig.savefig(out/f'{name}.{ext}',bbox_inches='tight',pad_inches=.08)
    plt.close(fig)

def figure_01(data,splits,out):
    fig,axs=plt.subplots(2,1,figsize=(8.3,5.9),layout='constrained',gridspec_kw={'height_ratios':[1,1.2]})
    ax=axs[0]; ax.plot(data.YearDay,data.Bacteria_abundance/1e6,color='#333333',lw=1,marker='.',ms=3)
    development=splits['final']['train']; historical=splits['final']['validation']
    boundary=(data.YearDay.iloc[development[-1]]+data.YearDay.iloc[historical[0]])/2
    ax.axvspan(boundary,data.YearDay.max()+3,color='#E69F00',alpha=.13)
    ax.axvline(boundary,color='#777777',ls='--',lw=.8)
    ax.set(xlabel='YearDay',ylabel=f'Observed abundance ({UNIT})',title='a  Course observations and reused historical holdout')
    ax.text(.02,.95,f'Development: first {len(development)} observations',transform=ax.transAxes,va='top',fontsize=8)
    ax.text(.99,.95,f'Historical\nholdout: {len(historical)}',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    ax=axs[1]; rows=[]
    for key in ('outer_1','outer_2','outer_3'):
        rows.append((key.replace('_',' ').title(),splits[key]))
    for i,s in enumerate(splits['outer_1']['inner'],1): rows.append((f'Outer 1 · inner {i}',s))
    rows.append(('Final fit → historical',splits['final']))
    for y,(name,s) in enumerate(rows):
        for k,c in [('train','#0072B2'),('gap','#999999'),('validation','#E69F00')]:
            ix=s[k]
            if ix: ax.broken_barh([(min(ix)+.5,len(ix))],(y-.3,.6),facecolors=c,label=k if y==0 else None)
    ax.set(yticks=range(len(rows)),yticklabels=[r[0] for r in rows],xlabel='Chronological observation number',xlim=(.5,len(data)+.5),title='b  Expanding temporal blocks (inner folds illustrated for outer fold 1)')
    ax.grid(axis='y',visible=False)
    ax.invert_yaxis(); ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,frameon=False)
    save(fig,out,'figure_01')

def figure_02(metrics,out):
    individual=metrics[metrics.seed.astype(str)!='ensemble']; ensemble=metrics[metrics.seed.astype(str)=='ensemble']
    fig,axs=plt.subplots(1,2,figsize=(9.2,4.2),layout='constrained')
    ax=axs[0]
    for j,scope in enumerate(sorted(individual.scope.unique())):
        g=individual[individual.scope==scope].groupby('family').rmse.mean().reindex(ORDER)
        ax.scatter(g,np.arange(9)+(j-1)*.17,s=24,marker=['o','s','^'][j],label=scope.replace('_',' ').title())
    ax.set(yticks=range(9),yticklabels=[LABELS[f] for f in ORDER],xlabel=f'Mean individual RMSE ({UNIT})',title='a  Outer-fold errors'); ax.invert_yaxis(); ax.legend(frameon=False,fontsize=7)
    ax=axs[1]; stochastic=['random_forest','shallow_mlp','residual_mlp','ae_ridge','dae_ridge']
    for i,f in enumerate(stochastic):
        vals=individual[individual.family==f].groupby('seed').rmse.mean().to_numpy()
        ax.scatter(vals,np.full(len(vals),i)+np.linspace(-.13,.13,len(vals)),color=COLORS[f],s=14,alpha=.65)
        ax.scatter([vals.mean()],[i],marker='|',s=180,color='black',label='Mean individual' if i==0 else None)
        ax.scatter([ensemble[ensemble.family==f].rmse.mean()],[i],marker='D',s=36,facecolors='white',edgecolors='black',label='Ensemble' if i==0 else None)
    ax.set(yticks=range(5),yticklabels=[LABELS[f] for f in stochastic],xlabel=f'Three-fold mean RMSE ({UNIT})',title='b  Seeds and ten-member ensembles'); ax.invert_yaxis(); ax.legend(frameon=False,fontsize=7)
    save(fig,out,'figure_02')

def figure_03(scores,selected,candidates,out):
    lookup={c['id']:c for c in candidates}; nn=scores[scores.family.isin(['shallow_mlp','residual_mlp'])].copy()
    for field in ['neural_parameter_count','weight_decay','dropout']: nn[field]=nn.candidate_id.map(lambda k:lookup[k][field])
    avg=nn.groupby(['family','candidate_id','epoch','neural_parameter_count','weight_decay','dropout'])[['train_rmse','rmse']].mean().reset_index()
    fig,axs=plt.subplots(2,2,figsize=(8.5,6),layout='constrained')
    for f in ['shallow_mlp','residual_mlp']:
        g=avg[avg.family==f]; c=COLORS[f]
        axs[0,0].scatter(g.neural_parameter_count,g.rmse,c=c,s=20,alpha=.6,label=LABELS[f])
        best=selected[f]; h=g[g.candidate_id==best['id']].sort_values('epoch')
        axs[0,1].plot(h.epoch,h.rmse,'o-',color=c,label=LABELS[f]+' validation')
        axs[0,1].plot(h.epoch,h.train_rmse,'s--',color=c,label=LABELS[f]+' training')
        for drop,marker in [(0,'o'),(.1,'s'),(.2,'^')]:
            h=g[g.dropout==drop]; axs[1,0].scatter(h.weight_decay,h.rmse,color=c,marker=marker,s=25,alpha=.6)
        axs[1,1].scatter(g.train_rmse,g.rmse,color=c,s=20,alpha=.6)
    axs[0,0].set(xscale='log',xlabel='Network parameters',ylabel=f'Validation RMSE ({UNIT})',title='a  Capacity across fixed candidates'); axs[0,0].legend(frameon=False)
    axs[0,1].set(xlabel='Checkpoint epoch',ylabel=f'RMSE ({UNIT})',title='b  Selected base configuration'); axs[0,1].legend(frameon=False,fontsize=6.7)
    axs[1,0].set(xscale='log',xlabel='AdamW weight decay',ylabel=f'Validation RMSE ({UNIT})',title='c  Regularization (○ 0; □ 0.1; △ 0.2 dropout)')
    axs[1,1].set(xlabel=f'Training RMSE ({UNIT})',ylabel=f'Validation RMSE ({UNIT})',title='d  Fit and later-block estimation')
    save(fig,out,'figure_03')

def figure_04(scores,candidates,out):
    lookup={c['id']:c for c in candidates}; s=scores[scores.family.isin(['ae_ridge','dae_ridge','pca_ridge'])].copy()
    s['dimension']=s.apply(lambda r:lookup[r.candidate_id]['latent_dimension'] if r.family!='pca_ridge' else int(r.candidate_id.split('_')[1]),axis=1)
    av=s.groupby(['family','candidate_id','epoch','alpha','dimension'])[['rmse','validation_reconstruction_mse']].mean().reset_index()
    fig,axs=plt.subplots(1,3,figsize=(10,3.4),layout='constrained')
    for f in ['pca_ridge','ae_ridge','dae_ridge']:
        g=av[av.family==f]; best=g.loc[g.groupby('dimension').rmse.idxmin()].sort_values('dimension')
        axs[0].plot(best.dimension,best.rmse,'o-',color=COLORS[f],label=LABELS[f])
        # Each base configuration/checkpoint uses its best inner Ridge penalty.
        h=g.loc[g.groupby(['candidate_id','epoch']).rmse.idxmin()]
        axs[1].scatter(h.validation_reconstruction_mse,h.rmse,s=17,color=COLORS[f],alpha=.6)
    axs[0].set(xlabel='Latent dimension',xticks=[2,4,6],ylabel=f'Validation RMSE ({UNIT})',title='a  Best inner score at each dimension'); axs[0].legend(frameon=False,fontsize=7)
    axs[1].set(xlabel='Validation reconstruction MSE',ylabel=f'Validation RMSE ({UNIT})',title='b  Reconstruction and regression')
    paired=s[(s.family!='pca_ridge')&(s.epoch==300)&np.isclose(s.alpha,10)].copy()
    paired['pair']=paired.candidate_id.map(lambda k:k.rsplit('_',1)[-1])
    p=paired.pivot(index=['pair','inner_fold','seed'],columns='family',values='rmse'); delta=(p.dae_ridge-p.ae_ridge).groupby('pair').mean()
    axs[2].axvline(0,color='#999999',lw=.8); axs[2].scatter(delta,np.arange(len(delta)),color='#D55E00',s=20)
    axs[2].xaxis.set_major_locator(MaxNLocator(4))
    axs[2].set(yticks=np.arange(len(delta)),yticklabels=delta.index,xlabel=f'DAE − AE RMSE ({UNIT})',ylabel='Matched base configuration',title='c  Matched noise diagnostic')
    save(fig,out,'figure_04')

def figure_05(predictions,out):
    ens=predictions.groupby(['family','row_index'])[['YearDay','observed','predicted']].mean().reset_index()
    fig,axs=plt.subplots(2,1,figsize=(8.4,5.4),layout='constrained',sharex=True,sharey=True)
    for ax,fams,title in zip(axs,[['ridge','random_forest','shallow_mlp','residual_mlp'],['pca_ridge','ae_ridge','dae_ridge']],['a  Direct regression','b  Environmental representations']):
        g=ens[ens.family==fams[0]]; ax.plot(g.YearDay,g.observed,'o-',color='black',lw=1.2,ms=3,label='Observed')
        for f in fams:
            g=ens[ens.family==f]; ax.plot(g.YearDay,g.predicted,color=COLORS[f],lw=1.4,ls='-' if f in ['residual_mlp','dae_ridge'] else '--',label=LABELS[f])
        ax.set(title=title,ylabel=f'Abundance ({UNIT})'); ax.legend(ncol=3,frameon=False,fontsize=7)
    axs[1].set_xlabel('YearDay · reused historical holdout')
    save(fig,out,'figure_05')

def generate_figures(results_root,output_dir,data_path,config_path):
    """Return five PNG paths; requires completed exported CSV/JSON, not checkpoints."""
    root=Path(results_root); out=Path(output_dir); style()
    data=pd.read_csv(data_path,sep='\t').sort_values('YearDay').reset_index(drop=True)
    cfg=json.loads(Path(config_path).read_text()); splits=json.loads((root/'splits.json').read_text())
    figure_01(data,splits,out)
    outer_metrics=pd.read_csv(root/'formal/metrics.csv')
    assert set(outer_metrics.scope)=={'outer_1','outer_2','outer_3'}, 'All three outer scopes are required'
    figure_02(outer_metrics,out)
    s=pd.read_csv(root/'historical_holdout/candidate_scores.csv')
    selected=json.loads((root/'historical_holdout/selected_configurations.json').read_text())['final']
    figure_03(s,selected,cfg['candidates'],out); figure_04(s,cfg['candidates'],out)
    figure_05(pd.read_csv(root/'historical_holdout/predictions.csv'),out)
    return {f'figure_{i:02d}':str(out/f'figure_{i:02d}.png') for i in range(1,6)}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('results_root'); p.add_argument('output_dir'); p.add_argument('data_path'); p.add_argument('config_path')
    a=p.parse_args(); print(json.dumps(generate_figures(**vars(a)),indent=2))

def write_tables(results_root,report_dir):
    """Create report tables directly from individual and ensemble metric exports."""
    root=Path(results_root); out=Path(report_dir); out.mkdir(parents=True,exist_ok=True)
    for directory,filename,title in [('formal','table_02_outer.md','Table 2. Outer chronological validation'),('historical_holdout','table_03_historical.md','Table 3. Reused historical holdout')]:
        m=pd.read_csv(root/directory/'metrics.csv'); individual=m[m.seed.astype(str)!='ensemble']; ensemble=m[m.seed.astype(str)=='ensemble']
        outer=directory=='formal'; rows=[]
        for fam in ORDER:
            g=individual[individual.family==fam]; folds=g.groupby('scope')[['rmse','mae','r2']].mean()
            errors=[]
            for metric in ['rmse','mae','r2']:
                if outer: errors.append(f'{folds[metric].mean():.3f} ± {folds[metric].std(ddof=1):.3f}')
                else: errors.append(f'{g[metric].mean():.3f}')
            sd=g.groupby('scope').rmse.std(ddof=1).mean()
            rows.append([LABELS[fam]]+errors+['—' if np.isnan(sd) else f'{sd:.3f}',f'{ensemble[ensemble.family==fam].rmse.mean():.3f}'])
        intro=(f'**{title}.** RMSE and MAE use $10^6$ cells mL⁻¹. '+('Metrics are mean ± sample SD across the three outer folds after averaging seed metrics within each fold. The seed-SD column averages the within-fold RMSE SDs; ensemble RMSE averages the three fold-specific ensemble errors.' if outer else 'Individual metrics average ten seeds for stochastic models. Seed SD is the RMSE standard deviation across those ten runs. Ensemble metrics score their equal-weight mean prediction. Deterministic models use one fit.')+'\n\n')
        headers=['Pipeline','RMSE','MAE','R²','Seed SD','Ensemble RMSE']
        text=intro+'| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'
        text+='\n'.join('| '+' | '.join(r)+' |' for r in rows)+'\n'
        (out/filename).write_text(text)
