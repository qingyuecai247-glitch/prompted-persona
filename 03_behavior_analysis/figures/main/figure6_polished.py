"""Polished abc/full-width-d Figure 6, redrawn from the archived source data."""
from pathlib import Path
import os
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get('PERSONA_REFERENCE_ROOT', REPO/'data/reference'))/'figure6/analysis/results'
OUT = Path(os.environ.get('PERSONA_FIGURE_OUTPUT', REPO/'outputs/figures'))
BLUE='#2F6F96'; ORANGE='#D47A0B'; DARK='#30343A'; GREY='#737D88'
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':7,
    'xtick.labelsize':6.2,'ytick.labelsize':6.2,'axes.linewidth':.6,
    'axes.edgecolor':DARK,'axes.labelcolor':DARK,'text.color':DARK,
    'xtick.color':DARK,'ytick.color':DARK,'axes.spines.top':False,'axes.spines.right':False,
    'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})


def heading(fig, ax, letter, title):
    pos=ax.get_position()
    fig.text(pos.x0-.045, pos.y1+.018, letter, fontsize=9, fontweight='bold',va='bottom')
    ax.set_title(title,loc='left',pad=7,fontsize=7)


def make_figure():
    OUT.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(8.2,5.2),facecolor='white')
    a=fig.add_axes([.105,.58,.225,.35]);b=fig.add_axes([.44,.58,.18,.35]);c=fig.add_axes([.775,.58,.21,.35])
    # 0.35 * 5.2 * 72 = 131.04 pt, vs 154.17 pt in the author's d panel.
    d=fig.add_axes([.105,.10,.655,.35]);values=fig.add_axes([.79,.10,.195,.35],sharey=d)
    umap=pd.read_csv(DATA/'umap_6subplots_nature_optimized_source_data.csv')
    umap=umap[umap.panel.astype(str).str.lower().eq('a')]
    offsets={'B5O_5':(5,-5),'S8A_2':(5,6),'S8O_1':(5,-3),'B5A_6':(4,4),'S8N_1':(5,-7),
             'B5E_3':(-5,6),'S8De_2':(-5,-6),'B5E_8':(5,0),'B5A_4':(-5,5),'B5C_7':(5,-1)}
    for key,g in umap.groupby('id_s',sort=False):
        a.scatter(g.UMAP1,g.UMAP2,s=3.2,color=g.color_hex.iloc[0],alpha=.62,linewidths=0)
        label=g.display_label.iloc[0];xy=g[['UMAP1','UMAP2']].median().to_numpy();off=offsets[label]
        a.annotate(label,xy=xy,xytext=off,textcoords='offset points',fontsize=5.4,color=GREY,
                   ha='left' if off[0]>0 else 'right',va='center')
    a.set(xlim=(-2.3,23.3),ylim=(-7.4,22.5),xlabel='UMAP 1',ylabel='UMAP 2')
    a.set_xticks([0,5,10,15,20]);a.set_yticks([-5,0,5,10,15,20])
    heading(fig,a,'a','Situation-centred geometry')

    variance=pd.read_csv(DATA/'two_factor_variance_decomposition.csv')
    v=variance[variance.analysis_split.eq('heldout') & variance.space.eq('pca')].iloc[0]
    names=['persona','situation','interaction_plus_residual'];est=np.array([v['eta2_'+x] for x in names])
    lo=np.array([v['eta2_'+x+'_ci_low'] for x in names]);hi=np.array([v['eta2_'+x+'_ci_high'] for x in names])
    b.bar(range(3),est,width=.60,color=[BLUE,ORANGE,'#B9C0C8'],zorder=2)
    b.errorbar(range(3),est,yerr=np.vstack([est-lo,hi-est]),fmt='none',ecolor=DARK,elinewidth=.7,capsize=2,capthick=.7,zorder=3)
    for i,e in enumerate(est):b.text(i,hi[i]+.02,f'{e:.3f}',ha='center',fontsize=6.3)
    b.set_xticks(range(3),['Persona','Situation','Interaction\n+ residual']);b.tick_params(axis='x',length=0,pad=4,labelsize=5.8)
    b.set(ylim=(0,.75),ylabel='Proportion of variance');b.set_yticks([0,.2,.4,.6])
    heading(fig,b,'b','Held-out variance decomposition')

    traits=['Extraversion','Agreeableness','Conscientiousness','Neuroticism','Openness']
    decoding=pd.read_csv(DATA/'heldout_trait_decoding.csv').set_index('trait_name').loc[traits]
    y=np.arange(5)[::-1];est=decoding.heldout_pearson_r.to_numpy();lo=decoding.heldout_pearson_ci_low.to_numpy();hi=decoding.heldout_pearson_ci_high.to_numpy()
    c.axvline(0,color='#A2ADBA',lw=.55,ls=(0,(3,3)))
    c.errorbar(est,y,xerr=np.vstack([est-lo,hi-est]),fmt='o',color=BLUE,markersize=3.4,elinewidth=.9,capsize=0)
    c.set_yticks(y,traits);c.tick_params(axis='y',length=0,pad=4,labelsize=6.1)
    c.set(xlim=(-.05,.98),ylim=(-.35,4.35),xlabel='Held-out Pearson $r$ (95% CI)');c.set_xticks([0,.25,.5,.75])
    heading(fig,c,'c','Trait information remains recoverable')

    causal=pd.read_csv(DATA/'sae_intervention_forest_source_data.csv');y=np.arange(len(causal))[::-1]
    bound=float(causal.equivalence_bound.iloc[0]);d.axvspan(-bound,bound,color='#F6F8FB',zorder=0)
    for edge in [-bound,bound]:d.axvline(edge,color='#A2ADBA',lw=.55,ls=(0,(3,3)),zorder=1)
    d.axvline(0,color='#58616C',lw=.7,zorder=1)
    labels=['Overall',*traits,'Positive direction','Negative direction']
    for i,row in causal.iterrows():
        colour=DARK if row.group=='primary' else ORANGE if row.group=='direction' else BLUE
        d.errorbar(row.estimate,y[i],xerr=[[row.estimate-row.ci_low],[row.ci_high-row.estimate]],fmt='o',
                   color=colour,markersize=3.5 if row.group=='primary' else 3.2,elinewidth=.9,capsize=1.6,capthick=.65,zorder=3)
        values.text(.22,y[i],f'{row.estimate: .3f}',ha='right',va='center',fontsize=6.1,color=DARK if i==0 else GREY,fontfamily='DejaVu Sans Mono')
        values.text(.99,y[i],f'[{row.ci_low: .3f}, {row.ci_high: .3f}]',ha='right',va='center',fontsize=6.0,color=DARK if i==0 else GREY,fontfamily='DejaVu Sans Mono')
    for edge in [6.5,1.5]:
        d.axhline(edge,color='#E3E7ED',lw=.45,zorder=1);values.axhline(edge,color='#E3E7ED',lw=.45)
    d.set_yticks(y,labels);d.get_yticklabels()[0].set_fontweight('bold');d.tick_params(axis='y',length=0,pad=5)
    d.set(xlim=(-.17,.17),ylim=(-.4,7.4),xlabel='Selected-feature minus matched-control effect (state-score points)')
    d.set_xticks([-.15,-.10,-.05,0,.05,.10,.15])
    heading(fig,d,'d','Matched-control causal effects')
    d.text(1,1.025,'equivalence bounds ±0.15',transform=d.transAxes,ha='right',va='bottom',fontsize=5.7,color=GREY)
    values.text(.10,1.025,'Estimate',transform=values.transAxes,ha='center',va='bottom',fontsize=6.1,color=GREY)
    values.text(.71,1.025,'95% CI',transform=values.transAxes,ha='center',va='bottom',fontsize=6.1,color=GREY)
    values.set_xlim(0,1);values.axis('off')
    stem=OUT/'sae_internal_functional_control_polished'
    for ext in ['pdf','svg','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=400,facecolor='white')
    fig.savefig(Path(str(stem)+'_preview.png'),dpi=160,facecolor='white')
    plt.close(fig)
    manifest={'refitted':False,'layout':'abc upper row, full-width d with aligned numerical columns below',
        'd_height_pt':131.04,'author_d_height_pt':154.167652,'d_height_ratio':131.04/154.167652,
        'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.csv')}}
    stem.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved',stem)

if __name__=='__main__':make_figure()
