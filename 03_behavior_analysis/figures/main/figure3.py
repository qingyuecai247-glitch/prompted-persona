"""Dependency-free vector render of archived Figure 3 results; no refitting."""
from pathlib import Path
import csv,re,json,html,os
from collections import Counter
REPO=Path(__file__).resolve().parents[3]
ROOT=Path(os.environ.get('PERSONA_REFERENCE_ROOT',REPO/'data/reference'))
OUT=Path(os.environ.get('PERSONA_FIGURE_OUTPUT',REPO/'outputs/figures'));OUT.mkdir(parents=True,exist_ok=True)
keys=['human','deepseek','grok3','gpt52','qwen32','phimed','gpt4omini','llama']
labels=['Human','DeepSeek-3.2','Grok-3','GPT-5.2','Qwen2.5-32B','Phi-3-medium','GPT-4o-mini','Llama-3.1-8B']
cnames=['Human','DeepSeek-3.2','Grok-3','GPT-5.2','Qwen2.5-32B-Instruct','Phi-3-medium-4k-instruct','GPT-4o-mini','Llama-3.1-8B-Instruct']
traits=['O','C','E','A','N'];colors=dict(zip(traits,['#D1494E','#3478B7','#F08A18','#4FA64F','#8255A6']))
rows=list(csv.DictReader((ROOT/'analysis/results/factor_modularity_centered_diamonds_phi_medium/representative_factor_loadings.csv').open()))
ci={}
for line in (ROOT/'Appendix_section/generated_factor_tables/fmi_estimates_current.tex').read_text().splitlines():
 p=[x.strip() for x in line.split('&')]
 if len(p)==5:
  try:ci[p[0]]=tuple(float(v) for v in p[1:4])
  except ValueError:pass
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="183mm" height="121mm" viewBox="0 0 1100 727"><rect width="1100" height="727" fill="white"/><g font-family="Arial, Helvetica, sans-serif" fill="#303030">']
def text(x,y,s,size=12,anchor='start',bold=False,color='#303030'):
 svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{"bold" if bold else "normal"}" fill="{color}">{html.escape(s)}</text>')
def line(x,y,x2,y2,color='#555555',width=1,extra=''):
 svg.append(f'<line x1="{x}" y1="{y}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" {extra}/>')
def rect(x,y,w,h,color,alpha=1):svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" opacity="{alpha}"/>')
def color(v):
 a=(248,248,248);b=(47,111,176) if v<0 else (201,69,69);t=min(abs(v),1)
 return '#'+''.join(f'{round(x*(1-t)+z*t):02x}' for x,z in zip(a,b))
text(13,17,'a',16,bold=True);est=[]
for j,(key,label,cname) in enumerate(zip(keys,labels,cnames)):
 group=[r for r in rows if r['model_key']==key]
 items=sorted({r['item'] for r in group},key=lambda s:(traits.index(s.split('_')[1]),s));index={x:i for i,x in enumerate(items)}
 mat=[[0]*5 for _ in items]
 for r in group:mat[index[r['item']]][int(r['factor_id'])-1]=float(r['loading'])
 fmi=[]
 for f in range(5):
  counts=Counter(items[i].split('_')[1] for i in range(40) if abs(mat[i][f])>.5)
  if counts:fmi.append(max(counts.values())/sum(counts.values()))
 point=sum(fmi)/len(fmi);est.append(dict(model=key,FMI=point,ci_low=ci[cname][1],ci_high=ci[cname][2],bootstrap_replicates=200))
 x=22+(j%4)*273;y=48+(j//4)*247;h=207;cw=21
 text(x+128,y-22,label,14,anchor='middle',bold=True)
 for f in range(5):text(x+25+cw*(f+.5),y-6,f'F{f+1}',11,anchor='middle',bold=True)
 for i,item in enumerate(items):
  yy=y+h-(i+1)*h/40;t=item.split('_')[1]
  for f in range(5):rect(x+25+f*cw,yy,cw-.45,h/40-.25,color(mat[i][f]))
  for f in range(5):
   v=abs(mat[i][f]);dest=y+h-(5.2+f*7.4)*h/40
   if v>.3:svg.append(f'<path d="M {x+133} {yy+h/80} C {x+168} {yy+h/80}, {x+192} {dest}, {x+235} {dest}" fill="none" stroke="{colors[t]}" stroke-width="{.5+v*1.4}" opacity="{.28+v*.48}"/>')
 for ti,t in enumerate(traits):
  yy=y+h-(ti*8+4)*h/40;text(x+10,yy+4,t,13,anchor='middle',bold=True,color=colors[t]);line(x+25,y+h-ti*8*h/40,x+130,y+h-ti*8*h/40,'white',1)
 for f in range(5):
  yy=y+h-(5.2+f*7.4)*h/40;line(x+236,yy-6,x+236,yy+6);text(x+242,yy+4,f'F{f+1}',11,bold=True)
text(28,528,'Big Five:',12,bold=True)
for t,label,x in zip(['N','A','E','C','O'],['Neuroticism','Agreeableness','Extraversion','Conscientiousness','Openness'],[135,278,438,584,774]):
 svg.append(f'<circle cx="{x}" cy="524" r="3.7" fill="{colors[t]}"/>');text(x+9,528,t+'  '+label,10.5)
# Loading-scale strip to the right, with no panel heading.
for i in range(130):rect(947+i,519,1.2,8,color(-1+2*i/129))
for val,x in [('-1',947),('0',1012),('1',1077)]:text(x,540,val,9,anchor='middle')
text(1012,554,'Factor loading',10,anchor='middle')
text(13,565,'b',16,bold=True)
x0=222;x1=1020
mapx=lambda v:x0+(v-.35)/.70*(x1-x0)
for v in [.4,.6,.8,1.]:line(mapx(v),566,mapx(v),674,'#ECEFF2',.7)
line(mapx(est[0]['FMI']),566,mapx(est[0]['FMI']),674,'#8296AA',1,extra='stroke-dasharray="4 3"')
for i,(d,label) in enumerate(zip(est,labels)):
 yy=574+i*13.7
 text(203,yy+4,label,12,anchor='end',bold=(i==0))
 line(mapx(d['ci_low']),yy,mapx(d['ci_high']),yy,'#2F6FB0',1.8)
 svg.append(f'<circle cx="{mapx(d["FMI"])}" cy="{yy}" r="3.3" fill="#2F6FB0" stroke="white" stroke-width=".6"/>')
 text(mapx(d['ci_high'])+9,yy+3.5,f'{d["FMI"]:.2f}',10.5,color='#505050')
line(x0,680,x1,680,'#454545',.8)
for v in [.4,.6,.8,1.]:
 x=mapx(v);line(x,680,x,684);text(x,697,f'{v:.1f}',11,anchor='middle')
text((x0+x1)/2,717,'Factor Modularity Index (FMI)',13,anchor='middle')
svg.append('</g></svg>');(OUT/'figure3_compact.svg').write_text('\n'.join(svg))
with (OUT/'figure3_compact_source_data.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=est[0]);w.writeheader();w.writerows(est)
(OUT/'figure3_compact_provenance.json').write_text(json.dumps({'refitted':False,'point_estimates':'archived factor loadings, |loading|>0.50','confidence_intervals':'existing Supplementary table, three-decimal rounded 95% percentile intervals; 200 bootstrap resamples','size_mm':[183,121],'Human':est[0],'factor_alignment':'source-specific column order and signs; no cross-source alignment'},indent=2))
print('Saved SVG and source data. Human:',est[0])
