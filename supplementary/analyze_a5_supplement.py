from __future__ import annotations
import csv, math
from pathlib import Path
from statistics import mean, stdev
from scipy import stats

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'independent-results'/'raw'
OUT=ROOT/'analysis'
rows=[]
for path in RAW.glob('*/summary.csv'):
    with path.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            r['rate']=int(r['rate']);r['train_seed']=int(r['train_seed'])
            r['independent_test_score']=float(r['independent_test_score'])
            rows.append(r)
scenes=[('LF','Claranet_failure.gml',500),('Abilene','Abilene.gml',500),('NSFCNET','NFSCNET.gml',500),('Surge','Claranet.gml',1500)]

def vals(arm,topo,rate):
    return [r['independent_test_score'] for r in sorted(rows,key=lambda x:x['train_seed']) if r['arm']==arm and r['topo']==topo and r['rate']==rate]

def paired(a,b,topo,rate):
    ds=[x-y for x,y in zip(vals(a,topo,rate),vals(b,topo,rate))]
    avg=mean(ds);sd=stdev(ds);half=stats.t.ppf(.975,9)*sd/math.sqrt(10)
    return {'mean_diff':avg,'ci95_low':avg-half,'ci95_high':avg+half,'cohens_dz':avg/sd if sd else 0,'t_p':float(stats.ttest_1samp(ds,0).pvalue),'wilcoxon_p':float(stats.wilcoxon(ds).pvalue)}

def write(name,data):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

a5o=[]
for scene,topo,rate in scenes:
    ospf=vals('OSPF',topo,rate)[0]
    ds=[x-ospf for x in vals('A5',topo,rate)]
    avg=mean(ds);sd=stdev(ds);half=stats.t.ppf(.975,9)*sd/math.sqrt(10)
    a5o.append({'scenario':scene,'mean_diff':avg,'ci95_low':avg-half,'ci95_high':avg+half,'cohens_dz':avg/sd,'t_p':float(stats.ttest_1samp(ds,0).pvalue)})
write('a5_vs_ospf_independent.csv',a5o)

extra=[]
for factor,a,b in [('Total budget','A5','A1b'),('Adaptation over zero-shot','A5','A0')]:
    for scene,topo,rate in scenes:
        r=paired(a,b,topo,rate);r.update({'factor':factor,'scenario':scene,'arm_a':a,'arm_b':b,'topo':topo,'rate':rate,'n_seeds':10})
        extra.append(r)
write('a5_budget_zero_independent.csv',extra)

with (OUT/'factor_contrasts_independent.csv').open(newline='',encoding='utf-8') as f:
    base=[r for r in csv.DictReader(f) if r['factor'] in ('PBRS','Pretraining','Freezing joint configuration')]
fields=['factor','scenario','arm_a','arm_b','topo','rate','n_seeds','mean_diff','ci95_low','ci95_high','cohens_dz','t_p','wilcoxon_p']
allrows=base+extra
with (OUT/'factor_contrasts_a5_independent.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(allrows)
print('A5 vs OSPF')
for r in a5o: print(r)
print('A5 budget/zero')
for r in extra: print(r['factor'],r['scenario'],r['mean_diff'],r['ci95_low'],r['ci95_high'])
