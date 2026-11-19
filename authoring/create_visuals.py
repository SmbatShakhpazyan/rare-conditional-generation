"""Vector research figures from saved records and labeled analytical examples.

No model calls, dataset images, new KID computation or simulated efficacy data.
SVGs are editable scientific figures; PNGs are rasterized with bundled sharp.
"""
from pathlib import Path
from html import escape
import json
import math
import random

HERE=Path(__file__).resolve().parent
DATA=json.loads((HERE.parent/'records/report_tables.json').read_text())
FIG=HERE.parent/'figures'; FIG.mkdir(parents=True,exist_ok=True)
BG='#ffffff'; INK='#182b3a'; MUTED='#536473'; GRID='#dce3e8'
B='#007b83'; C='#b95335'; LIGHT='#eaf4f4'; GOLD='#b5791e'; PURPLE='#6c5599'
INDEX=[]

def tx(x,y,s,size=28,anchor='start',fill=INK,weight='normal'):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{escape(str(s))}</text>'
def line(x1,y1,x2,y2,color=GRID,w=2,dash=''):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'
def rect(x,y,w,h,fill=LIGHT,stroke='none',rx=0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
def circle(x,y,r=7,fill=B):return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>'
def path(points,color=B,w=4,fill='none',dash=''):
    return '<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+f'" fill="{fill}" stroke="{color}" stroke-width="{w}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'
def arrow(x1,y1,x2,y2,color=MUTED,w=4):
    a=math.atan2(y2-y1,x2-x1); head=16
    p=[(x2-head*math.cos(a-.45),y2-head*math.sin(a-.45)),(x2,y2),(x2-head*math.cos(a+.45),y2-head*math.sin(a+.45))]
    return line(x1,y1,x2,y2,color,w)+path(p,color,w)
def box(x,y,w,h,title,lines,fill=LIGHT):
    s=rect(x,y,w,h,fill,GRID,12)+tx(x+w/2,y+42,title,29,'middle',weight='bold')
    for j,t in enumerate(lines):s+=tx(x+w/2,y+83+36*j,t,25,'middle')
    return s
def start(title,subtitle='',w=1800,h=1050):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="{BG}"/><g font-family="Liberation Sans, Arial, sans-serif">'+tx(70,66,title,38,weight='bold')+(tx(70,108,subtitle,25,fill=MUTED) if subtitle else '')
def save(name,s,caption,kind,source,h=1050):
    p=FIG/(name+'.svg');p.write_text(s+'</g></svg>')
    INDEX.append(dict(id=name,path=str(FIG/(name+'.png')),svg=p.name,caption=caption,kind=kind,source=source,height=h))
def axes(x,y,w,h,ymax,yticks,xlabels,title_y=''):
    s=''
    for v in yticks:
        yy=y+h-h*v/ymax;s+=line(x,yy,x+w,yy)+tx(x-18,yy+9,f'{v:g}',24,'end',MUTED)
    s+=line(x,y,x,y+h,MUTED)+line(x,y+h,x+w,y+h,MUTED)
    for i,l in enumerate(xlabels):s+=tx(x+w*(i+.5)/len(xlabels),y+h+48,l,27,'middle')
    if title_y:s+=tx(x,y-26,title_y,25,fill=MUTED)
    return s
def grouped_bars(name,title,subtitle,labels,series,ymax,ticks,caption,source,fmt='.3f',h=1000):
    x,y,w,hh=150,220,1530,570;s=start(title,subtitle,h=h)
    s+=axes(x,y,w,hh,ymax,ticks,labels)
    n=len(series);slot=w/len(labels);bw=min(130,slot/(n+1))
    for a,(label,values,color) in enumerate(series):
        s+=rect(160+a*460,150,30,22,color)+tx(205+a*460,171,label,27)
        for i,v in enumerate(values):
            xx=x+slot*(i+.5)+(a-(n-1)/2)*bw-bw*.42;bh=hh*v/ymax
            s+=rect(xx,y+hh-bh,bw*.84,bh,color)+tx(xx+bw*.42,y+hh-bh-14,format(v,fmt),24,'middle')
    s+=tx(x,y+hh+105,'Condition in order 00  01  10  11' if labels==['00','01','10','11'] else '',24,fill=MUTED)
    save(name,s,caption,'measured saved results',source,h)

def main():
    # 01 Mechanism map, with deliberate separation of overlapping families.
    s=start('Generative model mechanisms','Conceptual map of the reviewed families; these are not additional trained benchmark arms.',h=1170)
    cells=[('VAE','Log likelihood lower bound','Sample prior and decoder'),('Conditional GAN','Adversarial game','One generator forward'),('Autoregressive','Normalized factors','Sequential ancestral draws'),('Real NVP flow','Change of variables','Invert the learned map'),('DDPM diffusion','Noise-prediction loss','Discrete reverse chain'),('Score and SDE','Denoising score loss','Langevin or reverse SDE'),('Flow matching','Velocity regression','Integrate a learned ODE'),('Consistency','Trajectory agreement','Direct endpoint map'),('Latent diffusion','Learned compression','Denoise latent then decode')]
    for i,(a,b,c) in enumerate(cells):
        r,col=divmod(i,3);s+=box(70+col*570,160+r*310,530,250,a,[b,c],LIGHT if col%2==0 else '#f1eef7')
    save('v01_model_mechanisms',s,'Reviewed model mechanisms. Each node identifies the training signal and sampling operation. The categories overlap, and only the B/C conditional GAN cohort supplies the reported training benchmark.','conceptual diagram','Primary sources E01–E10 E30 E31',1170)

    # 02 Same endpoint distributions, illustrative paths only.
    s=start('Different paths from a prior to data','Analytical illustration only. No points are generated by a trained model.',h=1000)
    rng=random.Random(17)
    for panel,(label,color) in enumerate([('Straight transport path',B),('Noisy reverse path',PURPLE),('Endpoint consistency map',GOLD)]):
        xx=80+panel*570;s+=tx(xx+245,185,label,29,'middle',weight='bold')
        for i in range(18):
            a=(xx+35+rng.random()*60,max(270,min(690,350+rng.gauss(0,90))));b=(xx+400+rng.random()*75,max(270,min(690,530+rng.gauss(0,70))))
            if panel==0:pts=[a,b]
            elif panel==1:pts=[(a[0]+(b[0]-a[0])*t/12,max(270,min(690,a[1]+(b[1]-a[1])*t/12+rng.gauss(0,40)))) for t in range(13)]
            else:pts=[a,(xx+270,a[1]*.3+b[1]*.7),b]
            s+=path(pts,color,2)+circle(*a,5,color)+circle(*b,5,C)
        s+=tx(xx+65,760,'Prior',26,'middle')+tx(xx+440,760,'Data',26,'middle')
    s+=tx(90,890,'Path geometry does not establish fidelity, semantic adherence or numerical exactness.',28,fill=MUTED)
    save('v02_probability_paths',s,'Illustrative paths associated with transport, noisy diffusion and endpoint mappings. The trajectories are drawn for explanation, not obtained from trained models or used as an efficacy comparison.','analytical illustration','Constructed path schematic; E05 E08 E09 E10',1000)

    # 03 Study information flow.
    s=start('Leakage controlled model comparison','Teacher and target statistics use TRAIN. Selection uses VAL. TEST evaluates frozen selected recipes.',h=1080)
    s+=box(60,180,440,190,'Canonical image audit',['Identity plus exact / flip links','Declared near duplicate relation'])
    s+=box(670,180,460,190,'Union components',['Quarantine conflicts','Exclude prior held out content'])+arrow(500,275,665,275)
    s+=box(1300,180,430,190,'Fixed hash assignment',['No partition search','Whole components stay together'])+arrow(1130,275,1295,275)
    for xx,title,lines in [(70,'TRAIN  112,615',['Fit teacher and target statistics','Train each selected recipe']), (675,'VAL  23,156',['Ten checkpoints / five trajectories','Choose lambda kappa and EMA']), (1280,'TEST  25,217',['Six fresh confirmation models','Three fixed paired seeds'])]:
        s+=box(xx,560,450,210,title,lines)
    s+=arrow(1500,370,1500,475)+line(295,475,1500,475,MUTED,4)
    for xx in [295,900,1500]:s+=arrow(xx,475,xx,550)
    s+=arrow(1125,665,1275,665)
    s+=tx(900,910,'Disjointness is guaranteed only under the declared relations.',30,'middle',weight='bold')
    save('v03_study_design',s,'Protocol information flow. Components, rather than filenames alone, define the new partitions. Validation-selected recipes are retrained at three fresh paired seeds before test evaluation.','protocol diagram','Frozen split and study protocol',1080)

    # 04 Exact image allocation including exclusions.
    vals=[112615,23156,25217,41611];names=['TRAIN','VAL','TEST','Excluded'];colors=[B,'#78a9ad',PURPLE,'#c5cdd3']
    s=start('Where the 202,599 canonical images went','The excluded count includes old held out components; it is not the number of annotation conflicts.',h=800)
    x=90
    for v,n,col in zip(vals,names,colors):
        ww=1620*v/sum(vals);s+=rect(x,260,ww,190,col)+tx(x+ww/2,350,f'{v:,}',31,'middle','#ffffff' if col not in ['#c5cdd3'] else INK,'bold');x+=ww
    for i,(v,n,col) in enumerate(zip(vals,names,colors)):
        xx=90+i*420;s+=rect(xx,540,28,28,col)+tx(xx+42,565,n,28)+tx(xx,625,f'{100*v/sum(vals):.2f}%',37,weight='bold')
    save('v04_image_allocation',s,'Exact image allocation after component exclusion, conflict quarantine and fixed hash assignment. Counts sum to 202,599. The four conflict components contain 219 rows; the larger excluded total has several causes.','measured data audit','Saved component split report',800)

    # 05 group counts displayed on a linear square-root axis with explicit raw labels.
    s=start('Image support is highly uneven across conditions','Bar height uses square root scaling for readability; labels give the exact image counts.',h=1000)
    groups=['00','01','10','11'];x,y,w,hh=140,230,1540,540;top=math.sqrt(55000)
    for v in [0,1000,5000,15000,30000,50000]:
        yy=y+hh-hh*math.sqrt(v)/top;s+=line(x,yy,x+w,yy)+tx(x-20,yy+8,f'{v:,}',23,'end')
    for a,row in enumerate(DATA['E1_component_split']['rows']):
        color=[B,PURPLE,GOLD][a];s+=rect(170+a*490,150,28,24,color)+tx(215+a*490,173,row['partition'].upper(),28)
        for g,v in enumerate(row['image_counts']):
            xx=x+(g+.5)*w/4+(a-1)*90-35;bh=hh*math.sqrt(v)/top;s+=rect(xx,y+hh-bh,70,bh,color)+tx(xx+35,y+hh-bh-18,f'{v:,}',22,'middle')
    for g in range(4):s+=tx(x+(g+.5)*w/4,y+hh+55,groups[g],30,'middle')
    s+=tx(900,925,'The rare condition has 942 training images in 282 union components.',29,'middle')
    save('v05_condition_support',s,'Image counts for every partition and annotation bin. The square-root vertical scale is explicitly labeled; values above bars are untransformed counts.','measured data audit','E1_component_split',1000)

    # 06 Actual earlier source exposure, not quality.
    grouped_bars('v06_earlier_content_exposure','Earlier held out content had training linked equivalents','Preliminary audit only; these are source-image presentation counts, not new quality results.',['101','202','303'],[('Linked to old VAL',[28,33,30],B),('Linked to old TEST',[67,78,73],C)],90,[0,20,40,60,80], 'Training presentations in three earlier sampling histories linked to exact-equivalent old validation or test content. The twelve earlier runs share three distinct sampling histories.','Saved preliminary exposure audit',fmt='.0f')

    # 07 Architecture widths with exact trainable count totals.
    s=start('The common residual conditional GAN','Both methods have the same learned architecture. C adds a separate fixed discriminator teacher.',h=900)
    widths=[128,128,64,32,16];sizes=['4 x 4','8 x 8','16 x 16','32 x 32','64 x 64']
    for i,(ch,res) in enumerate(zip(widths,sizes)):
        xx=110+i*315;hh=150+25*i;s+=box(xx,430-hh/2,240,hh,f'{ch} channels',[res],LIGHT)
        if i<4:s+=arrow(xx+245,430,xx+305,430)
    s+=tx(80,190,'G: Gaussian z in 128 dimensions + four condition labels',31,weight='bold')
    s+=tx(80,620,'Shared BN -> ReLU -> 3 x 3 convolution -> tanh; RGB output 3 x 64 x 64',29)
    s+=tx(80,690,'G 737,315 trainable parameters',31,fill=B,weight='bold')+tx(80,745,'D 888,545 trainable parameters; 128 dimensional sum pooled features',31)
    s+=tx(80,805,'C teacher 888,545 fixed parameters; no extra trainable generator parameters',29,fill=MUTED)
    save('v07_gan_architecture',s,'Generator feature widths and resolutions. Appendix A gives exact block operations and parameter counts. Both B and C use the same generator and discriminator; the extra C teacher is frozen.','architecture diagram','Executed models.py and parameter-count records',900)

    # 08 all 10 validation checkpoints, all conditions.
    rows=DATA['E3_all_validation_points'];values=[[r['kid_by_group'][g]['x1000'] for g in groups] for r in rows]
    lo=min(min(v) for v in values);hi=max(max(v) for v in values)
    s=start('Every validation result is reported','KID x 1000; lower is better. Two EMAs per trajectory give ten points, not ten independent trainings.',h=1250)
    for j,g in enumerate(groups):s+=tx(800+j*240,178,g,30,'middle',weight='bold')
    for i,(r,vs) in enumerate(zip(rows,values)):
        yy=210+i*86;lab=f"{r['method']}  lambda {r['lambda']:g}  kappa {r['kappa']:g}  EMA {r['ema_decay']:g}";s+=tx(70,yy+45,lab,25)
        for j,v in enumerate(vs):
            a=(v-lo)/(hi-lo);rgb=tuple(round((1-a)*p+a*q) for p,q in zip((231,246,245),(196,88,56)));col='#'+''.join(f'{v:02x}' for v in rgb)
            s+=rect(690+j*240,yy,220,67,col)+tx(800+j*240,yy+43,f'{v:.3f}',28,'middle',INK if a<.65 else '#ffffff')
    s+=tx(70,1140,'Selected C: lambda 1, kappa 0, EMA 0.99. The tolerance is inactive.',30,weight='bold')
    save('v08_validation_heatmap',s,'All ten saved validation points and all four conditional scores. Color spans the displayed score range. Selection uses the rare condition 11, not a favorable subset of test results.','measured saved results','E3_all_validation_points',1250)

    # 09 exact EMA initialization coefficient, not images.
    s=start('Slow EMA retains initialization at short horizons','Analytical parameter coefficient eta raised to the number of successful updates; not an image mixture.',h=950)
    x,y,w,hh=160,210,1490,560
    for v in [0,.25,.5,.75,1]:yy=y+hh*(1-v);s+=line(x,yy,x+w,yy)+tx(x-20,yy+8,f'{v:g}',26,'end')
    for eta,col in [(.99,B),(.999,C)]:
        s+=path([(x+w*t/1500,y+hh*(1-eta**t)) for t in range(0,1501,5)],col,6)
        s+=tx(950,280 if eta==.99 else 335,f'eta {eta}: coefficient at 1500 = {eta**1500:.6g}',29,fill=col,weight='bold')
    for t in [0,300,600,900,1200,1500]:s+=tx(x+w*t/1500,y+hh+44,str(t),26,'middle')
    s+=tx(900,895,'Successful generator updates',29,'middle')
    save('v09_ema_retention',s,'Initialization coefficient in the exact parameter EMA recursion. This mathematical calculation does not establish that EMA initialization caused the observed quality gap; BN buffers were copied separately.','analytical calculation','eta^T from the saved EMA grid',950)

    # 10 paired test results.
    points={(r['method'],r['seed']):r['kid_by_group']['11']['x1000'] for r in DATA['E4_all_confirmation_test_points']}
    s=start('Three fresh paired seeds expose heterogeneous outcomes','Rare condition KID x 1000; lower is better. All six confirmation models are shown.',h=1000)
    x1,x2,y0,hh,low,high=460,1280,220,580,140,230
    Y=lambda v:y0+hh*(high-v)/(high-low)
    for v in range(140,231,10):s+=line(220,Y(v),1530,Y(v))+tx(190,Y(v)+8,str(v),26,'end')
    for seed,col in [(101,B),(202,PURPLE),(303,GOLD)]:
        b,c=points['B',seed],points['C',seed];s+=line(x1,Y(b),x2,Y(c),col,5)+circle(x1,Y(b),12,col)+circle(x2,Y(c),12,col)
        s+=tx(x1-30,Y(b)+9,f'{seed}: {b:.3f}',27,'end',col)+tx(x2+25,Y(c)+9,f'{c:.3f}',27,fill=col)
    s+=tx(x1,870,'Balanced baseline B',31,'middle',weight='bold')+tx(x2,870,'Correction C',31,'middle',weight='bold')
    s+=tx(900,945,'Only seed 202 improves with C; the three seed mean favors B.',29,'middle')
    save('v10_paired_test_seeds',s,'Paired rare-condition outcomes for all three fixed confirmation seeds. Connecting lines denote matched seeds, not trajectories in optimization time.','measured saved results','E4_all_confirmation_test_points',1000)

    means=DATA['E4_fixed_seed_means']
    grouped_bars('v11_all_condition_means','The mean comparison favors B in all four conditions','KID x 1000; descriptive arithmetic means across the same three fixed seeds.',groups,[(r['method'],[r['kid_by_group'][g]['x1000'] for g in groups],B if r['method']=='B' else C) for r in means],220,[0,50,100,150,200], 'Four conditional mean KIDs from the saved six-model cohort. These are descriptive means over fixed seeds, not independent population estimates.','E4_fixed_seed_means')

    # 12 only nominal one-sided upper, not symmetric interval.
    ci=DATA['E5_primary_conditional_comparison'];d,u=ci['point_difference_x1000'],ci['one_sided_upper_x1000']
    s=start('The specified comparison does not demonstrate improvement','C minus B, rare condition KID x 1000. Negative favors C; positive favors B.',h=790)
    X=lambda v:180+(v+5)/35*1450
    s+=line(X(0),200,X(0),570,MUTED,3,'9 8')+line(X(d),370,X(u),370,B,7)+line(X(u),260,X(u),480,B,6)+circle(X(d),370,14,B)
    s+=tx(X(d)-20,310,f'Mean +{d:.3f}',29,'end')+tx(X(u)+20,305,f'Upper +{u:.3f}',29)
    for v in [-5,0,5,10,15,20,25,30]:s+=line(X(v),570,X(v),590,MUTED)+tx(X(v),630,str(v),26,'middle')
    s+=tx(900,710,'Nominal one sided 98.75% bound; conditional on fixed models and selected banks.',28,'middle')
    save('v12_conditional_upper_bound',s,'The point difference and nominal one-sided conditional upper bound. The horizontal segment is not a two-sided confidence interval. Training-seed populations, teacher fitting and hyperparameter selection are not resampled.','measured saved comparison','E5_primary_conditional_comparison',790)

    # 13 PRDC quantities as small multiples, preserving zero recall.
    s=start('Feature neighborhood diagnostics remain weak','PRDC, strict open balls, k = 3; means over the fixed seeds. No metric is multiplied by 1000.',h=1030)
    for panel,key in enumerate(['precision','density','coverage']):
        xx=110+panel*560;topv=max(r['prdc_mean_by_group'][g][key] for r in means for g in groups)*1.25
        s+=tx(xx+235,190,key.title(),31,'middle',weight='bold')
        for tick in [0,topv/2,topv]:
            yy=740-450*tick/topv;s+=line(xx,yy,xx+470,yy)+tx(xx-12,yy+8,f'{tick:.2f}',22,'end')
        for i,g in enumerate(groups):
            for j,r in enumerate(means):
                v=r['prdc_mean_by_group'][g][key];bh=450*v/topv;s+=rect(xx+55+i*110+j*36,740-bh,32,bh,B if r['method']=='B' else C)
            s+=tx(xx+88+i*110,785,g,24,'middle')
    s+=rect(100,860,28,24,B)+tx(145,884,'B',28)+rect(260,860,28,24,C)+tx(305,884,'C',28)
    s+=tx(900,958,'Recall is exactly zero in all 24 model by condition records.',30,'middle',weight='bold')
    save('v13_prdc_diagnostics',s,'Precision, density and coverage means in all four conditions, with separate vertical scales across panels. Recall is exactly zero in all 24 individual model-by-condition records. Zero finite-neighborhood recall is not a proof of global support disjointness.','measured saved diagnostics','E4_fixed_seed_means and E6_PRDC',1030)

    # 14 constructed null direction.
    grouped_bars('v14_contrast_counterexample','A zero signed interaction can hide the wrong distribution','Constructed scalar feature means. Real and generated groups differ identically.',groups,[('Real feature mean',[0,0,0,0],B),('Generated feature mean',[1,1,1,1],C)],1.3,[0,.25,.5,.75,1], 'Constructed counterexample to identification by the signed contrast. With c=(1,-1,-1,1)/4, both alternating means equal zero, while each group has a different feature mean. The GAN adversarial term is a separate constraint.','Elementary signed-contrast algebra',fmt='.0f')
    INDEX[-1]['kind']='constructed mathematical counterexample'

    # 15 stale-gradient example.
    s=start('A stale detached witness need not reduce the current discrepancy','Constructed scalar case: old tracker 1, target 0, scale 1, lambda 1, tolerance 0.',h=1000)
    x,y,w,hh=180,220,1440,570;X=lambda v:x+w*(v+2)/4;Y=lambda v:y+hh*(5-v)/10
    for v in [-4,-2,0,2,4]:s+=line(x,Y(v),x+w,Y(v))+tx(x-22,Y(v)+8,str(v),26,'end')
    s+=line(X(0),y,X(0),y+hh,MUTED,2,'8 7')
    ts=[-2+i/100 for i in range(401)];s+=path([(X(v),Y(v*v)) for v in ts],B,6)+path([(X(v),Y(2*v)) for v in ts],C,6)
    s+=circle(X(-1),Y(1),12,B)+circle(X(-1),Y(-2),12,C)
    s+=arrow(X(-1),Y(1)-50,X(-.3),Y(1)-50,B,6)+arrow(X(-1),Y(-2)+50,X(-1.7),Y(-2)+50,C,6)
    s+=tx(1100,280,'Current square: delta squared',29,fill=B)+tx(1100,325,'Detached surrogate: 2 delta',29,fill=C)
    for v in [-2,-1,0,1,2]:s+=tx(X(v),850,str(v),28,'middle')
    s+=tx(900,925,'At delta = -1, the two descent directions are opposite.',30,'middle',weight='bold')
    save('v15_stale_witness',s,'Scalar staleness counterexample. Arrows show descent in the scalar current contrast for each displayed objective at delta=-1. The centered linear surrogate can be negative and is not the current squared distance.','constructed mathematical counterexample','Literal auxiliary loss algebra',1000)

    # 16 complete screening relation through all 15 band pairs.
    s=start('Why six hash bands cover the declared distance threshold','If at most four bits differ, at least two of six disjoint bands match exactly.',h=900)
    sizes=[11,11,11,11,10,10]
    for i,v in enumerate(sizes):
        xx=100+i*270;s+=box(xx,190,220,145,f'Band {i+1}',[f'{v} bits'])
    yy=470
    for i in range(6):
        for j in range(i+1,6):
            k=i*5-i*(i-1)//2+(j-i-1);xx=105+(k%5)*330;ry=yy+(k//5)*92
            s+=rect(xx,ry,280,65,LIGHT,GRID,7)+tx(xx+140,ry+42,f'{i+1} + {j+1}',31,'middle')
    s+=tx(900,825,'All 15 band pairs are indexed; aligned RGB checks then decide the declared relation.',28,'middle')
    save('v16_hash_band_proof',s,'Candidate-completeness construction for a 64-bit Hamming threshold of four, using six disjoint bands and all fifteen band pairs. Completeness concerns this declared screening relation, not every visual or semantic duplicate.','proof diagram','Declared pHash candidate policy and pigeonhole argument',900)

    # 17 signed component covariance example.
    s=start('Sum group influences before squaring','Constructed two component example from the exact training-statistic definition.',h=900)
    s+=box(90,200,720,340,'Common group directions cancel',['Group influences align identically','Signed component totals: 0 and 0','Cluster variability u = 0'],LIGHT)
    s+=box(990,200,720,340,'Interaction directions reinforce',['Group influences follow + - - +','Signed component totals: -1/2 and +1/2','Cluster variability u = 1'],'#fff1e9')
    s+=tx(900,700,'Separate group variances would discard cross group covariance.',32,'middle',weight='bold')
    s+=tx(900,770,'The plug in scale is not a calibrated confidence radius.',30,'middle',fill=MUTED)
    save('v17_component_covariance',s,'Two analytical component configurations illustrate cancellation and reinforcement in the signed cluster influence formula. No new dataset estimates are shown.','constructed mathematical example','Teacher-statistic formula and CPU fixture logic',900)

    # 18 artifact availability, no inference about unknown model bytes.
    s=start('Saved results and recovered model bytes are different evidence','All ten validation points, six test points and the bootstrap report have saved aggregate values.',h=900)
    for i,(name,ready) in enumerate([('B101',True),('B202',False),('B303',False),('C101',True),('C202',True),('C303',True)]):
        xx=100+(i%3)*570;yy=210+(i//3)*230;s+=box(xx,yy,480,175,name,['Complete generator' if ready else 'Complete generator missing'],LIGHT if ready else '#fff1e9')
    s+=tx(900,755,'Four of six complete generators; teacher and reference banks remain unavailable.',30,'middle',weight='bold')
    s+=tx(900,815,'Preserved records support an arithmetic audit, not a self contained six model replay.',28,'middle',fill=MUTED)
    save('v18_artifact_availability',s,'Availability of selected confirmation-generator bytes. Metric completion records are retained for all six, but only four complete generator archives were recovered.','artifact inventory','Verified local delivery manifest',900)

    # 19 proposed algorithm, explicitly not benchmarked.
    s=start('A stronger conditional matching proposal remains untested','Prospective design only. No quality advantage or completed new experiment is claimed.',h=960)
    labels=[('Separate TRAIN representation',['Fit on a separate training fold','Freeze parameters and normalization']),('Per condition kernel targets',['Positive weights for every condition','Use only identified target groups']),('Validation and untouched test',['Declare all search costs and seeds','Evaluate after recipe selection']),('Fresh batch discrepancy',['Independent paired batches','Off diagonal kernel estimates'])]
    for i,(title,ls) in enumerate(labels):
        xx=80+(i%2)*880;yy=190+(i//2)*330;s+=box(xx,yy,760,225,title,ls,LIGHT if i<2 else '#f1eef7')
    s+=arrow(840,300,950,300)+arrow(1330,415,1330,510)+arrow(950,635,850,635)
    s+=tx(900,900,'Zero characteristic kernel MMD identifies its target space, not automatically image semantics.',27,'middle')
    save('v19_prospective_kernel_design',s,'Prospective conditional-kernel matching design. It requires identified targets for each fitted condition and new untouched confirmation data. It is not a tested improvement over B or C.','untested proposed algorithm','Deep theory supplement and established MMD properties',960)

    # 20 shareable standalone overview, also a monograph overview.
    s=start('Rare conditional generation research overview','A complete methods report, a negative short budget experiment, and an explicit roadmap for stronger tests.',h=1150)
    s+=box(65,175,780,215,'The empirical question',['Can a TRAIN only signed feature correction','improve the rare observed annotation condition?'])
    s+=box(955,175,780,215,'The comparison',['Balanced conditional GAN B vs corrected GAN C','Same backbone, updates and paired seeds'])
    s+=box(65,485,780,225,'What was measured',['112,615 TRAIN / 23,156 VAL / 25,217 TEST','Ten validation checkpoints and six test models'])
    s+=box(955,485,780,225,'What the result says',['Rare KID x 1000: B 166.663 / C 185.544','C does not demonstrate improvement'],'#fff1e9')
    s+=arrow(845,280,945,280)+arrow(1345,400,1345,475)+arrow(855,595,945,595)
    s+=tx(900,840,'Reviewed model families are explained; only B and C supply this benchmark.',31,'middle',weight='bold')
    s+=tx(900,905,'No global novelty, general convergence, semantic fairness or full replay claim.',29,'middle',fill=MUTED)
    s+=tx(900,1005,'Paper + original reference code + aggregate records + four recovered generators',29,'middle',fill=B)
    save('v20_research_overview',s,'Standalone overview of the research question, experimental controls, evidence and limits. The other reviewed model families are reference implementations, not additional benchmark arms.','research overview diagram','Verified study and model-library scope',1150)

    (FIG/'figure_index.json').write_text(json.dumps(INDEX,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'figures':len(INDEX),'directory':str(FIG)}))

if __name__=='__main__':main()
