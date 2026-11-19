"""Export the complete monograph as readable Markdown with LaTeX displays."""
import argparse
import json
import re
from pathlib import Path

CLONES={
'contrast':r'c=(1,-1,-1,1)/4,\quad r=\sum_g c_g\mu_g,\quad \mu_g=\frac1{n_g}\sum_{i:g_i=g}\psi(X_i)',
'cluster':r'v_{aj}=\sum_g\frac{c_g}{n_g}\sum_{i\in a:g_i=g}(\psi_j(X_i)-\mu_{gj}),\quad u_j^2=\frac{M}{M-1}\sum_a v_{aj}^2',
'scale':r's_j=\max\left\{\left[\frac14\sum_g n_g^{-1}\sum_{i:g_i=g}(\psi_j(X_i)-\mu_{gj})^2\right]^{1/2},10^{-3}\right\}',
'witness':r'd_{t-1,j}=(\bar\delta_{t-1,j}-r_j)/s_j,\quad q_{t-1,j}=\operatorname{sign}(d_{t-1,j})(|d_{t-1,j}|-\kappa u_j/s_j)_+',
'candidate_loss':r'L_{G,C}=L_{G,B}+\frac{2\lambda}{128}\sum_j\operatorname{stopgrad}(q_{t-1,j})(\hat\delta_{t,j}-r_j)/s_j',
'batch_contrast':r'\hat\delta_t=\sum_gc_g\frac1{16}\sum_{i=1}^{16}\psi(G_{\theta_t}(z_i,g;B_t))',
'kid':r'\widehat{\mathrm{KID}}=\frac{\sum_{i\ne j}k(x_i,x_j)}{m(m-1)}+\frac{\sum_{i\ne j}k(y_i,y_j)}{n(n-1)}-\frac{2\sum_{i,j}k(x_i,y_j)}{mn},\quad k(x,y)=(1+x^Ty/2048)^3',
}
REPL={'θ':r'\theta ','δ':r'\delta ','Δ':r'\Delta ','λ':r'\lambda ','κ':r'\kappa ','η':r'\eta ','σ':r'\sigma ','μ':r'\mu ','ψ':r'\psi ','Φ':r'\Phi ','ρ':r'\rho ','ε':r'\varepsilon ','ξ':r'\xi ','α':r'\alpha ','β':r'\beta ','φ':r'\varphi ','ω':r'\omega ','τ':r'\tau ','γ':r'\gamma ','ζ':r'\zeta ','𝔼':r'\mathbb{E}','ℛ':r'\mathcal{R}','ℒ':r'\mathcal{L}','ℝ':r'\mathbb{R}','𝒩':r'\mathcal{N}','ℋ':r'\mathcal{H}','ℱ':r'\mathcal{F}','⊘':r'\oslash ','⊙':r'\odot ','⊗':r'\otimes ','‖':r'\Vert ','∇':r'\nabla ','∂':r'\partial ','ℓ':r'\ell ','≥':r'\geq ','≤':r'\leq ','→':r'\to ','←':r'\gets ','∞':r'\infty ','∈':r'\in ','∩':r'\cap ','∅':r'\varnothing ','∑':r'\sum ','∏':r'\prod ','∫':r'\int ','∝':r'\propto ','≠':r'\ne ','≈':r'\approx ','≡':r'\equiv ','∀':r'\forall ','×':r'\times ','·':r'\cdot ','⟨':r'\langle ','⟩':r'\rangle ','−':'-','′':r"^{\prime}",'ᵀ':'^{T}','²':'^{2}','₊':'_{+}'}
def plain(s):
    chars=list(str(s));parts=[];i=0
    while i<len(chars):
        c=chars[i];value=REPL.get(c,c)
        if i+1<len(chars) and chars[i+1] in ['̄','̂','̃']:
            value={'̄':r'\bar','̂':r'\hat','̃':r'\tilde'}[chars[i+1]]+'{'+value+'}';i+=1
        parts.append(value);i+=1
    result=''.join(parts)
    return result
def latex(x):
    if isinstance(x,str):return plain(x)
    if isinstance(x,list):return ''.join(latex(v) for v in x)
    k=x['kind']
    if k=='text':return r'\mathrm{'+x['text'].replace('_',r'\_')+'}' if x.get('upright') else plain(x['text'])
    if k in ['sub','sup','subsup']:
        s='{'+latex(x['base'])+'}'
        if 'sub'in x:s+='_{'+latex(x['sub'])+'}'
        if 'sup'in x:s+='^{'+latex(x['sup'])+'}'
        return s
    if k=='frac':return r'\frac{'+latex(x['num'])+'}{'+latex(x['den'])+'}'
    if k=='radical':return r'\sqrt{'+latex(x['base'])+'}'
    if k=='accent':return {'̂':r'\hat','̄':r'\bar','̃':r'\tilde'}.get(x.get('char','̂'),r'\hat')+'{'+latex(x['base'])+'}'
    if k=='nary':
        s={'∑':r'\sum','∏':r'\prod','∫':r'\int'}.get(x.get('char','∑'),r'\sum')
        if x.get('sub'):s+='_{'+latex(x['sub'])+'}'
        if x.get('sup'):s+='^{'+latex(x['sup'])+'}'
        return s+r'\left('+latex(x['body'])+r'\right)'
    raise ValueError(k)

def main():
    a=argparse.ArgumentParser();a.add_argument('--content',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    d=json.loads(args.content.read_text());out=[];eq=0
    for b in d['blocks']:
        k=b['type']
        if k=='title':out.append('# '+b['text'])
        elif k=='subtitle':out.append(b['text'])
        elif k=='heading':
            if b.get('anchor'):out.append('<a id="'+b['anchor']+'"></a>')
            out.append('#'*(b.get('level',1)+1)+' '+b['text'])
        elif k=='toc_entry':out.append('- ['+b['text']+'](#'+b['anchor']+')')
        elif k=='paragraph':out.append(b['text'])
        elif k=='bullet':out.append('- '+b['text'])
        elif k=='equation':
            eq+=1;formula=CLONES[b['clone']] if 'clone'in b else latex(b['math']);out.append('$$\n'+formula+'\n$$\n\n('+str(eq)+')')
        elif k=='code':out.append('```text\n'+b['text']+'\n```')
        elif k=='table':
            esc=lambda x:str(x).replace('|',r'\|').replace('\n',' ')
            out.append('**'+b.get('caption','')+'**\n\n'+'| '+' | '.join(map(esc,b['headers']))+' |\n'+'| '+' | '.join('---' for _ in b['headers'])+' |\n'+'\n'.join('| '+' | '.join(map(esc,row))+' |' for row in b['rows']))
        elif k=='figure':out.append('!['+b['caption'].replace('[','(').replace(']',')')+'](../figures/'+Path(b['path']).name+')\n\n'+b['caption'])
        elif k=='reference':out.append(b['text']+' [Primary paper]('+b['url']+')')
        elif k=='page_break':pass
        else:raise ValueError(k)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rendered='\n\n'.join(out)
    args.output.write_text('\n'.join(line.rstrip() for line in rendered.splitlines())+'\n')
    print(json.dumps({'markdown':str(args.output),'display_equations':eq,'bytes':args.output.stat().st_size}))

if __name__=='__main__':main()
