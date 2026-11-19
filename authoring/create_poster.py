"""Precise English LinkedIn artwork from saved benchmark means."""
from pathlib import Path
from html import escape
import json

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'poster';OUT.mkdir(exist_ok=True)
URL='https://github.com/SmbatShakhpazyan/rare-conditional-generation'

def t(x,y,v,size=24,color='#132d3c',weight='normal',anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(v)}</text>'
def rect(x,y,w,h,color,rx=0):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" rx="{rx}"/>'

def main():
    bg='#f7f4ed';ink='#132d3c';teal='#007b83';rust='#b95335';mute='#556775'
    s=f'<svg xmlns="http://www.w3.org/2000/svg" width="2160" height="2160" viewBox="0 0 1080 1080"><rect width="1080" height="1080" fill="{bg}"/><g font-family="Liberation Sans,Arial,sans-serif">'
    s+=t(76,65,'RESEARCH REPORT  /  METHODS + CODE + RESULTS',20,teal,'bold')
    s+=t(72,153,'Rare conditional',70,ink,'bold')+t(72,231,'image generation',70,ink,'bold')
    s+=t(76,281,'A leakage-audited conditional GAN study',29,mute)
    s+=t(76,365,'The tested correction did not',37,ink,'bold')+t(76,412,'beat the baseline on average.',37,ink,'bold')
    s+=t(76,466,'Rare-condition KID x 1,000  |  lower is better',24,mute)
    # Bars start at zero and use the same 0..250 scale.
    left,width,maximum=76,760,250
    for label,value,color,y in [('Balanced baseline B',166.663322,teal,535),('Correction C',185.543785,rust,666)]:
        s+=t(left,y,label,27,ink,'bold')
        s+=rect(left,y+22,width,47,'#e5e6df',4)
        s+=rect(left,y+22,width*value/maximum,47,color,4)
        s+=t(965,y+61,f'{value:.3f}',43,color,'bold','end')
    s+=t(76,805,'10 validation checkpoints',27,ink,'bold')+t(76,845,'3 fresh paired test seeds',27,ink,'bold')
    s+=t(76,892,'1,500 updates  /  kappa = 0 selected  /  4 of 6 generator archives',20,mute)
    s+=rect(76,931,928,2,'#c9d3d4')
    s+=t(76,972,'Smbat Shakhpazyan',25,ink,'bold')+t(76,1008,'Paper, source code and saved benchmark records on GitHub',21,mute)
    s+=t(76,1040,'github.com/SmbatShakhpazyan/rare-conditional-generation',21,teal)
    s+='</g></svg>'
    (OUT/'linkedin_research_poster.svg').write_text(s)
    (OUT/'linkedin_post_en.txt').write_text(f'''Can a feature-interaction correction help a rare condition in image generation?

I am sharing a detailed research report on a leakage-audited conditional GAN study, with mathematical derivations, architecture and training details, all validation choices, and the complete saved comparison.

The result is negative. Across three fresh paired test seeds, mean rare-condition KID x 1,000 was 166.663 for the balanced baseline and 185.544 for the proposed correction. Lower is better. Validation selected kappa=0, so the selected uncertainty threshold was inactive.

The repository includes the research paper, visual explanations, the executed study code, original reference implementations for the reviewed model families, links and source audits for related papers, aggregate benchmark records, and the four recovered generator archives.

The reference implementations are not additional completed benchmark arms. Two complete generator archives and supporting reference/teacher/split inputs remain unavailable, so a full six-model numerical replay is not claimed.

Research report and code: {URL}

#GenerativeAI #MachineLearning #Research #Reproducibility #GANs
''')
    (OUT/'linkedin_post_hy.txt').write_text(f'''Կարո՞ղ է հատկանիշների փոխազդեցության շտկումը բարելավել հազվադեպ պայմանի համար պատկերի գեներացումը։

Հրապարակում եմ մանրամասն հետազոտական աշխատանք՝ conditional GAN-ի, տվյալների բաժանման աուդիտի, մաթեմատիկական ձևակերպումների և բոլոր validation/test արդյունքների ներկայացմամբ։

Այս փորձի արդյունքը բացասական է․ երեք նոր զուգակցված seed-երի միջին rare-condition KID × 1000-ը baseline-ի համար 166.663 է, շտկված մոդելի համար՝ 185.544։ Ավելի փոքր արժեքն ավելի լավ է։ Ընտրվել է κ=0, ուստի ընտրված տարբերակում անորոշության շեմը չի գործել։

Փաթեթում կան հոդվածը, տեսապատկերները, իրական փորձի կոդը, ուսումնասիրված ընտանիքների հղումային իրականացումները, հոդվածների հղումները, ամփոփ չափումները և վերականգնված չորս գեներատորների կշիռները։ Հղումային մոդելները լրացուցիչ benchmark փորձեր չեն։ Երկու գեներատորի ամբողջական ֆայլերը և որոշ օժանդակ մուտքեր դեռ բացակայում են։

Հոդվածն ու կոդը՝ {URL}

#GenerativeAI #MachineLearning #Research #Reproducibility
''')
    (OUT/'alt_text_en.txt').write_text('Research report poster for rare conditional image generation. A bar chart shows mean rare-condition KID times 1,000: balanced baseline B 166.663, correction C 185.544; lower is better. The correction did not beat the baseline on average across three paired test seeds. Ten validation checkpoints, 1,500 training updates and selected kappa zero are disclosed. Four of six generator archives were recovered. The poster links to the public research repository.\n')
    (OUT/'poster_metadata.json').write_text(json.dumps(dict(canvas='1080 x 1080 logical pixels',export='2160 x 2160 PNG',language='English',source='Saved three-fixed-seed means',new_quality_metrics=False,linkedin_published=False,repository_url=URL),indent=2)+'\n')
    print('Created SVG poster and English Armenian caption drafts')

if __name__=='__main__':main()
