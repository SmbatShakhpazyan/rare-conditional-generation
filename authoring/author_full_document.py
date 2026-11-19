"""English research documents using bundled python-docx and native Word math.

All displayed equations are native OMML. Seven established study equations
are copied without mathematical alteration from the verified prior manuscript.
The remaining equations use explicit OMML structures, not a LaTeX parser.
"""
from __future__ import annotations
import argparse
import copy
import json
import re
import zipfile
from pathlib import Path
from lxml import etree
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT

BASE = Path(__file__).resolve().parents[2]
OLD = BASE / 'outputs/generative_research_2026_11_19/research_paper_hy.docx'
BODY = 'Liberation Serif'
HEAD = 'Liberation Serif'
MATH = 'DejaVu Math TeX Gyre'
MNS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
CLONE_KEYS = ['contrast', 'cluster', 'scale', 'witness', 'candidate_loss',
              'batch_contrast', 'kid']
MATH_FILE = Path(__file__).resolve().parent / 'native_math_library.xml'
if MATH_FILE.exists():
    library_xml = etree.fromstring(MATH_FILE.read_bytes())
    MATH_LIBRARY = {node.get('key'):node[0] for node in library_xml}
else:
    with zipfile.ZipFile(OLD) as archive:
        xml = etree.fromstring(archive.read('word/document.xml'))
        original_math = xml.findall('.//{' + MNS + '}oMath')
    assert len(original_math) == len(CLONE_KEYS)
    MATH_LIBRARY = dict(zip(CLONE_KEYS, original_math))
assert set(MATH_LIBRARY) == set(CLONE_KEYS)

def font(run, name=BODY, size=None, bold=None, italic=None):
    run.font.name = name
    run.font.color.rgb = RGBColor(0, 0, 0)
    if size is not None: run.font.size = Pt(size)
    if bold is not None: run.bold = bold
    if italic is not None: run.italic = italic
    props = run._element.get_or_add_rPr()
    fonts = props.find(qn('w:rFonts'))
    if fonts is None:
        fonts = OxmlElement('w:rFonts'); props.insert(0, fonts)
    for key in ['ascii', 'hAnsi', 'eastAsia', 'cs']:
        fonts.set(qn('w:' + key), name)
    for key in ['asciiTheme', 'hAnsiTheme', 'eastAsiaTheme', 'csTheme']:
        fonts.attrib.pop(qn('w:' + key), None)
    return run

def mrun(text, upright=False):
    node = OxmlElement('m:r')
    if upright:
        props = OxmlElement('m:rPr')
        style = OxmlElement('m:sty'); style.set(qn('m:val'), 'p')
        props.append(style); node.append(props)
    props = OxmlElement('w:rPr'); fonts = OxmlElement('w:rFonts')
    for key in ['ascii', 'hAnsi', 'cs']: fonts.set(qn('w:' + key), MATH)
    props.append(fonts)
    size = OxmlElement('w:sz'); size.set(qn('w:val'), '22'); props.append(size)
    node.append(props)
    value = OxmlElement('m:t'); value.text = str(text)
    value.set(qn('xml:space'), 'preserve')
    node.append(value)
    return node

def append_math(parent, value):
    if isinstance(value, str):
        parent.append(mrun(value)); return
    if isinstance(value, list):
        for element in value: append_math(parent, element)
        return
    kind = value['kind']
    if kind == 'text':
        parent.append(mrun(value['text'], value.get('upright', False))); return
    if kind in ['sub', 'sup', 'subsup']:
        tags = {'sub': 'sSub', 'sup': 'sSup', 'subsup': 'sSubSup'}
        node = OxmlElement('m:' + tags[kind])
        base = OxmlElement('m:e'); append_math(base, value['base']); node.append(base)
        for tag in ['sub', 'sup']:
            if tag in value:
                child = OxmlElement('m:' + tag); append_math(child, value[tag]); node.append(child)
        parent.append(node); return
    if kind == 'frac':
        node = OxmlElement('m:f')
        for tag, key in [('num', 'num'), ('den', 'den')]:
            child = OxmlElement('m:' + tag); append_math(child, value[key]); node.append(child)
        parent.append(node); return
    if kind == 'nary':
        node = OxmlElement('m:nary'); props = OxmlElement('m:naryPr')
        char = OxmlElement('m:chr'); char.set(qn('m:val'), value.get('char', '∑'))
        props.append(char)
        location = OxmlElement('m:limLoc'); location.set(qn('m:val'), 'undOvr')
        props.append(location)
        for tag in ['sub', 'sup']:
            if not value.get(tag):
                hide = OxmlElement('m:' + tag + 'Hide'); hide.set(qn('m:val'), '1')
                props.append(hide)
        node.append(props)
        for tag in ['sub', 'sup']:
            child = OxmlElement('m:' + tag); append_math(child, value.get(tag, '')); node.append(child)
        body = OxmlElement('m:e'); append_math(body, value['body']); node.append(body)
        parent.append(node); return
    if kind == 'accent':
        node = OxmlElement('m:acc'); props = OxmlElement('m:accPr')
        char = OxmlElement('m:chr'); char.set(qn('m:val'), value.get('char', '̂'))
        props.append(char); node.append(props)
        body = OxmlElement('m:e'); append_math(body, value['base']); node.append(body)
        parent.append(node); return
    if kind == 'radical':
        node = OxmlElement('m:rad'); props = OxmlElement('m:radPr')
        hide = OxmlElement('m:degHide'); hide.set(qn('m:val'), '1'); props.append(hide)
        node.append(props); node.append(OxmlElement('m:deg'))
        body = OxmlElement('m:e'); append_math(body, value['base']); node.append(body)
        parent.append(node); return
    raise ValueError('Unsupported native math structure: ' + kind)

def equation(block):
    if 'clone' in block:
        node = copy.deepcopy(MATH_LIBRARY[block['clone']])
    else:
        node = OxmlElement('m:oMath'); append_math(node, block['math'])
    if block.get('math_size'):
        for size in node.findall('.//' + qn('w:sz')):
            size.set(qn('w:val'), str(round(2 * block['math_size'])))
    return node

def inline(p, text, size=None):
    # Small explicit native inline-math vocabulary for the verified manuscript.
    terms = {
        'D_tr': {'kind':'sub','base':'D','sub':'tr'},
        'D_val': {'kind':'sub','base':'D','sub':'val'},
        'D_te': {'kind':'sub','base':'D','sub':'te'},
        'T_g': {'kind':'sub','base':'T','sub':'g'},
        'θ̄_η': {'kind':'sub','base':{'kind':'accent','base':'θ','char':'̄'},'sub':'η'},
        'δ̂_t': {'kind':'sub','base':{'kind':'accent','base':'δ','char':'̂'},'sub':'t'},
    }
    for base, sub in [('a','g'),('b','g'),('n','g'),('x','i'),('f','i'),('G','i'),('Δ','B'),('x','1'),('x','m'),('y','1'),('y','n')]:
        terms[base+'_'+sub] = {'kind':'sub','base':base,'sub':sub}
    for base, sub in [('Q','θ'),('p','θ'),('q','φ'),('μ','φ'),('σ','φ'),('G','θ'),('D','ω'),('f','θ'),('x','a'),('x','b'),('x','0'),('x','1'),('x','2'),('x','t'),('x','s'),('∇','x'),('m','t'),('s','j'),('u','j'),('μ','g')]:
        terms[base+sub] = {'kind':'sub','base':base,'sub':sub}
    terms['P*'] = {'kind':'sup','base':'P','sup':'*'}
    terms['x̂s'] = {'kind':'sub','base':{'kind':'accent','base':'x','char':'̂'},'sub':'s'}
    for power in ['-3','-4','-6','-8']:
        terms['10^'+power] = {'kind':'sup','base':'10','sup':power.replace('-','−')}
    pattern = r'(?<!\w)('+'|'.join(re.escape(x) for x in sorted(terms,key=len,reverse=True))+r')(?!\w)'
    for piece in re.split(r'(\*\*[^*]+\*\*)', str(text)):
        if piece:
            bold = piece.startswith('**')
            for part in re.split(pattern, piece[2:-2] if bold else piece):
                if part in terms:
                    node=OxmlElement('m:oMath'); append_math(node,terms[part]); p._p.append(node)
                elif part:
                    font(p.add_run(part),size=size,bold=True if bold else None)

def link(p, label, url):
    relation = p.part.relate_to(url, RT.HYPERLINK, is_external=True)
    node = OxmlElement('w:hyperlink'); node.set(qn('r:id'), relation)
    run = OxmlElement('w:r'); props = OxmlElement('w:rPr')
    color = OxmlElement('w:color'); color.set(qn('w:val'), '17456B'); props.append(color)
    run.append(props); text = OxmlElement('w:t'); text.text = label; run.append(text)
    node.append(run); p._p.append(node)

def table(doc, block):
    headers, rows = block['headers'], block['rows']
    if block.get('caption'):
        p = doc.add_paragraph(style='Caption'); p.paragraph_format.keep_with_next = True
        inline(p, block['caption'], 9.7)
    t = doc.add_table(rows=1, cols=len(headers)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    widths = block.get('widths', [6.8 / len(headers)] * len(headers))
    assert abs(sum(widths) - 6.8) < .3 and len(widths) == len(headers)
    for col, width in zip(t.columns, widths): col.width = Inches(width)
    props = t._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        b = OxmlElement('w:' + edge)
        for key, value in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
            b.set(qn('w:' + key), value)
        borders.append(b)
    props.append(borders)
    margins = OxmlElement('w:tblCellMar')
    for edge, amount in [('top', '90'), ('bottom', '90'), ('left', '90'), ('right', '90')]:
        m = OxmlElement('w:' + edge); m.set(qn('w:w'), amount); m.set(qn('w:type'), 'dxa')
        margins.append(m)
    props.append(margins)
    for i, values in enumerate([headers] + rows):
        cells = t.rows[0].cells if i == 0 else t.add_row().cells
        trpr = t.rows[i]._tr.get_or_add_trPr(); trpr.append(OxmlElement('w:cantSplit'))
        if i == 0: trpr.append(OxmlElement('w:tblHeader'))
        for j, (cell, value) in enumerate(zip(cells, values)):
            cell.width = Inches(widths[j]); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]; fmt = p.paragraph_format
            fmt.space_after = Pt(0); fmt.space_before = Pt(0); fmt.line_spacing = 1.05
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in block.get('left_columns', [0]) else WD_ALIGN_PARAGRAPH.CENTER
            inline(p, value, block.get('size', 9.3))
            if i == 0:
                for run in p.runs: run.bold = True
                shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'E8EDF2')
                cell._tc.get_or_add_tcPr().append(shade)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return t

def setup(title):
    doc = Document(); sec = doc.sections[0]
    sec.page_width = Inches(8.5); sec.page_height = Inches(11)
    sec.left_margin = sec.right_margin = Inches(.8)
    sec.top_margin = Inches(.7); sec.bottom_margin = Inches(.7)
    sec.footer_distance = Inches(.3)
    for name in ['Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Heading 3', 'Heading 4', 'Caption']:
        s = doc.styles[name]; s.font.name = BODY; s.font.color.rgb = RGBColor(0, 0, 0)
        s.paragraph_format.widow_control = True
        if name != 'Normal': s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        ppr = s.element.find(qn('w:pPr'))
        if ppr is not None:
            for border in list(ppr.findall(qn('w:pBdr'))): ppr.remove(border)
        rpr = s.element.get_or_add_rPr(); fonts = rpr.find(qn('w:rFonts'))
        if fonts is None:
            fonts = OxmlElement('w:rFonts'); rpr.insert(0, fonts)
        for key in ['ascii', 'hAnsi', 'eastAsia', 'cs']: fonts.set(qn('w:' + key), BODY)
        for key in ['asciiTheme', 'hAnsiTheme', 'eastAsiaTheme', 'csTheme']:
            fonts.attrib.pop(qn('w:' + key), None)
    n = doc.styles['Normal']; n.font.size = Pt(11)
    n.paragraph_format.line_spacing = 1.1; n.paragraph_format.space_after = Pt(6)
    n.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for name, size, before, after in [('Heading 1', 17, 14, 9), ('Heading 2', 14, 13, 7), ('Heading 3', 12, 10, 5), ('Heading 4', 11, 8, 4)]:
        s = doc.styles[name]; s.font.size = Pt(size); s.font.bold = True
        s.paragraph_format.space_before = Pt(before); s.paragraph_format.space_after = Pt(after)
        s.paragraph_format.keep_with_next = True
    doc.styles['Title'].font.size = Pt(20); doc.styles['Title'].font.bold = True
    doc.styles['Title'].paragraph_format.space_after = Pt(8)
    doc.styles['Subtitle'].font.size = Pt(11)
    doc.styles['Subtitle'].paragraph_format.space_after = Pt(7)
    doc.styles['Caption'].font.size = Pt(9.7)
    doc.styles['Caption'].paragraph_format.space_after = Pt(6)
    props = OxmlElement('m:mathPr'); mf = OxmlElement('m:mathFont'); mf.set(qn('m:val'), MATH)
    props.append(mf); doc.settings.element.append(props)
    footer = sec.footer.paragraphs[0]; footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
    doc.core_properties.title = title; doc.core_properties.author = 'Smbat Shakhpazyan'
    doc.core_properties.subject = 'Conditional generative modeling and rare attribute combinations'
    return doc

def build(source, output):
    content = json.loads(Path(source).read_text()); doc = setup(content['title'])
    equation_number = 0
    for block in content['blocks']:
        typ = block['type']
        if typ == 'title': doc.add_paragraph(block['text'], 'Title')
        elif typ == 'subtitle': doc.add_paragraph(block['text'], 'Subtitle')
        elif typ == 'heading':
            if block.get('page_break'): doc.add_page_break()
            p=doc.add_heading(block['text'], level=block.get('level', 1))
            if block.get('anchor'):
                identifier=str(len(doc.paragraphs))
                begin=OxmlElement('w:bookmarkStart');begin.set(qn('w:id'),identifier);begin.set(qn('w:name'),block['anchor'])
                end=OxmlElement('w:bookmarkEnd');end.set(qn('w:id'),identifier)
                p._p.insert(0,begin);p._p.append(end)
        elif typ == 'toc_entry':
            p=doc.add_paragraph();p.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after=Pt(8)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(6.65),WD_TAB_ALIGNMENT.RIGHT)
            node=OxmlElement('w:hyperlink');node.set(qn('w:anchor'),block['anchor'])
            run=OxmlElement('w:r');value=OxmlElement('w:t');value.text=block['text'];run.append(value);node.append(run);p._p.append(node)
            font(p.add_run('\t'+str(block.get('page',''))),size=11)
        elif typ in ['paragraph', 'bullet']:
            p = doc.add_paragraph(style='List Bullet' if typ == 'bullet' else 'Normal')
            inline(p, block['text'], block.get('size'))
            if block.get('keep'): p.paragraph_format.keep_with_next = True
        elif typ == 'code':
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.0; p.paragraph_format.keep_together = True
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            font(p.add_run(block['text']), 'Liberation Mono', 8.6)
        elif typ == 'equation':
            equation_number += 1
            p = doc.add_paragraph(); fmt = p.paragraph_format
            fmt.space_before = Pt(5); fmt.space_after = Pt(7); fmt.keep_together = True
            fmt.alignment = WD_ALIGN_PARAGRAPH.LEFT
            fmt.tab_stops.add_tab_stop(Inches(3.2), WD_TAB_ALIGNMENT.CENTER)
            fmt.tab_stops.add_tab_stop(Inches(6.75), WD_TAB_ALIGNMENT.RIGHT)
            p.add_run('\t'); p._p.append(equation(block))
            font(p.add_run('\t(' + str(equation_number) + ')'), size=10)
        elif typ == 'table': table(doc, block)
        elif typ == 'figure':
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            figure=Path(block['path'])
            if not figure.is_absolute():figure=Path(source).resolve().parent/figure
            p.add_run().add_picture(str(figure), width=Inches(block.get('width', 6.6)))
            cap = doc.add_paragraph(style='Caption'); inline(cap, block['caption'], 9.7)
        elif typ == 'page_break': doc.add_page_break()
        elif typ == 'reference':
            p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.left_indent = Inches(.2)
            p.paragraph_format.first_line_indent = Inches(-.2)
            inline(p, block['text'], 9.6)
            if block.get('url'): p.add_run(' '); link(p, 'Primary paper', block['url'])
        else: raise ValueError(typ)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); doc.save(output)
    return {'output': str(output), 'equations': equation_number, 'blocks': len(content['blocks'])}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--content', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    print(json.dumps(build(args.content, args.output)))
