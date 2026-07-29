from pathlib import Path
import re, html
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def styles():
 s=getSampleStyleSheet()
 s.add(ParagraphStyle(name='TitleX',parent=s['Title'],fontName='Helvetica-Bold',fontSize=19,leading=24,textColor=colors.HexColor('#17365D'),alignment=TA_CENTER,spaceAfter=8))
 s.add(ParagraphStyle(name='H1X',parent=s['Heading1'],fontName='Helvetica-Bold',fontSize=14,leading=17.5,textColor=colors.HexColor('#17365D'),spaceBefore=8,spaceAfter=5))
 s.add(ParagraphStyle(name='H2X',parent=s['Heading2'],fontName='Helvetica-Bold',fontSize=11.2,leading=14,textColor=colors.HexColor('#2F5597'),spaceBefore=6,spaceAfter=4))
 s.add(ParagraphStyle(name='H3X',parent=s['Heading3'],fontName='Helvetica-Bold',fontSize=9.8,leading=12.5,textColor=colors.HexColor('#4472C4'),spaceBefore=5,spaceAfter=3))
 s.add(ParagraphStyle(name='BodyX',parent=s['BodyText'],fontName='Helvetica',fontSize=9,leading=13,spaceAfter=5))
 s.add(ParagraphStyle(name='SmallX',parent=s['BodyText'],fontName='Helvetica',fontSize=7.8,leading=10))
 s.add(ParagraphStyle(name='CodeX',parent=s['Code'],fontName='Courier',fontSize=7.1,leading=9.2,leftIndent=6,rightIndent=6,backColor=colors.HexColor('#F4F6F8'),borderColor=colors.HexColor('#D9E2F3'),borderWidth=.5,borderPadding=6,spaceBefore=3,spaceAfter=7))
 s.add(ParagraphStyle(name='FormulaX',parent=s['BodyText'],fontName='Helvetica',fontSize=11.5,leading=16,alignment=TA_CENTER,spaceBefore=4,spaceAfter=7))
 s.add(ParagraphStyle(name='NoteBlue',parent=s['BodyText'],fontName='Helvetica-Bold',fontSize=8.9,leading=12.7,backColor=colors.HexColor('#EAF2F8'),borderColor=colors.HexColor('#5B9BD5'),borderWidth=.7,borderPadding=7,spaceBefore=4,spaceAfter=7))
 s.add(ParagraphStyle(name='NoteYellow',parent=s['BodyText'],fontName='Helvetica-Bold',fontSize=8.9,leading=12.7,backColor=colors.HexColor('#FFF2CC'),borderColor=colors.HexColor('#D6B656'),borderWidth=.6,borderPadding=7,spaceBefore=4,spaceAfter=7))
 return s

def _read_group(text, start):
 depth=0
 for i in range(start,len(text)):
  if text[i]=='{': depth+=1
  elif text[i]=='}':
   depth-=1
   if depth==0: return text[start+1:i], i+1
 raise ValueError('Unbalanced braces in formula')

def _replace_fractions(text):
 out=''; i=0
 while i<len(text):
  if text.startswith('\\frac',i):
   j=i+5
   if j<len(text) and text[j]=='{':
    num,j2=_read_group(text,j)
    if j2<len(text) and text[j2]=='{':
     den,j3=_read_group(text,j2)
     out+='('+_replace_fractions(num)+') / ('+_replace_fractions(den)+')'; i=j3; continue
  out+=text[i]; i+=1
 return out

def latex_plain(x):
 x=x.strip().strip('$')
 x=x.replace('\\left','').replace('\\right','').replace('\\quad','   ').replace('\\times',' x ').replace('\\ge',' >= ').replace('\\le',' <= ')
 x=re.sub(r'\\text\{([^{}]*)\}',r'\1',x)
 x=_replace_fractions(x)
 x=re.sub(r'_\{([^{}]+)\}',r'<sub>\1</sub>',x)
 x=re.sub(r'_([A-Za-z0-9]+)',r'<sub>\1</sub>',x)
 x=x.replace('\\','')
 return x

def inline(x):
 # Handle emphasis enclosing a complete math expression.
 outer_bold=x.startswith('**') and x.endswith('**')
 if outer_bold: x=x[2:-2]
 parts=re.split(r'(\$[^$]+\$)',x); out=[]
 for p in parts:
  if p.startswith('$') and p.endswith('$'): out.append(latex_plain(p))
  else:
   q=html.escape(p,quote=False)
   q=re.sub(r'`([^`]+)`',r"<font name='Courier'>\1</font>",q)
   q=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',q)
   q=re.sub(r'\*([^*]+)\*',r'<i>\1</i>',q)
   out.append(q)
 result=''.join(out).replace('—','-').replace('–','-')
 return '<b>'+result+'</b>' if outer_bold else result

def footer(canvas,doc):
 canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#D9E2F3')); canvas.line(18*mm,15*mm,192*mm,15*mm); canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#666666')); canvas.drawString(18*mm,10*mm,doc.title[:90]); canvas.drawRightString(192*mm,10*mm,f'Page {doc.page}'); canvas.restoreState()

def table(block,s):
 h=block['headers']; rows=block['rows']; data=[[Paragraph('<b>'+inline(c)+'</b>',s['SmallX']) for c in h]]
 for r in rows:
  rr=list(r)+['']*(len(h)-len(r)); data.append([Paragraph(inline(c),s['SmallX']) for c in rr[:len(h)]])
 total=174*mm
 if len(h)==2: widths=[.28*total,.72*total]
 elif len(h)==3: widths=[.22*total,.48*total,.30*total]
 elif len(h)==4: widths=[.16*total,.16*total,.18*total,.50*total]
 else: widths=[total/len(h)]*len(h)
 t=Table(data,colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#D9E2F3')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#17365D')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#B4C6E7')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F7F9FC')]),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)])); return t

def render_pdf(document,output_path):
 s=styles(); p=Path(output_path); p.parent.mkdir(parents=True,exist_ok=True); doc=SimpleDocTemplate(str(p),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=16*mm,bottomMargin=20*mm,title=document.title); story=[]
 for b in document.blocks:
  if b.kind=='heading': story.append(Paragraph(inline(b.value),s['TitleX' if b.level==1 else 'H1X' if b.level==2 else 'H2X' if b.level==3 else 'H3X']))
  elif b.kind=='paragraph': story.append(Paragraph(inline(b.value),s['BodyX']))
  elif b.kind=='formula': story.append(Paragraph(latex_plain(b.value),s['FormulaX']))
  elif b.kind=='code': story.append(Paragraph(html.escape(b.value).replace(' ','&#160;').replace('\n','<br/>'),s['CodeX']))
  elif b.kind=='table': story += [table(b.value,s),Spacer(1,5)]
  elif b.kind=='bullets': story += [Paragraph('&#8226;&#160;'+inline(x),s['BodyX']) for x in b.value]
  elif b.kind=='numbered': story += [Paragraph(f'{i}.&#160;&#160;'+inline(x),s['BodyX']) for i,x in enumerate(b.value,1)]
  elif b.kind=='note': story.append(Paragraph(inline(b.value),s['NoteYellow' if 'Important' in b.value else 'NoteBlue']))
 doc.build(story,onFirstPage=footer,onLaterPages=footer); return p
