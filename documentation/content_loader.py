from __future__ import annotations
import re
from document_model import Block, Document

def parse_markdown(slug, source):
    lines=source.splitlines(); blocks=[]; title=slug.replace('_',' ').title(); i=0
    while i<len(lines):
        s=lines[i].strip()
        if not s: i+=1; continue
        if s in {'---','***','___'}: i+=1; continue
        if s.startswith('```'):
            lang=s[3:].strip(); buf=[]; i+=1
            while i<len(lines) and not lines[i].strip().startswith('```'): buf.append(lines[i]); i+=1
            i+=1; blocks.append(Block('code','\n'.join(buf),language=lang)); continue
        if s=='$$':
            buf=[]; i+=1
            while i<len(lines) and lines[i].strip()!='$$': buf.append(lines[i]); i+=1
            i+=1; blocks.append(Block('formula',' '.join(x.strip() for x in buf))); continue
        m=re.match(r'^(#{1,6})\s+(.+)$',s)
        if m:
            level=len(m.group(1)); txt=m.group(2); title=txt if level==1 else title; blocks.append(Block('heading',txt,level=level)); i+=1; continue
        if s.startswith('|') and i+1<len(lines) and re.match(r'^\|?[\s:|-]+\|?$',lines[i+1].strip()):
            def cells(r): return [c.strip() for c in r.strip().strip('|').split('|')]
            headers=cells(s); i+=2; rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'): rows.append(cells(lines[i])); i+=1
            blocks.append(Block('table',{'headers':headers,'rows':rows})); continue
        if re.match(r'^[-*]\s+',s):
            items=[]
            while i<len(lines) and re.match(r'^[-*]\s+',lines[i].strip()): items.append(re.sub(r'^[-*]\s+','',lines[i].strip())); i+=1
            blocks.append(Block('bullets',items)); continue
        if re.match(r'^\d+\.\s+',s):
            items=[]
            while i<len(lines) and re.match(r'^\d+\.\s+',lines[i].strip()): items.append(re.sub(r'^\d+\.\s+','',lines[i].strip())); i+=1
            blocks.append(Block('numbered',items)); continue
        if s.startswith('>'):
            q=[]
            while i<len(lines) and lines[i].strip().startswith('>'): q.append(lines[i].strip()[1:].strip()); i+=1
            blocks.append(Block('note',' '.join(q))); continue
        if s.startswith('**Version:**'):
            meta=[s]; i+=1
            while i<len(lines) and lines[i].strip(): meta.append(lines[i].strip()); i+=1
            blocks.append(Block('paragraph','<br/>'.join(meta))); continue
        para=[s]; i+=1
        while i<len(lines):
            n=lines[i].strip()
            if not n or n.startswith(('#','```','$$','|','>')) or re.match(r'^[-*]\s+',n) or re.match(r'^\d+\.\s+',n): break
            para.append(n); i+=1
        blocks.append(Block('paragraph',' '.join(para)))
    return Document(slug,title,blocks)
