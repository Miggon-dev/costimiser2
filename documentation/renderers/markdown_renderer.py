from pathlib import Path

def render_markdown(document, output_path):
    # SOURCE is canonical; render the parsed model to keep the renderer independent.
    lines=[]
    for b in document.blocks:
        if b.kind=='heading': lines.append('#'*b.level+' '+b.value)
        elif b.kind=='paragraph': lines.append(b.value)
        elif b.kind=='code': lines += [f'```{b.language or ""}',b.value,'```']
        elif b.kind=='formula': lines += ['$$',b.value,'$$']
        elif b.kind=='table':
            h=b.value['headers']; lines += ['| '+' | '.join(h)+' |','| '+' | '.join('---' for _ in h)+' |']
            lines += ['| '+' | '.join(r)+' |' for r in b.value['rows']]
        elif b.kind=='bullets': lines += ['- '+x for x in b.value]
        elif b.kind=='numbered': lines += [f'{i}. {x}' for i,x in enumerate(b.value,1)]
        elif b.kind=='note': lines += ['> '+b.value]
        lines.append('')
    p=Path(output_path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8'); return p
