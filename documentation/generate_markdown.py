from pathlib import Path
from registry import DOCUMENTS
from renderers.markdown_renderer import render_markdown
for d in DOCUMENTS: print(render_markdown(d,Path('generated/markdown')/(d.slug+'.md')))
