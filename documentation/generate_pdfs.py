from pathlib import Path
from registry import DOCUMENTS
from renderers.reportlab_renderer import render_pdf
for d in DOCUMENTS: print(render_pdf(d,Path('generated/pdf')/(d.slug+'.pdf')))
