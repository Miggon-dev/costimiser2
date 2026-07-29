# Unified document generators v2

The five modules in `content/` are the canonical source for both Markdown and PDF. They were rebuilt from the final reference PDFs.

```bash
pip install -r requirements.txt
python generate_all.py
```

Outputs are written to `generated/markdown/` and `generated/pdf/`.

The ReportLab renderer includes explicit conversion of the limited LaTeX notation used by the scoring documents, including subscripts, fractions, comparison operators, and multiplication.
