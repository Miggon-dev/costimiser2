import subprocess,sys
subprocess.run([sys.executable,'generate_markdown.py'],check=True)
subprocess.run([sys.executable,'generate_pdfs.py'],check=True)
