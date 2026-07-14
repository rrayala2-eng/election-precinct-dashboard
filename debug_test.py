import candidate_lookup as cl
import pdfplumber
import re

pdf_path, is_combined = cl._download_office_pdf(2018, 'Primary', 'ASS')
with pdfplumber.open(pdf_path) as pdf:
    text = '\n'.join(p.extract_text() or '' for p in pdf.pages)

match = re.search(
    r'(?:STATE ASSEMBLY MEMBER|STATE ASSEMBLYMEMBER|MEMBER OF THE STATE ASSEMBLY)\s+DISTRICT\s+15\b(.*?)(?=District\s+\d+\b|\Z)',
    text, re.DOTALL | re.IGNORECASE
)
print(match.group(1) if match else 'NOT FOUND')