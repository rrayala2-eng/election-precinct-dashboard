import candidate_lookup as cl
import pdfplumber

write_in_path = cl._download_write_in_pdf(2018, 'Primary')
print("Write-in PDF found:", write_in_path is not None)

if write_in_path:
    with pdfplumber.open(write_in_path) as pdf:
        text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
    import re
    match = re.search(
        r'(?:STATE ASSEMBLY MEMBER|STATE ASSEMBLYMEMBER|MEMBER OF THE STATE ASSEMBLY)\s+DISTRICT\s+15\b(.*?)(?=District\s+\d+\b|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    print("AD15 write-in block:", match.group(1) if match else "NOT FOUND IN TEXT")