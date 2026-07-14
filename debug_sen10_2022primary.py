import candidate_lookup as cl
import pdfplumber

pdf_path, is_combined = cl._download_office_pdf(2022, 'Primary', 'SEN')

with pdfplumber.open(pdf_path) as pdf:
    text_default = '\n'.join(p.extract_text() or '' for p in pdf.pages)
    text_layout = '\n'.join(p.extract_text(layout=True) or '' for p in pdf.pages)

def show_context(text, label):
    idx = text.find("Wahab")
    print(f"=== {label}: 'Wahab' found at index {idx} ===")
    if idx >= 0:
        print(text[max(0, idx-300):idx+200])
    print()

show_context(text_default, "DEFAULT extraction")
show_context(text_layout, "LAYOUT=TRUE extraction")