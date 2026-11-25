from pathlib import Path
from fpdf import FPDF

folder_path = Path(__file__).resolve().parent
output_pdf = folder_path / "report.pdf"
output_txt = folder_path / "report.txt"

title = "Project Summary"
body = "This PDF was generated with Python and saved directly into the folder."

# 1. Create PDF file
pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font("Helvetica", style="B", size=16)
pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", size=12)
pdf.ln(4)
pdf.multi_cell(0, 8, body)
pdf.output(str(output_pdf))

# 2. Create raw text file so you can view the text directly in VS Code
raw_text = f"{title}\n{'-' * len(title)}\n{body}\n"
output_txt.write_text(raw_text, encoding="utf-8")

print(f"PDF saved to: {output_pdf}")
print(f"Raw text saved to: {output_txt}")