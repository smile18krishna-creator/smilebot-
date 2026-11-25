import fitz  # PyMuPDF

PAGES = [
    (
        "PAGE 1: WORK HOURS & HYBRID POLICY\n\n"
        "Acme Tech follows a hybrid work model. All full-time employees are expected to work "
        "from the office at least 3 days per week, with Tuesday and Thursday designated as mandatory "
        "anchor days. Core collaboration hours are 10:00 AM to 4:00 PM IST, Monday through Friday. "
        "New hires are eligible for a one-time home office setup stipend of up to Rs. 15,000 for an "
        "ergonomic chair, external monitor, or UPS backup. GST tax invoices must be uploaded to the "
        "HR portal within 60 days of the employee's joining date."
    ),
    (
        "PAGE 2: PAID TIME OFF (PTO) & SICK LEAVE\n\n"
        "Full-time employees receive 18 days of annual paid leave per calendar year, accrued at "
        "1.5 days per month. Up to 6 unused annual leave days may carry over into the next calendar "
        "year; any remaining balance above 6 days expires on December 31. Employees also receive "
        "8 paid sick days per year. A doctor's medical certificate is required only if sick leave "
        "exceeds 3 consecutive working days. Any PTO request of 3 or more consecutive days must be "
        "submitted for manager approval at least 10 working days in advance."
    ),
    (
        "PAGE 3: EXPENSES & CLIENT TRAVEL\n\n"
        "During approved client travel, meal expenses are reimbursed up to a maximum of Rs. 1,200 "
        "per day. Alcohol and tobacco purchases are strictly excluded from reimbursement under all "
        "circumstances. Remote and hybrid staff may claim a monthly home broadband allowance of up "
        "to Rs. 1,000 per month, submitted once per quarter through the finance portal. All expense "
        "claims must be filed within 30 calendar days of the transaction date; claims submitted "
        "after 30 days are automatically rejected."
    ),
    (
        "PAGE 4: LEARNING STIPEND & COMPANY HARDWARE\n\n"
        "Acme Tech provides an annual Learning and Certification Stipend of up to Rs. 25,000 per "
        "year for technical courses, books, or cloud certifications. Employees must obtain written "
        "manager pre-approval before purchasing a course and submit a completion or passing "
        "certificate for reimbursement. Upon resignation or termination, company-issued laptops and "
        "accessories must be returned to the IT desk within 48 hours of the employee's last working "
        "day. Unreturned equipment will be deducted from the final settlement."
    ),
]


def create_policy_pdf(output_path: str = "company_policy.pdf"):
    doc = fitz.open()
    for page_text in PAGES:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 545, 790)
        page.insert_textbox(rect, page_text, fontsize=11, fontname="helv")
    doc.save(output_path)
    doc.close()
    print(f"Created {output_path} ({len(PAGES)} pages)")


if __name__ == "__main__":
    create_policy_pdf()
