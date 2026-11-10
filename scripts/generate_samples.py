"""
generate_samples.py - Create synthetic sample documents for local testing.

Run from anywhere:
    python scripts/generate_samples.py

Writes ``data/lab_report.docx`` and ``data/pathology_report.pdf``.
"""

from pathlib import Path

import docx
import fitz

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def create_docx_sample() -> Path:
    """Create a small lab report as a DOCX file."""
    output_path = DATA_DIR / "lab_report.docx"
    document = docx.Document()

    document.add_heading("Lab Report", level=1)

    document.add_heading("Patient Information", level=2)
    document.add_paragraph("Patient Name: Meera Sharma")
    document.add_paragraph("Date of Birth: 15/06/1966")
    document.add_paragraph("MRN: LAB-2024-0183")
    document.add_paragraph("Date Collected: 2024-01-20")

    document.add_heading("Complete Blood Count", level=2)
    document.add_paragraph("Hemoglobin: 12.8 g/dL (Reference: 12.0-15.5)")
    document.add_paragraph("WBC: 6.2 x 10^3/uL (Reference: 4.0-10.0)")
    document.add_paragraph("Platelets: 245 x 10^3/uL (Reference: 150-400)")

    document.add_heading("Comprehensive Metabolic Panel", level=2)
    document.add_paragraph("All values within normal limits.")

    document.add_heading("Tumor Markers", level=2)
    document.add_paragraph("CA 15-3: 32 U/mL (Reference: <30 U/mL)")
    document.add_paragraph("Note: Slightly elevated CA 15-3.")

    document.add_heading("Impression", level=2)
    document.add_paragraph(
        "Normal organ function. Tumor marker slightly elevated, "
        "consistent with known breast malignancy."
    )

    document.save(output_path)
    print(f"  ✅ Created: {output_path}")
    return output_path


def create_pdf_sample() -> Path:
    """Create a two-page pathology report as a PDF file."""
    output_path = DATA_DIR / "pathology_report.pdf"
    pdf_document = fitz.open()

    page1_text = """
CITY PATHOLOGY LAB
    
Pathology Report
    
Patient: Meera Sharma
DOB: 15/06/1966
MRN: PATH-2024-0183
    
Specimen: Left breast core needle biopsy
Date Collected: 2024-01-15
Date Reported: 2024-01-18
    
GROSS DESCRIPTION:
Three core biopsies measuring 1.5 cm, 1.8 cm, and 1.2 cm in length.
All cores submitted for processing.
    
MICROSCOPIC DESCRIPTION:
Sections show invasive ductal carcinoma (IDC), Nottingham histologic
grade 2 (moderately differentiated).
    
Tubule formation score: 2
Nuclear pleomorphism score: 2
Mitotic count score: 1
    
Total Nottingham score: 5/9
"""
    page1 = pdf_document.new_page()
    page1.insert_text(fitz.Point(50, 100), page1_text, fontsize=10)

    page2_text = """
IMMUNOHISTOCHEMISTRY RESULTS:
    
ER: Positive (90%, strong intensity)
PR: Positive (70%, moderate intensity)
HER2: Negative (1+ by IHC)
Ki-67: 25%
    
DIAGNOSIS:
Invasive ductal carcinoma of the left breast, Grade 2.
    
Hormone Receptor Status:
    ER Positive
    PR Positive
    HER2 Negative (Triple Negative status: Negative)
    
Pathologic Stage: pT2 (tumor size estimated 3.2 cm on gross)
    
COMMENTS:
All immunohistochemical stains have been reviewed.
Appropriate controls show expected results.
"""
    page2 = pdf_document.new_page()
    page2.insert_text(fitz.Point(50, 100), page2_text, fontsize=10)

    pdf_document.save(output_path)
    pdf_document.close()
    print(f"  ✅ Created: {output_path}")
    return output_path


def main() -> None:
    """Create every sample document."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Creating sample documents for OncoLink testing...")
    print()

    create_docx_sample()
    create_pdf_sample()

    print()
    print("All sample documents created successfully!")
    print("You can find them in the data/ folder.")


if __name__ == "__main__":
    main()
