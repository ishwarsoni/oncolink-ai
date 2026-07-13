# 🏥 OncoLink — AI Clinical Intelligence Platform

OncoLink is an AI-powered platform that ingests unstructured oncology documents (PDFs, clinical notes, discharge summaries, radiology reports), extracts structured patient data using LLMs, and generates unified harmonized patient summaries with conflict detection.

---

## ✨ Features

- **Multi-format ingestion** — PDF, DOCX, TXT support
- **AI extraction** — NVIDIA Nemotron-3 Ultra extracts structured clinical fields (diagnosis, stage, biomarkers, medications, ECOG, adverse events)
- **Validation** — Pydantic schema enforcement with field-level error reporting
- **Normalization** — Standardizes abbreviations, capitalization, and terminology
- **Harmonization** — Merges data from multiple documents into one unified patient record
- **Conflict detection** — Flags cross-document discrepancies (age, name, biomarker values)
- **Summary generation** — Text report with conflicts, biomarkers, medications, adverse events
- **Export** — Download summaries as .txt or .pdf

---

## 📁 Project Structure

```
OncoLink/
├── app.py                      # Streamlit UI
├── requirements.txt
├── .env                        # NVIDIA API key (gitignored)
│
├── backend/
│   ├── document_loader.py      # File router (PDF/TXT/DOCX)
│   ├── pdf_reader.py
│   ├── docx_reader.py
│   ├── text_reader.py
│   ├── llm_client.py           # NVIDIA NIM API client
│   ├── prompt_builder.py       # Extraction prompt templates
│   ├── json_parser.py          # Clean LLM output → valid JSON
│   ├── schema.py               # Pydantic models (ClinicalData)
│   ├── models.py               # Schema metadata
│   ├── validator.py            # Pydantic validation
│   ├── normalizer.py           # Value standardization
│   ├── extractor.py            # Pipeline orchestrator
│   ├── harmonizer.py           # Multi-document merge engine
│   ├── merge_utils.py          # Merge helpers
│   ├── conflict_detector.py    # Cross-doc discrepancy detection
│   └── summarizer.py           # Patient summary generator
│
├── data/
│   └── test_docs/              # Sample test documents
├── scripts/                    # Test scripts
├── logs/                       # Application logs
└── assets/
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| UI Framework | Streamlit |
| LLM Inference | NVIDIA NIM (Nemotron-3 Ultra) |
| API Client | OpenAI SDK (OpenAI-compatible endpoint) |
| Data Validation | Pydantic v2 |
| PDF Parsing | PyMuPDF |
| DOCX Parsing | python-docx |
| PDF Generation | fpdf2 |

---

## 🚀 Quick Start

```bash
# 1. Clone and enter directory
git clone <repo-url>
cd OncoLink

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate    # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set NVIDIA API key
# Edit .env and replace 'nvapi-your-key-here'
# Get a free key at: https://build.nvidia.com/

# 5. Launch
streamlit run app.py
```

---

## 📋 Pipeline

```
Upload PDF/TXT/DOCX
       ↓
  Text Extraction
       ↓
  AI Extraction (NVIDIA NIM)
       ↓
  JSON Cleaning & Parsing
       ↓
  Pydantic Validation
       ↓
  Normalization
       ↓
  ┌─────────────────────┐
  │ Per-document results │ ← shown in UI after extraction
  └─────────────────────┘
       ↓
  Harmonization (merge all docs)
       ↓
  Conflict Detection
       ↓
  Patient Summary + Export
```

---

## 📊 Supported Clinical Fields

- Patient identity (name, age, gender)
- Cancer diagnosis, type, stage (TNM)
- Biomarkers (EGFR, ALK, ROS1, PD-L1, HER2, ER, PR, Ki-67, etc.)
- Current medications and previous treatments
- ECOG performance status
- Follow-up plan and next steps
- Adverse events / toxicities

---

## 📝 License

Educational / demonstration purposes.
