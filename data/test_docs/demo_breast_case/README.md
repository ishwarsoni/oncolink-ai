# Demo breast cancer case (synthetic text documents)

Five plain-text oncology documents for one fictional patient, intended for
manual UI testing of multi-document upload, extraction, harmonization,
conflict review, provenance, exports, and evidence-grounded chat.

**All content is synthetic.** Names, identifiers, dates, and clinical details
are invented for software testing and must not be treated as real patient data
or medical advice. Do not upload real patient records to this educational
application.

## Documents

| File | Document type | Date |
|---|---|---|
| `01_oncology_consultation.txt` | Medical oncology consultation | 03-Feb-2026 |
| `02_pathology_report.txt` | Surgical pathology report | 10-Mar-2026 |
| `03_chemotherapy_start_note.txt` | Oncology progress note, cycle 1 | 02-Apr-2026 |
| `04_cycle3_progress_note.txt` | Oncology progress note, cycle 3 | 14-May-2026 |
| `05_tumour_board_summary.txt` | Multidisciplinary tumour board | 21-May-2026 |

The case follows one patient through diagnosis, surgery, adjuvant
chemotherapy, and radiotherapy/endocrine planning. Headings use the section
names the chunker recognizes (`DIAGNOSIS`, `BIOMARKERS`, `CURRENT MEDICATIONS`,
`ADVERSE EVENTS`, `FOLLOW-UP PLAN`, `NEXT STEPS`) so retrieval citations
include useful section context.

## Intentional discrepancies

These are deliberate to exercise the conflict and provenance UI. The app must
show both values rather than silently choosing one.

- **Age:** 52 in most documents, 54 in the chemotherapy start note.
- **Stage:** Stage IIB in the pathology and later notes, Stage IIIA in the
  chemotherapy start note; the tumour board explicitly discusses the
  discrepancy.
- **ER:** 90% in the consultation biopsy, 80% in the pathology report and
  tumour board.
- **Ki-67:** 25% in the consultation, 30% in the pathology report and tumour
  board.
- **ECOG:** 0 in the consultation and cycle 3 note, 1 in the chemotherapy start
  note and tumour board.
- **HER2 wording:** every note documents HER2-negative disease, but some record
  the result as `negative` and others as `IHC 1+`. The conflict view may flag
  this as a value difference; it is a wording difference, not a change in
  clinical meaning, and is a useful example of why conflicts need human review.
- **Medication and treatment lists:** change over time across documents, so
  harmonization may report differences as well as chronological updates.

## Suggested chat questions

- What is the patient's diagnosis and pathologic stage?
- What biomarkers are documented, and do the documents agree?
- Summarize the surgery and pathology findings.
- Which chemotherapy regimen was given, and how many cycles were completed?
- What adverse events are documented?
- What is the planned radiotherapy and endocrine treatment?
- What information differs between the uploaded documents?

## Using the documents

1. Start the app with `streamlit run app.py`.
2. In **Uploaded Documents**, upload all five `.txt` files and select
   **Process documents**.
3. Open **Patient Record**, run **AI extraction**, then **Run harmonization**.
4. Review the inline conflict flags, **Document Sources & Provenance**, and the
   **Conflicts & Warnings** section.
5. Open **AI Clinical Chat**, build the index if needed, and ask one of the
   questions above. Citations should name the source file and report
   `Page not available` because these are TXT files.
