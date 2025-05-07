from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import numpy as np
from io import BytesIO
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Keywords ---
pymupdf_keywords = [
    "Tabular Detail - Non Guaranteed",
    "Annual Cost Summary",
]

pdfplumber_keywords = [
    "Your policy's illustrated values",
    "Your policy's current charges summary",
    "Basic Ledger, Non-guaranteed scenario",
    "Policy Charges Ledger",
]

TABULAR_HEADERS = [
    "end of year", "age", "annualized premium outlay", "loans/partial surrenders",
    "total loan balance", "net annual outlay",
    "accumulated value (current)", "net surrender value (current)", "net death benefit (current)",
    "accumulated value (guaranteed)", "net surrender value (guaranteed)", "net death benefit (guaranteed)"
]

COST_SUMMARY_HEADERS = [
    "Yr", "Age", "Premium Outlay", "Base COI", "Policy Expense",
    "Per 1000 (Base)", "Total Charges", "Loan Charges", "Loans & Partial Surrenders",
    "Index Segment", "Investment Gain/Loss", "Net Policy Value EOY",
    "Net Death Benefit EOY"
]

# --- Extractor Functions for PyMuPDF ---
def extract_projection_table(page):
    lines = []
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        for line in block.get("lines", []):
            for span in line["spans"]:
                y = span["bbox"][1]
                if 350 < y < 750:
                    lines.append((y, span["bbox"][0], span["text"].strip()))

    lines.sort(key=lambda x: (round(x[0], 1), x[1]))
    table_data, current_row, last_y = [], [], None
    for y, x, text in lines:
        if last_y is None or abs(y - last_y) < 6:
            current_row.append((x, text))
        else:
            current_row.sort()
            row = [t for _, t in current_row]
            if sum(c.replace(",", "").replace(".", "").isdigit() for c in row) >= 5:
                table_data.append(row)
            current_row = [(x, text)]
        last_y = y
    if current_row:
        current_row.sort()
        row = [t for _, t in current_row]
        if sum(c.replace(",", "").replace(".", "").isdigit() for c in row) >= 5:
            table_data.append(row)
    table_data = [row[:len(TABULAR_HEADERS)] + [''] * (len(TABULAR_HEADERS) - len(row)) for row in table_data]
    return pd.DataFrame(table_data, columns=TABULAR_HEADERS)

def extract_cost_summary_table(page):
    lines = []
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        for line in block.get("lines", []):
            for span in line["spans"]:
                y = span["bbox"][1]
                if 300 < y < 750:
                    lines.append((y, span["bbox"][0], span["text"].strip()))

    lines.sort(key=lambda x: (round(x[0], 1), x[1]))
    table_data, current_row, last_y = [], [], None
    for y, x, text in lines:
        if last_y is None or abs(y - last_y) < 6:
            current_row.append((x, text))
        else:
            current_row.sort()
            row = [t for _, t in current_row]
            if sum(c.replace(",", "").replace(".", "").isdigit() for c in row) >= 10:
                table_data.append(row)
            current_row = [(x, text)]
        last_y = y
    if current_row:
        current_row.sort()
        row = [t for _, t in current_row]
        if sum(c.replace(",", "").replace(".", "").isdigit() for c in row) >= 10:
            table_data.append(row)
    table_data = [row[:len(COST_SUMMARY_HEADERS)] + [''] * (len(COST_SUMMARY_HEADERS) - len(row)) for row in table_data]
    return pd.DataFrame(table_data, columns=COST_SUMMARY_HEADERS)

# --- API Endpoint ---
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        pdf_file = BytesIO(content)
        results = []

        found_keywords = []

        # --- Search all text ---
        full_text = ""
        with fitz.open(stream=pdf_file, filetype="pdf") as doc:
            for page in doc:
                full_text += page.get_text("text").lower()

        for key in pymupdf_keywords + pdfplumber_keywords:
            if key.lower() in full_text:
                found_keywords.append(key)

        if not found_keywords:
            return JSONResponse(content={"message": "No matching keywords found."}, status_code=200)

        # --- Re-open for processing ---
        pdf_file.seek(0)

        # -- PyMuPDF-based extraction
        if any(k in found_keywords for k in pymupdf_keywords):
            with fitz.open(stream=pdf_file, filetype="pdf") as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text("text").lower()

                    if "tabular detail - non guaranteed" in text:
                        df = extract_projection_table(page)
                        if not df.empty:
                            results.append({
                                "source": "Tabular Detail",
                                "page": page_num + 1,
                                "data": df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
                            })

                    if "annual cost summary" in text:
                        df = extract_cost_summary_table(page)
                        if not df.empty:
                            results.append({
                                "source": "Annual Cost Summary",
                                "page": page_num + 1,
                                "data": df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
                            })

        # -- pdfplumber-based extraction
        if any(k in found_keywords for k in pdfplumber_keywords):
            pdf_file.seek(0)
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = (page.extract_text() or "").lower()

                    for keyword in pdfplumber_keywords:
                        if keyword.lower() in text:
                            tables = page.extract_tables()
                            for table in tables:
                                if not table or len(table) < 2:
                                    continue
                                df = pd.DataFrame(table[1:], columns=table[0])
                                if not df.empty:
                                    results.append({
                                        "source": keyword,
                                        "page": page_num + 1,
                                        "data": df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
                                    })

        if not results:
            return JSONResponse(content={"message": "Keywords matched but no tables extracted."}, status_code=200)

        return JSONResponse(content={"tables": results}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
