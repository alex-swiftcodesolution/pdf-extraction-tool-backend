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
from collections import Counter

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

# ---------------------------- Config ----------------------------

PYMUPDF_KEYWORDS = [
    "Tabular Detail - Non Guaranteed",
    "Annual Cost Summary"
]

PDFPLUMBER_KEYWORDS = [
    "Your policy's illustrated values",
    "Your policy's current charges summary",
    "Basic Ledger, Non-guaranteed scenario",
    "Policy Charges Ledger"
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

# ---------------------- PyMuPDF Extraction ----------------------

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

# ------------------- pdfplumber Flexible Logic ------------------

def extract_tables_with_flexible_headers(pdf):
    tables_by_text = {text: [] for text in PDFPLUMBER_KEYWORDS}
    for page in pdf.pages:
        text = (page.extract_text() or "").lower()
        for keyword in PDFPLUMBER_KEYWORDS:
            if keyword.lower() in text:
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "text",
                    "intersection_tolerance": 5,
                    "snap_tolerance": 3,
                })

                for table in tables:
                    cleaned = [[str(cell).strip() if cell else "" for cell in row] for row in table]
                    cleaned = [row for row in cleaned if any(cell for cell in row)]
                    if not cleaned:
                        continue

                    header_keywords = {"year", "age"}
                    header_row_index = -1
                    for idx, row in enumerate(cleaned[:6]):
                        normalized = [cell.lower() for cell in row]
                        if any(any(kw in cell for kw in header_keywords) for cell in normalized):
                            header_row_index = idx
                            break

                    if header_row_index == -1:
                        continue

                    header_rows = cleaned[max(0, header_row_index - 2):header_row_index + 1]
                    max_cols = max(len(row) for row in header_rows)
                    df_header = pd.DataFrame([row + [""] * (max_cols - len(row)) for row in header_rows]).ffill(axis=1)

                    headers = [
                        " ".join(str(df_header.iloc[row_idx, col_idx]).strip() for row_idx in range(len(df_header)))
                        for col_idx in range(df_header.shape[1])
                    ]

                    def deduplicate_headers(headers):
                        counts = Counter()
                        result = []
                        for h in headers:
                            counts[h] += 1
                            result.append(f"{h}_{counts[h]}" if counts[h] > 1 else h)
                        return result

                    headers = deduplicate_headers(headers)
                    data_rows = cleaned[header_row_index + 1:]
                    data_rows = [row + [""] * (len(headers) - len(row)) for row in data_rows]
                    df = pd.DataFrame(data_rows, columns=headers)
                    df["Source_Text"] = keyword
                    df["Page_Number"] = page.page_number
                    if not df.empty:
                        tables_by_text[keyword].append(df)
    return tables_by_text

# --------------------------- Main API ---------------------------

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        pdf_file = BytesIO(content)

        results = []

        # Detect keywords first
        all_text = ""
        with fitz.open(stream=pdf_file, filetype="pdf") as doc:
            for page in doc:
                all_text += page.get_text("text").lower()

        found_keywords = [k for k in PYMUPDF_KEYWORDS + PDFPLUMBER_KEYWORDS if k.lower() in all_text]

        if not found_keywords:
            return JSONResponse(content={"message": "No matching keywords found."}, status_code=200)

        # PyMuPDF processing
        if any(k in found_keywords for k in PYMUPDF_KEYWORDS):
            pdf_file.seek(0)
            with fitz.open(stream=pdf_file, filetype="pdf") as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text("text").lower()

                    if "tabular detail - non guaranteed" in text:
                        df = extract_projection_table(page)
                        if not df.empty:
                            results.append({
                                "source": "Tabular Detail - Non Guaranteed",
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

        # pdfplumber processing
        if any(k in found_keywords for k in PDFPLUMBER_KEYWORDS):
            pdf_file.seek(0)
            with pdfplumber.open(pdf_file) as pdf:
                tables = extract_tables_with_flexible_headers(pdf)
                for source_text, df_list in tables.items():
                    for df in df_list:
                        if not df.empty:
                            results.append({
                                "source": source_text,
                                "page": int(df["Page_Number"].iloc[0]),
                                "data": df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
                            })

        if not results:
            return JSONResponse(content={"message": "Keywords matched but no tables extracted."}, status_code=200)

        return JSONResponse(content={"tables": jsonable_encoder(results)}, status_code=200)

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
