from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import pdfplumber
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from io import BytesIO
from collections import Counter

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keywords to search
pdfplumber_keywords = [
    "Your policy's illustrated values",
    "Your policy's current charges summary",
    "Basic Ledger, Non-guaranteed scenario",
    "Policy Charges Ledger",
]
pymupdf_keywords = [
    "Tabular Detail - Non Guaranteed",
    "Annual Cost Summary",
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
    "Index Segment", "Investment Gain/Loss", "Net Policy Value EOY", "Net Death Benefit EOY"
]

def deduplicate_headers(headers):
    counts = Counter()
    result = []
    for h in headers:
        counts[h] += 1
        result.append(f"{h}_{counts[h]}" if counts[h] > 1 else h)
    return result

def extract_projection_table(page):
    blocks = page.get_text("dict")["blocks"]
    lines = []
    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                y = span["bbox"][1]
                if y < 350 or y > 750:
                    continue
                if text:
                    lines.append((y, span["bbox"][0], text))
    if not lines:
        return None
    lines.sort(key=lambda x: (round(x[0], 1), x[1]))

    table_data, current_row, last_y = [], [], None
    for y, x, text in lines:
        if last_y is None or abs(y - last_y) < 6:
            current_row.append((x, text))
        else:
            current_row.sort(key=lambda x: x[0])
            row = [t for _, t in current_row]
            if sum(t.replace(",", "").replace(".", "").isdigit() for t in row) >= 5:
                table_data.append(row)
            current_row = [(x, text)]
        last_y = y
    if current_row:
        current_row.sort(key=lambda x: x[0])
        row = [t for _, t in current_row]
        if sum(t.replace(",", "").replace(".", "").isdigit() for t in row) >= 5:
            table_data.append(row)

    max_cols = len(TABULAR_HEADERS)
    table_data = [row[:max_cols] + [''] * (max_cols - len(row)) for row in table_data]
    return pd.DataFrame(table_data, columns=TABULAR_HEADERS)

def extract_cost_summary_table(page):
    blocks = page.get_text("dict")["blocks"]
    lines = []
    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                y = span["bbox"][1]
                if y < 300 or y > 750:
                    continue
                if text:
                    lines.append((y, span["bbox"][0], text))
    if not lines:
        return None
    lines.sort(key=lambda x: (round(x[0], 1), x[1]))

    table_data, current_row, last_y = [], [], None
    for y, x, text in lines:
        if last_y is None or abs(y - last_y) < 6:
            current_row.append((x, text))
        else:
            current_row.sort(key=lambda x: x[0])
            row = [t for _, t in current_row]
            if sum(t.replace(",", "").replace(".", "").isdigit() for t in row) >= 10:
                table_data.append(row)
            current_row = [(x, text)]
        last_y = y
    if current_row:
        current_row.sort(key=lambda x: x[0])
        row = [t for _, t in current_row]
        if sum(t.replace(",", "").replace(".", "").isdigit() for t in row) >= 10:
            table_data.append(row)

    max_cols = len(COST_SUMMARY_HEADERS)
    table_data = [row[:max_cols] + [''] * (max_cols - len(row)) for row in table_data]
    return pd.DataFrame(table_data, columns=COST_SUMMARY_HEADERS)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    content = await file.read()
    pdf_bytes = BytesIO(content)

    all_results = []

    # First handle PyMuPDF-based pages
    doc = fitz.open(stream=pdf_bytes.getvalue(), filetype="pdf")
    pages_text = [page.get_text("text") for page in doc]

    for i, (page, text) in enumerate(zip(doc, pages_text)):
        lowered = text.lower()
        if "tabular detail - non guaranteed" in lowered:
            df = extract_projection_table(page)
            if df is not None:
                df["Source_Text"] = "Tabular Detail - Non Guaranteed"
                df["Page_Number"] = i + 1
                all_results.append(df)
        elif "annual cost summary" in lowered:
            df = extract_cost_summary_table(page)
            if df is not None:
                df["Source_Text"] = "Annual Cost Summary"
                df["Page_Number"] = i + 1
                all_results.append(df)
    doc.close()

    # Now handle pdfplumber-based pages
    pdf_bytes.seek(0)
    with pdfplumber.open(pdf_bytes) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            for keyword in pdfplumber_keywords:
                if keyword.lower() in page_text.lower():
                    tables = page.extract_tables({
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "text",
                        "intersection_tolerance": 5,
                        "snap_tolerance": 3,
                    })
                    if not tables:
                        continue
                    for table in tables:
                        cleaned = [
                            [str(cell).strip() if cell else "" for cell in row]
                            for row in table if any(cell and str(cell).strip() for cell in row)
                        ]
                        if not cleaned:
                            continue

                        # detect header row
                        header_row_index = -1
                        for idx, row in enumerate(cleaned[:6]):
                            if any("year" in cell.lower() or "age" in cell.lower() for cell in row):
                                header_row_index = idx
                                break
                        if header_row_index == -1:
                            continue

                        max_header_rows = 3
                        start_header_idx = max(0, header_row_index - (max_header_rows - 1))
                        header_rows = cleaned[start_header_idx:header_row_index + 1]

                        max_cols = max(len(row) for row in header_rows)
                        normalized_rows = [row + [""] * (max_cols - len(row)) for row in header_rows]
                        df_header = pd.DataFrame(normalized_rows).ffill(axis=1)

                        headers = [
                            " ".join(
                                str(df_header.iloc[row_idx, col_idx]).strip()
                                for row_idx in range(len(df_header))
                                if str(df_header.iloc[row_idx, col_idx]).strip()
                            )
                            for col_idx in range(df_header.shape[1])
                        ]
                        headers = deduplicate_headers(headers)
                        data_rows = cleaned[header_row_index + 1:]
                        data_rows = [row + [""] * (len(headers) - len(row)) for row in data_rows]

                        df = pd.DataFrame(data_rows, columns=headers)
                        if df.empty:
                            continue
                        df["Source_Text"] = keyword
                        df["Page_Number"] = page.page_number
                        all_results.append(df)

    if not all_results:
        return JSONResponse(content={"message": "No tables found."}, status_code=200)

    output = []
    for df in all_results:
        df = df.replace([np.nan, np.inf, -np.inf], None)
        output.append({
            "source_text": df["Source_Text"].iloc[0],
            "page_number": int(df["Page_Number"].iloc[0]),
            "data": df.drop(columns=["Source_Text", "Page_Number"]).to_dict(orient="records")
        })

    return JSONResponse(content={"tables": jsonable_encoder(output)}, status_code=200)
