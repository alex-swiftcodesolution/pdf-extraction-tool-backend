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
import re

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
    "Annual Cost Summary",
    "Policy Charges and Other Expenses"
]

PDFPLUMBER_KEYWORDS = [
    "Your policy's illustrated values",
    "Your policy's current charges summary",
    "Basic Ledger, Non-guaranteed scenario",
    "Policy Charges Ledger",
    "Current Illustrated Rate*",
]

# NW
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

# MN
ILLUSTRATED_VALUES_HEADERS = [
    "Year","Age","Premium Outlay","Net Outlay",
    "[Guaranteed Values][2.00% crediting rate and maximum charges]Surrender Value",
    "[Guaranteed Values][2.00% crediting rate and maximum charges]Surrender Value",
    "[Non-Guaranteed Values][4.25% alternative crediting rate and current charges]Cash Value",
    "[Non-Guaranteed Values][4.25% alternative crediting rate and current charges]Surrender Value",
    "[Non-Guaranteed Values][4.25% alternative crediting rate and current charges]Death Benefit",
    "[Non-Guaranteed Values][6.59% illustrated crediting rate and current charges]Cash Value",
    "[Non-Guaranteed Values][6.59% illustrated crediting rate and current charges]Surrender Value",
    "[Non-Guaranteed Values][6.59% illustrated crediting rate and current charges]Death Benfit"
]
# MN
POLICY_CURRENT_CHARGES_SUMMARY_HEADERS=["Year","Age","Premium Outlay","Premium Charge","Cost of Insurance Charge","Policy Issue Charge","Additional Charges","Bonus Interest Credit","Additional Policy Credits","Surrenders and Loans","Interest and Crediting Earned","[Non-Guaranteed Values][Using illustrated crediting rates and current charges]Cash Value","[Non-Guaranteed Values][Using illustrated crediting rates and current charges]Surrender Value","[Non-Guaranteed Values][Using illustrated crediting rates and current charges]Death Benfit"]

# LSW
POLICY_CHARGES_HEADERS = [
    "Policy Year", "Age", "Premium Outlay", "Premium Expense Charge",
    "cost of insurance", "cost of other benefits", "policy fee",
    "Expense Charge", "Accumulated value charge", "Policy charges", "interest credit",
    "additional bonus", "total credits", "accumulated value", "Surrender charges", "cash surrender value",
    "net death benefit","ex"
]

# ---------------------- PyMuPDF Extraction ----------------------

# NW
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

# NW
def extract_cost_summary_table(page):
    lines = []
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        for line in block.get("lines", []):
            for span in line["spans"]:
                y = span["bbox"][1]
                if 100 < y < 750:
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

# LSW
def extract_policy_charges_table(page, POLICY_CHARGES_HEADERS):
    # Define the function to check if text is numeric or currency
    def is_numeric_or_currency(text):
        # Matches numbers like 123, 123.45, $123.45, -$1,234.56
        return bool(re.match(r'^-?\$?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\$?\d+(\.\d+)?$', text))
    
    lines = []
    blocks = page.get_text("dict")["blocks"]
    
    # Extract all text lines within the defined vertical range
    for block in blocks:
        for line in block.get("lines", []):
            for span in line["spans"]:
                y = span["bbox"][1]
                # Include only relevant rows based on vertical position
                if 20 < y < 1000:
                    lines.append((y, span["bbox"][0], span["text"].strip()))

    # Sort by Y (row-wise), then by X (column-wise)
    lines.sort(key=lambda x: (round(x[0], 1), x[1]))

    # Group into rows
    table_data, current_row, last_y = [], [], None
    for y, x, text in lines:
        if last_y is None or abs(y - last_y) < 5:
            current_row.append((x, text))
        else:
            current_row.sort()  # Sort left to right
            # Filter out non-numeric or non-currency values for each row
            row = [t for _, t in current_row if is_numeric_or_currency(t)]
            if len(row) >= 5:  # Only add rows with 10 or more valid entries
                table_data.append(row)
            current_row = [(x, text)]
        last_y = y

    # Append the final row, ensuring the filter is applied
    if current_row:
        current_row.sort()
        row = [t for _, t in current_row if is_numeric_or_currency(t)]
        if len(row) >= 10:  # Only add rows with 10 or more valid entries
            table_data.append(row)

    # If no valid data was extracted, return an empty DataFrame
    if not table_data:
        return pd.DataFrame()

    # Normalize all rows to exactly 17 columns
    expected_cols = 18
    normalized_data = [
        row[:expected_cols] + [''] * (expected_cols - len(row))
        for row in table_data
    ]

    # Optional: print shape and sample rows for debug
    print(f"Extracted {len(normalized_data)} rows with {expected_cols} columns.")
    for r in normalized_data[:3]:
        print("Sample row:", r)

    # Create DataFrame using provided headers
    df = pd.DataFrame(normalized_data, columns=POLICY_CHARGES_HEADERS)

    # Transpose the DataFrame (rows into columns)
    df_transposed = df.T

    # Flip the columns: Reverse the order of the columns
    df_transposed = df_transposed.iloc[:, ::-1]  # This reverses the column order

    return df_transposed

# ------------------- pdfplumber Flexible Logic ------------------

def extract_tables_with_flexible_headers(pdf):
    tables_by_text = {text: [] for text in PDFPLUMBER_KEYWORDS}
    # Collect rows for "Policy Charges Ledger" and "Policy Charges and Other Expenses" across pages
    policy_charges_ledger_rows = []
    policy_charges_ledger_pages = []

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
                    # Keep all rows, including empty ones, to avoid premature truncation
                    cleaned = [[str(cell).strip() if cell else "" for cell in row] for row in table]
                    if not cleaned:
                        continue

                    if keyword.lower() == "your policy's illustrated values":
                        # Logic for "Your policy's illustrated values" (unchanged)
                        headers = ILLUSTRATED_VALUES_HEADERS
                        cleaned = [row for row in cleaned if any(cell for cell in row)]  # Remove empty rows here
                        if len(cleaned) < 5:
                            logger.warning(f"Table on page {page.page_number} has {len(cleaned)} rows, cannot skip 5 rows")
                            data_rows = cleaned
                        else:
                            remaining_rows = cleaned[5:]  # Skip first 5 rows
                            data_rows = []
                            i = 0
                            while i < len(remaining_rows):
                                data_rows.extend(remaining_rows[i:i+5])
                                i += 6  # Skip the next row after 5
                            if not data_rows:
                                logger.info(f"No data rows remain after skipping top 5 rows and applying skip pattern on page {page.page_number}")
                                continue
                        data_rows = [row[:len(headers)] + [""] * (len(headers) - len(row)) for row in data_rows]
                        df = pd.DataFrame(data_rows, columns=headers)
                        df["Source_Text"] = keyword
                        df["Page_Number"] = page.page_number
                        if not df.empty:
                            tables_by_text[keyword].append(df)

                    elif keyword.lower() == "your policy's current charges summary":
                        # Logic for "Your policy's current charges summary" (unchanged)
                        
                        headers = POLICY_CURRENT_CHARGES_SUMMARY_HEADERS
                        cleaned = [row for row in cleaned if any(cell for cell in row)]  # Remove empty rows here
                        if len(cleaned) < 3:
                            logger.warning(f"Table on page {page.page_number} has {len(cleaned)} rows, cannot skip 3 rows")
                            data_rows = cleaned
                        else:
                            remaining_rows = cleaned[3:]  # Skip first 3 rows
                            data_rows = []
                            i = 0
                            while i < len(remaining_rows):
                                data_rows.extend(remaining_rows[i:i+5])
                                i += 6  # Skip the next row after 5
                            if not data_rows:
                                logger.info(f"No data rows remain after skipping top 3 rows and applying skip pattern on page {page.page_number}")
                                continue
                        data_rows = [row[:len(headers)] + [""] * (len(headers) - len(row)) for row in data_rows]
                        df = pd.DataFrame(data_rows, columns=headers)
                        df["Source_Text"] = keyword
                        df["Page_Number"] = page.page_number
                        if not df.empty:
                            tables_by_text[keyword].append(df)

                    elif keyword.lower() == "policy charges ledger":
                        # Collect rows for multi-page processing
                        policy_charges_ledger_rows.extend(cleaned)
                        policy_charges_ledger_pages.append(page.page_number)

                    else:
                        # Existing logic for other keywords
                        cleaned = [row for row in cleaned if any(cell for cell in row)]  # Remove empty rows here
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

    # Process "Policy Charges Ledger" rows
    if policy_charges_ledger_rows:
        cleaned = [row for row in policy_charges_ledger_rows if any(cell.strip() for cell in row)]
        if not cleaned:
            logger.info("No non-empty rows found for Policy Charges Ledger")
        else:
            header_keywords = {"year", "age"}
            header_row_index = -1
            for idx, row in enumerate(cleaned[:6]):
                normalized = [cell.lower() for cell in row]
                if any(any(kw in cell for kw in header_keywords) for cell in normalized):
                    header_row_index = idx
                    break

            if header_row_index == -1:
                logger.warning("No header row found for Policy Charges Ledger")
            else:
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
                df["Source_Text"] = "Policy Charges Ledger"
                df["Page_Number"] = min(policy_charges_ledger_pages)
                if not df.empty:
                    tables_by_text["Policy Charges Ledger"].append(df)

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
                                "keyword": "Tabular Detail - Non Guaranteed",
                                "extractor": "PyMuPDF",
                                "data": df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
                            })

                    if "annual cost summary" in text:
                        df = extract_cost_summary_table(page)
                        if not df.empty:
                            results.append({
                                "source": "Annual Cost Summary",
                                "page": page_num + 1,
                                "keyword": "Annual Cost Summary",
                                "extractor": "PyMuPDF",
                                "data": df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
                            })
                            
                    if "policy charges and other expenses" in text.lower():
                        df = extract_policy_charges_table(page,POLICY_CHARGES_HEADERS)
                        if not df.empty:
                            print("API Data:", df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records"))
                            results.append({
                                "source": "Policy Charges and Other Expenses",
                                "page": page_num + 1,
                                "keyword": "Policy Charges and Other Expenses",
                                "extractor": "PyMuPDF",
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
                                "keyword": source_text,
                                "extractor": "pdfplumber",
                                "data": df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
                            })

        if not results:
            return JSONResponse(content={"message": "Keywords matched but no tables extracted."}, status_code=200)

        return JSONResponse(content={"tables": jsonable_encoder(results)}, status_code=200)

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")