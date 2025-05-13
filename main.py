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
    # NW
    "Annual Cost Summary",
    "Tabular Detail - Non Guaranteed",
    # LSW
    "Policy Charges and Other Expenses",
    "Current Illustrated Rate*" 
]

PDFPLUMBER_KEYWORDS = [
    # MN
    "Your policy's illustrated values",
    "Your policy's current charges summary",
    # ALZ
    "Basic Ledger, Non-guaranteed scenario",
    "Policy Charges Ledger",
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

# LSW
CURRENT_ILLUSTRATED_RATE_HEADERS = [
    "Policy year","Age","Premium Outlay","Planned annual income","Planned annual loan","Accumulated loan amount",
    "Weighted average interest rate","Accumulated value","Cash surrender value","Net death benefit"
]

# ---------------------- PyMuPDF Extraction ----------------------

# NW
# def extract_projection_table(page):
#     lines = []
#     blocks = page.get_text("dict")["blocks"]
#     for block in blocks:
#         for line in block.get("lines", []):
#             for span in line["spans"]:
#                 y = span["bbox"][1]
#                 if 350 < y < 750:
#                     lines.append((y, span["bbox"][0], span["text"].strip()))

#     lines.sort(key=lambda x: (round(x[0], 1), x[1]))
#     table_data, current_row, last_y = [], [], None
#     for y, x, text in lines:
#         if last_y is None or abs(y - last_y) < 6:
#             current_row.append((x, text))
#         else:
#             current_row.sort()
#             row = [t for _, t in current_row]
#             if sum(c.replace(",", "").replace(".", "").isdigit() for c in row) >= 5:
#                 table_data.append(row)
#             current_row = [(x, text)]
#         last_y = y
#     if current_row:
#         current_row.sort()
#         row = [t for _, t in current_row]
#         if sum(c.replace(",", "").replace(".", "").isdigit() for c in row) >= 5:
#             table_data.append(row)
#     table_data = [row[:len(TABULAR_HEADERS)] + [''] * (len(TABULAR_HEADERS) - len(row)) for row in table_data]
#     return pd.DataFrame(table_data, columns=TABULAR_HEADERS)

# --- TUESDAY ---
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

    # --- INSERT: Sum columns 4, 5, 6, 7, 8 (0-based: 3, 4, 5, 6, 7) into a custom column ---
    sum_indices = [3, 4, 5, 6, 7]
    # Calculate sum for each row, handling non-numeric values
    def safe_float(val):
        try:
            return float(val.replace('$', '').replace(',', '').strip())
        except (ValueError, AttributeError):
            return 0.0
    
    summed_values = [
        sum(safe_float(row[i]) for i in sum_indices if i < len(row))
        for row in table_data
    ]
    
    # Create table_data with only the summed column
    table_data = [[str(summed_values[idx])] for idx in range(len(table_data))]
    filtered_headers = ["Total Financial Metrics Sum"]  # Custom column name
    expected_cols = 1
    # --- END INSERT ---

    table_data = [row[:expected_cols] + [''] * (expected_cols - len(row)) for row in table_data]
    return pd.DataFrame(table_data, columns=filtered_headers)
# --- TUESDAY ---

# NW
# def extract_cost_summary_table(page):
#     lines = []
#     blocks = page.get_text("dict")["blocks"]
#     for block in blocks:
#         for line in block.get("lines", []):
#             for span in line["spans"]:
#                 y = span["bbox"][1]
#                 if 100 < y < 750:
#                     lines.append((y, span["bbox"][0], span["text"].strip()))

#     lines.sort(key=lambda x: (round(x[0], 1), x[1]))
#     table_data, current_row, last_y = [], [], None
#     for y, x, text in lines:
#         if last_y is None or abs(y - last_y) < 6:
#             current_row.append((x, text))
#         else:
#             current_row.sort()
#             row = [t for _, t in current_row]
#             if sum(c.replace(",", "").replace(".", "").isdigit() for c in row) >= 10:
#                 table_data.append(row)
#             current_row = [(x, text)]
#         last_y = y
#     if current_row:
#         current_row.sort()
#         row = [t for _, t in current_row]
#         if sum(c.replace(",", "").replace(".", "").isdigit() for c in row) >= 10:
#             table_data.append(row)
#     table_data = [row[:len(COST_SUMMARY_HEADERS)] + [''] * (len(COST_SUMMARY_HEADERS) - len(row)) for row in table_data]
#     return pd.DataFrame(table_data, columns=COST_SUMMARY_HEADERS)

# --- TUESDAY ---
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

    # --- INSERT: Filter specific columns (1, 2, 3, 10, 11, 12; 0-based: 0, 1, 2, 9, 10, 11) ---
    selected_indices = [0, 1, 2, 9, 10, 11]
    table_data = [[row[i] for i in selected_indices if i < len(row)] for row in table_data]
    expected_cols = len(selected_indices)
    filtered_headers = [COST_SUMMARY_HEADERS[i] for i in selected_indices if i < len(COST_SUMMARY_HEADERS)]
    # --- END INSERT ---

    table_data = [row[:expected_cols] + [''] * (expected_cols - len(row)) for row in table_data]
    return pd.DataFrame(table_data, columns=filtered_headers)
# --- TUESDAY ---

# LSW
# def extract_policy_charges_table(page, POLICY_CHARGES_HEADERS):
#     def is_numeric_or_currency(text):
#         # Matches currency (e.g., $1,234.56), integers (e.g., 1234), decimals (e.g., 1234.56)
#         return bool(re.match(r'^-?\$?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$', text))
    
#     lines = []
#     blocks = page.get_text("dict")["blocks"]
    
#     for block in blocks:
#         for line in block.get("lines", []):
#             for span in line["spans"]:
#                 y = span["bbox"][1]
#                 if 20 < y < 1000:
#                     lines.append((y, span["bbox"][0], span["text"].strip()))

#     lines.sort(key=lambda x: (round(x[0], 1), x[1]))  # Ensures top-to-bottom order
#     table_data, current_row, last_y = [], [], None
#     for y, x, text in lines:
#         if last_y is None or abs(y - last_y) < 5:
#             current_row.append((x, text))
#         else:
#             current_row.sort()
#             row = [t for _, t in current_row if is_numeric_or_currency(t)]
            
#             print("Current row before filtering:", [t for _, t in current_row])
            
#             if len(row) >= 3:  # Relaxed to 3 to ensure last row is included
#                 table_data.append(row)
#             current_row = [(x, text)]
#         last_y = y

#     if current_row:
#         current_row.sort()
#         row = [t for _, t in current_row if is_numeric_or_currency(t)]
        
#         print("Current row before filtering:", [t for _, t in current_row])
        
#         if len(row) >= 3:
#             table_data.append(row)  # Ensure last row is included

#     if not table_data:
#         logger.info("No valid table data extracted for Policy Charges and Other Expenses")
#         return pd.DataFrame()

#     expected_cols = len(POLICY_CHARGES_HEADERS)
    
#     normalized_data = [
#         row[:expected_cols] + [''] * (expected_cols - len(row))
#         for row in table_data
#     ][::-1]
    
#     df = pd.DataFrame(normalized_data, columns=POLICY_CHARGES_HEADERS)
    
#     df_transposed = df.transpose()
    
#     return df_transposed

# --- TUESDAY ---
def extract_policy_charges_table(page, POLICY_CHARGES_HEADERS):
    def is_numeric_or_currency(text):
        return bool(re.match(r'^-?\$?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$', text))
    
    lines = []
    blocks = page.get_text("dict")["blocks"]
    
    for block in blocks:
        for line in block.get("lines", []):
            for span in line["spans"]:
                y = span["bbox"][1]
                if 20 < y < 1000:
                    lines.append((y, span["bbox"][0], span["text"].strip()))

    # --- MODIFY: Sort lines by y-coordinate (top-to-bottom) and then x-coordinate (left-to-right) ---
    lines.sort(key=lambda x: (round(x[0], 1), x[1]))
    table_data, current_row, last_y = [], [], None
    for y, x, text in lines:
        if last_y is None or abs(y - last_y) < 5:  # Group lines within 5 units as a single row
            current_row.append((x, text))
        else:
            # Sort by x-coordinate to ensure correct column order
            current_row.sort()
            row = [t for _, t in current_row if is_numeric_or_currency(t)]
            if len(row) >= 3:  # Ensure row has enough numeric values
                table_data.append(row)
            current_row = [(x, text)]
        last_y = y

    if current_row:
        current_row.sort()
        row = [t for _, t in current_row if is_numeric_or_currency(t)]
        if len(row) >= 3:
            table_data.append(row)

    if not table_data:
        logger.info("No valid table data extracted for Policy Charges and Other Expenses")
        return pd.DataFrame()

    # --- MODIFY: Filter specific columns (1, 2, 3, 4, 8, 9, 10; 0-based: 0, 1, 2, 3, 7, 8, 9) ---
    selected_indices = [0, 1, 2, 3, 7, 8, 9]
    table_data = [[row[i] for i in selected_indices if i < len(row)] for row in table_data]
    expected_cols = len(selected_indices)
    filtered_headers = [POLICY_CHARGES_HEADERS[i] for i in selected_indices if i < len(POLICY_CHARGES_HEADERS)]
    # --- END MODIFY ---

    # --- MODIFY: Remove row reversal to preserve correct row-column structure ---
    normalized_data = [
        row[:expected_cols] + [''] * (expected_cols - len(row))
        for row in table_data
    ]
    # --- END MODIFY ---

    # --- MODIFY: Create DataFrame with correct headers and data ---
    df = pd.DataFrame(normalized_data, columns=filtered_headers)
    # --- END MODIFY ---
    
    return df
# --- TUESDAY ---

# LSW
# def extract_current_illustrated_rate_table(page):
#     def is_numeric_or_currency(text):
#         return bool(re.match(r'^-?\$?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$|^-?\d*\.\d+%?$', text))    
    
#     lines = []
#     blocks = page.get_text("dict")["blocks"]
    
#     for block in blocks:
#         for line in block.get("lines", []):
#             for span in line["spans"]:
#                 y = span["bbox"][1]
#                 if 190 < y < 800:  # Adjusted range for flexibility
#                     lines.append((y, span["bbox"][0], span["text"].strip()))

#     lines.sort(key=lambda x: (round(x[0], 1), x[1]))
#     table_data, current_row, last_y = [], [], None
#     for y, x, text in lines:
#         if last_y is None or abs(y - last_y) < 6:
#             current_row.append((x, text))
#         else:
#             current_row.sort()
#             row = [t for _, t in current_row]
#             valid_values = [t for t in row if is_numeric_or_currency(t)]
#             if len(row) >= 3:  # Require at least 5 numeric/currency values
#                 table_data.append(row)
#             current_row = [(x, text)]
#         last_y = y
    
#     if current_row:
#         current_row.sort()
#         row = [t for _, t in current_row if is_numeric_or_currency(t)]
#         if len(row) >= 5:
#             table_data.append(row)
    
#     if not table_data:
#         return pd.DataFrame()
    
#     expected_cols = len(CURRENT_ILLUSTRATED_RATE_HEADERS)
#     normalized_data = [
#         row[:expected_cols] + [''] * (expected_cols - len(row))
#         for row in table_data
#     ]
    
#     df = pd.DataFrame(normalized_data, columns=CURRENT_ILLUSTRATED_RATE_HEADERS)
#     return df

# --- TUESDAY ---
def extract_current_illustrated_rate_table(page):
    def is_numeric_or_currency(text):
        return bool(re.match(r'^-?\$?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$|^-?\d*\.\d+%?$', text))    
    
    lines = []
    blocks = page.get_text("dict")["blocks"]
    
    for block in blocks:
        for line in block.get("lines", []):
            for span in line["spans"]:
                y = span["bbox"][1]
                if 190 < y < 800:
                    lines.append((y, span["bbox"][0], span["text"].strip()))

    lines.sort(key=lambda x: (round(x[0], 1), x[1]))
    table_data, current_row, last_y = [], [], None
    for y, x, text in lines:
        if last_y is None or abs(y - last_y) < 6:
            current_row.append((x, text))
        else:
            current_row.sort()
            row = [t for _, t in current_row]
            valid_values = [t for t in row if is_numeric_or_currency(t)]
            if len(row) >= 3:
                table_data.append(row)
            current_row = [(x, text)]
        last_y = y
    
    if current_row:
        current_row.sort()
        row = [t for _, t in current_row if is_numeric_or_currency(t)]
        if len(row) >= 5:
            table_data.append(row)
    
    if not table_data:
        return pd.DataFrame()
    
    # --- INSERT: Filter only column 10 (0-based: 9) ---
    selected_indices = [9]
    table_data = [[row[i] for i in selected_indices if i < len(row)] for row in table_data]
    expected_cols = len(selected_indices)
    filtered_headers = [CURRENT_ILLUSTRATED_RATE_HEADERS[i] for i in selected_indices if i < len(CURRENT_ILLUSTRATED_RATE_HEADERS)]
    # --- END INSERT ---

    normalized_data = [
        row[:expected_cols] + [''] * (expected_cols - len(row))
        for row in table_data
    ]
    
    df = pd.DataFrame(normalized_data, columns=filtered_headers)
    return df
# --- TUESDAY ---

# ------------------- pdfplumber Flexible Logic ------------------

def extract_tables_with_flexible_headers(pdf):
    tables_by_text = {text: [] for text in PDFPLUMBER_KEYWORDS}
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
                    cleaned = [[str(cell).strip() if cell else "" for cell in row] for row in table]
                    if not cleaned:
                        continue

                    if keyword.lower() == "your policy's illustrated values":
                        headers = ILLUSTRATED_VALUES_HEADERS
                        cleaned = [row for row in cleaned if any(cell for cell in row)]
                        if len(cleaned) < 5:
                            logger.warning(f"Table on page {page.page_number} has {len(cleaned)} rows, cannot skip 5 rows")
                            data_rows = cleaned
                        else:
                            remaining_rows = cleaned[5:]
                            data_rows = []
                            i = 0
                            while i < len(remaining_rows):
                                data_rows.extend(remaining_rows[i:i+5])
                                i += 6
                            if not data_rows:
                                logger.info(f"No data rows remain after skipping top 5 rows on page {page.page_number}")
                                continue
                        data_rows = [row[:len(headers)] + [""] * (len(headers) - len(row)) for row in data_rows]

                        # --- INSERT: Filter specific columns (1, 2, 3, 4, 11, 12, 13; 0-based: 0, 1, 2, 3, 10, 11, 12) ---
                        selected_indices = [0, 1, 2, 3, 10, 11, 12]
                        filtered_headers = [headers[i] for i in selected_indices if i < len(headers)]
                        data_rows = [[row[i] for i in selected_indices if i < len(row)] for row in data_rows]
                        # --- END INSERT ---

                        df = pd.DataFrame(data_rows, columns=filtered_headers)
                        df["Source_Text"] = keyword
                        df["Page_Number"] = page.page_number
                        if not df.empty:
                            tables_by_text[keyword].append(df)

                    elif keyword.lower() == "your policy's current charges summary":
                        headers = POLICY_CURRENT_CHARGES_SUMMARY_HEADERS
                        cleaned = [row for row in cleaned if any(cell for cell in row)]
                        if len(cleaned) < 3:
                            logger.warning(f"Table on page {page.page_number} has {len(cleaned)} rows, cannot skip 3 rows")
                            data_rows = cleaned
                        else:
                            remaining_rows = cleaned[3:]
                            data_rows = []
                            i = 0
                            while i < len(remaining_rows):
                                data_rows.extend(remaining_rows[i:i+5])
                                i += 6
                            if not data_rows:
                                logger.info(f"No data rows remain after skipping top 3 rows on page {page.page_number}")
                                continue
                        data_rows = [row[:len(headers)] + [""] * (len(headers) - len(row)) for row in data_rows]

                        # --- INSERT: Sum columns 4, 5, 6, 7 (0-based: 3, 4, 5, 6) into a custom column, exclude them ---
                        sum_indices = [3, 4, 5, 6]
                        # Calculate sum for each row, handling non-numeric values
                        def safe_float(val):
                            try:
                                return float(val.replace('$', '').replace(',', '').strip())
                            except (ValueError, AttributeError):
                                return 0.0
                        
                        summed_values = [
                            sum(safe_float(row[i]) for i in sum_indices if i < len(row))
                            for row in data_rows
                        ]
                        
                        # Select all columns except sum_indices, then append summed column
                        filtered_indices = [i for i in range(len(headers)) if i not in sum_indices]
                        filtered_headers = [headers[i] for i in filtered_indices]
                        filtered_headers.append("Total Charges Sum")  # Custom column name
                        
                        data_rows = [
                            [row[i] for i in filtered_indices if i < len(row)] + [str(summed_values[idx])]
                            for idx, row in enumerate(data_rows)
                        ]
                        # --- END INSERT ---

                        df = pd.DataFrame(data_rows, columns=filtered_headers)
                        df["Source_Text"] = keyword
                        df["Page_Number"] = page.page_number
                        if not df.empty:
                            tables_by_text[keyword].append(df)
                            
                    # Handle "Basic Ledger, Non-guaranteed scenario"
                    elif keyword.lower() == "basic ledger, non-guaranteed scenario":
                        cleaned = [row for row in cleaned if any(cell for cell in row)]
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

                        # headers = deduplicate_headers(headers)
                        data_rows = cleaned[header_row_index + 1:]
                        data_rows = [row + [""] * (len(headers) - len(row)) for row in data_rows]

                        # --- INSERT: Filter specific columns (1, 2, 3, 4, 8, 9, 10; 0-based: 0, 1, 2, 3, 7, 8, 9) ---
                        selected_indices = [0, 1, 2, 3, 7, 8, 9]
                        headers = [headers[i] for i in selected_indices if i < len(headers)]
                        data_rows = [[row[i] for i in selected_indices if i < len(row)] for row in data_rows]
                        # --- END INSERT ---

                        df = pd.DataFrame(data_rows, columns=headers)
                        df["Source_Text"] = keyword
                        df["Page_Number"] = page.page_number
                        if not df.empty:
                            tables_by_text[keyword].append(df)

                    elif keyword.lower() == "policy charges ledger":
                        policy_charges_ledger_rows.extend(cleaned)
                        policy_charges_ledger_pages.append(page.page_number)

                    else:
                        cleaned = [row for row in cleaned if any(cell for cell in row)]
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
                
                # --- INSERT: Filter only column 10 (0-based: 9) ---
                selected_indices = [9]
                headers = [headers[i] for i in selected_indices if i < len(headers)]
                data_rows = [[row[i] for i in selected_indices if i < len(row)] for row in data_rows]
                # --- END INSERT ---
                
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
                        df = extract_policy_charges_table(page, POLICY_CHARGES_HEADERS)
                        if not df.empty:
                            results.append({
                                "source": "Policy Charges and Other Expenses",
                                "page": page_num + 1,
                                "keyword": "Policy Charges and Other Expenses",
                                "extractor": "PyMuPDF",
                                "data": df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
                            })
                            
                    if "current illustrated rate*" in text.lower():
                        df = extract_current_illustrated_rate_table(page)
                        if not df.empty:
                            results.append({
                                "source": "Current Illustrated Rate*",
                                "page": page_num + 1,
                                "keyword": "Current Illustrated Rate*",
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