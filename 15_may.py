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
    "Tabular Detail - Non Guaranteed",
    "Annual Cost Summary",
    # LSW
    "Current Illustrated Rate*",
    "Policy Charges and Other Expenses",
]

PDFPLUMBER_KEYWORDS = [
    # MN
    "Your policy's illustrated values",
    "Your policy's current charges summary",
    # ALZ
    "Basic Ledger, Non-guaranteed scenario",
    "Policy Charges Ledger",
]

# UNIVERSAL HEADER FOR ALL TABLES
UNIVERSAL_HEADER_FOR_SEVEN_COL_TABLES = [
    "Age","Policy Year","Premium Outlay","Net Outlay","Cash Value","Surrender Value","Death Benefit"
]
UNIVERSAL_HEADER_FOR_ONE_COL_TABLES = [
    "Charges"
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
    "[Guaranteed Values][2.00% crediting rate and maximum charges]Death Benefit",
    "[Non-Guaranteed Values][4.25% alternative crediting rate and current charges]Cash Value",
    "[Non-Guaranteed Values][4.25% alternative crediting rate and current charges]Surrender Value",
    "[Non-Guaranteed Values][4.25% alternative crediting rate and current charges]Death Benefit",
    "[Non-Guaranteed Values][6.59% illustrated crediting rate and current charges]Cash Value",
    "[Non-Guaranteed Values][6.59% illustrated crediting rate and current charges]Surrender Value",
    "[Non-Guaranteed Values][6.59% illustrated crediting rate and current charges]Death Benfit"
]

# MN
POLICY_CURRENT_CHARGES_SUMMARY_HEADERS = [
    "Year","Age","Premium Outlay","Premium Charge",
    "Cost of Insurance Charge","Policy Issue Charge",
    "Additional Charges","Bonus Interest Credit",
    "Additional Policy Credits","Surrenders and Loans",
    "Interest and Crediting Earned",
    "[Non-Guaranteed Values][Using illustrated crediting rates and current charges]Cash Value",
    "[Non-Guaranteed Values][Using illustrated crediting rates and current charges]Surrender Value",
    "[Non-Guaranteed Values][Using illustrated crediting rates and current charges]Death Benfit"
]

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

# ---------------------- Helper Functions ----------------------

# Added: Helper function to check if a cell contains English words (non-numeric/non-currency text)
def has_english_words(text):
    """
    Returns True if the text contains English words (non-numeric, non-currency, non-percentage).
    Allows numbers, currency ($X,XXX.XX), percentages (X.XX%), and empty strings.
    """
    if not text or text in ("", None, np.nan):
        return False
    # Remove currency symbols, commas, and percentage signs
    cleaned = re.sub(r'[\$,%]', '', str(text)).strip()
    # Check if the remaining text is purely numeric or a decimal
    return not bool(re.match(r'^-?\d*\.?\d*$', cleaned))

# Added: Function to extract fields (illustration_date, insured_name, etc.)
def extract_fields(pdf_text):
    """
    Extracts specified fields from PDF text using regex patterns.
    Returns a dictionary with field values or None if not found.
    """
    fields = {
        "illustration_date": None,
        "insured_name": None,
        "initial_death_benefit": None,
        "assumed_ror": None,
        "minimum_initial_pmt": None
    }
    
    # Normalize text for case-insensitive matching
    text = pdf_text.lower()
    
    # Patterns for each field
    patterns = {
        "illustration_date": r"illustration\s*date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},\s+\d{4})",
        "insured_name": r"insured\s*(?:name)?[:\s]*([a-z\s]+?)(?=\n|$|[a-z\s]*:)",
        "initial_death_benefit": r"initial\s*death\s*benefit[:\s]*[\$]?([\d,]+\.?\d*)",
        "assumed_ror": r"assumed\s*ror[:\s]*([\d.]+%)",
        "minimum_initial_pmt": r"minimum\s*initial\s*pmt[:\s]*[\$]?([\d,]+\.?\d*)"
    }
    
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fields[field] = match.group(1).strip()
    
    return fields

# ---------------------- PyMuPDF Extraction ----------------------

def extract_tabular_detail_non_guaranteed(page):
    import numpy as np
    import re

    def has_english_words(text):
        return bool(re.search(r"[a-zA-Z]", str(text)))

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

    print("Extracted raw rows before processing:")
    for r in table_data:
        print(r)

    # Step 1: Adjust rows to align missing values after "Lapse" or assumed Lapse columns
    expected_len = max(len(row) for row in table_data) if table_data else 12
    expected_len = max(expected_len, 12)  # Ensure at least 12 columns
    last_lapse_indices = set()
    non_lapse_indices = set(range(expected_len))
    default_lapse_indices = {6, 7, 8}  # Assumed Lapse columns

    # First pass: Check for explicit Lapse values
    has_lapse = False
    for row in table_data:
        lapse_indices = [i for i, val in enumerate(row) if val.lower() == 'lapse']
        if lapse_indices:
            last_lapse_indices = set(lapse_indices)
            non_lapse_indices = set(range(expected_len)) - last_lapse_indices
            has_lapse = True
            break

    # If no Lapse values found, use default Lapse indices
    if not has_lapse:
        last_lapse_indices = default_lapse_indices
        non_lapse_indices = set(range(expected_len)) - last_lapse_indices

    # Second pass: Adjust rows
    adjusted_data = []
    for row in table_data:
        if has_lapse and set([i for i, val in enumerate(row) if val.lower() == 'lapse']):
            # Row with explicit Lapse values, keep as is
            new_row = row + ["_"] * (expected_len - len(row))
            adjusted_data.append(new_row)
            continue

        # Adjust row to place underscores in Lapse columns and map values to non-Lapse columns
        new_row = ["_"] * expected_len
        # Map values to non-Lapse indices
        val_idx = 0
        non_lapse_list = sorted(non_lapse_indices)
        # Handle rows with fewer columns by mapping the last values to the last non-Lapse indices
        if len(row) >= len(non_lapse_indices):
            # Normal case: enough values to fill non-Lapse indices
            for i in non_lapse_list:
                if val_idx < len(row):
                    new_row[i] = row[val_idx]
                    val_idx += 1
        else:
            # Short row: map values to early non-Lapse indices, then place last values in final non-Lapse indices
            values_to_place = len(row)
            if values_to_place > 6:  # More than the first 6 non-Lapse indices (0,1,2,3,4,5)
                # Place the last 3 values in indices  nihilism
                for i in non_lapse_list[:6]:
                    if val_idx < values_to_place - 3:
                        new_row[i] = row[val_idx]
                        val_idx += 1
                # Place the last 3 values in the last 3 non-Lapse indices (9,10,11)
                for i in non_lapse_list[-3:]:
                    if val_idx < values_to_place:
                        new_row[i] = row[val_idx]
                        val_idx += 1
            else:
                # Very short row: fill early non-Lapse indices
                for i in non_lapse_list[:values_to_place]:
                    new_row[i] = row[val_idx]
                    val_idx += 1
        adjusted_data.append(new_row)

    table_data = adjusted_data

    print("\nRows after adjusting for Lapse alignment:")
    for r in table_data:
        print(r)

    # Step 2: Select specific columns
    selected_indices = [0, 1, 2, 3, 9, 10, 11]
    table_data = [[row[i] if i < len(row) else "_" for i in selected_indices] for row in table_data]

    print("\nRows after selecting specific columns:")
    for r in table_data:
        print(r)

    # Step 3: Filtering
    def is_invalid(row):
        count = 0
        for cell in row:
            if cell in ("", None, np.nan):
                count += 1
            elif has_english_words(cell) and cell != "_":
                count += 1
        return count >= 5

    table_data = [row for row in table_data if not is_invalid(row)]

    print("\nRows after filtering (based on empty/English word rule):")
    for r in table_data:
        print(r)

    keyword = "Tabular Detail - Non Guaranteed"
    result = [tuple(row + [keyword, page.number + 1]) for row in table_data]

    print("\nFinal extracted tuples:")
    for r in result:
        print(r)

    return result

def extract_annual_cost_summary(page):
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

    sum_indices = [3, 4, 5, 7]
    def safe_float(val):
        try:
            return float(val.replace('$', '').replace(',', '').strip())
        except (ValueError, AttributeError):
            return 0.0

    summed_values = [
        sum(safe_float(row[i]) for i in sum_indices if i < len(row))
        for row in table_data
    ]

    # Filter out zero or None sums
    summed_values = [val for val in summed_values if val not in (0.0, None, np.nan)]

    keyword = "Annual Cost Summary"
    return [(val, keyword, page.number + 1) for val in summed_values]

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
        return []
    
    selected_indices = [0,1,2,3,7,8,9]
    
    keyword = "Current Illustrated Rate*"

    results = []
    for row in table_data:
        selected = [row[i] if i < len(row) else "" for i in selected_indices]
        if all(cell.strip() for cell in selected):  # skip if the selected cell is empty
            results.append(tuple(selected + [keyword, page.number + 1]))

    return results

# --- CHATGPT ---
def extract_policy_charges_table(page, POLICY_CHARGES_HEADERS):
    import re
    import logging

    logger = logging.getLogger(__name__)

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

    lines.sort(key=lambda x: (round(x[0], 1), x[1]))

    table_data, current_row, last_y = [], [], None
    for y, x, text in lines:
        if last_y is None or abs(y - last_y) < 5:
            current_row.append((x, text))
        else:
            current_row.sort()
            row = [t for _, t in current_row if is_numeric_or_currency(t)]
            if len(row) >= 3:
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
        return []

    # 🔁 Transpose: rows -> columns
    transposed_data = list(map(list, zip(*table_data)))

    # 🔄 Reverse each row of the transposed table to make the last column the first
    reversed_data = [row[::-1] for row in transposed_data]

    # 🪵 Log the reversed table
    logger.info("Reversed Policy Charges Table:")
    for i, row in enumerate(reversed_data):
        logger.info(f"Row {i + 1}: {row}")

    # Select only the desired columns: 1, 2, 3, 4, 8, 9, 10
    selected_columns = [[row[9]] for row in reversed_data]

    # 🪵 Log the selected columns table
    logger.info("Selected Columns from Reversed Policy Charges Table:")
    for i, row in enumerate(selected_columns):
        logger.info(f"Row {i + 1}: {row}")

    return selected_columns
# --- CHATGPT ---

# ------------------- pdfplumber Flexible Logic ------------------

def extract_tables_with_flexible_headers(pdf):
    tables_by_text = {text: [] for text in PDFPLUMBER_KEYWORDS}
    policy_charges_ledger_rows = []
    policy_charges_ledger_pages = []
    
    logger = logging.getLogger(__name__)

    for page in pdf.pages:
        text = (page.extract_text() or "").lower()
        for keyword in PDFPLUMBER_KEYWORDS:
            if keyword.lower() in text:
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "text",
                    "intersection_tolerance": 8,
                    "snap_tolerance": 5,
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
                            data_rows = cleaned[5:]
                            if not data_rows:
                                logger.info(f"No data rows remain after skipping top 5 rows on page {page.page_number}")
                                continue
                        data_rows = [row[:len(headers)] + [""] * (len(headers) - len(row)) for row in data_rows]

                        selected_indices = [0, 1, 2, 3, 9, 10, 11]
                        filtered_headers = [headers[i] for i in selected_indices if i < len(headers)]
                        data_rows = [[row[i] for i in selected_indices if i < len(row)] for row in data_rows]
                        
                        # Modified: Filter rows with no empty cells and no English words
                        data_rows = [
                            row for row in data_rows
                            if all(cell not in ("", None, np.nan) for cell in row)
                        ]
                        
                        tables_by_text[keyword].extend([
                            tuple(row + [keyword, page.page_number])
                            for row in data_rows
                        ])

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

                        sum_indices = [3, 4, 5, 6]
                        def safe_float(val):
                            try:
                                val = str(val).strip()
                                if val.startswith('(') and val.endswith(')'):
                                    val = '-' + val[1:-1]
                                val = val.replace('$', '').replace(',', '')
                                return float(val)
                            except (ValueError, AttributeError):
                                return 0.0
                        
                        summed_values = [
                            sum(safe_float(row[i]) for i in sum_indices if i < len(row))
                            for row in data_rows
                        ]

                        # Modified: Filter out zero or None sums (indicating empty/invalid cells)
                        summed_values = [val for val in summed_values if val not in (0.0, None, np.nan)]
                        summed_values = [abs(val) for val in summed_values]
                        tables_by_text[keyword].extend([
                            (val, keyword, page.page_number)
                            for val in summed_values
                        ])

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

                        data_rows = cleaned[header_row_index + 1:]
                        data_rows = [row + [""] * (len(headers) - len(row)) for row in data_rows]

                        selected_indices = [0, 1, 2, 3, 7, 8, 9]
                        filtered_headers = [headers[i] for i in selected_indices if i < len(headers)]
                        data_rows = [[row[i] for i in selected_indices if i < len(row)] for row in data_rows]
                        
                        # Modified: Filter rows with no empty cells and no English words
                        data_rows = [
                            row for row in data_rows
                            if all(cell not in ("", None, np.nan) for cell in row) and
                            all(not has_english_words(cell) for cell in row)
                        ]
                        
                        tables_by_text[keyword].extend([
                            tuple(row + [keyword, page.page_number])
                            for row in data_rows
                        ])

                    elif keyword.lower() == "policy charges ledger":
                        policy_charges_ledger_rows.extend(cleaned)
                        policy_charges_ledger_pages.append(page.page_number)

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
                
                selected_indices = [9]
                filtered_headers = [headers[i] for i in selected_indices if i < len(headers)]
                data_rows = [[row[i] for i in selected_indices if i < len(row)] for row in data_rows]

                # Modified: Filter rows with no empty cells and no English words
                data_rows = [
                    row for row in data_rows
                    if all(cell not in ("", None, np.nan) for cell in row) and
                    all(not has_english_words(cell) for cell in row)
                ]
                
                keyword = "Policy Charges Ledger"
                tables_by_text[keyword].extend([
                    tuple(row + [keyword, min(policy_charges_ledger_pages)])
                    for row in data_rows
                ])

    combined_tables = {}
    for keyword in PDFPLUMBER_KEYWORDS:
        if tables_by_text[keyword]:
            # Determine the number of data columns (excluding Source_Text and Page_Number)
            num_data_columns = len(tables_by_text[keyword][0]) - 2  # Subtract 2 for metadata columns

            # Assign headers based on number of data columns
            if num_data_columns == 7:
                # Use universal header for 7-column tables
                headers = UNIVERSAL_HEADER_FOR_SEVEN_COL_TABLES
            elif num_data_columns == 1:
                # Use universal header for 1-column tables
                headers = UNIVERSAL_HEADER_FOR_ONE_COL_TABLES
            # else:
            #     # Fallback to existing headers for other cases
            #     if keyword.lower() == "your policy's current charges summary":
            #         headers = ["Total Charges Sum"]
            #     elif keyword.lower() == "policy charges ledger":
            #         headers = [filtered_headers[0] if filtered_headers else "Column_10"]
            #     else:
            #         selected_indices = {
            #             "your policy's illustrated values": [0, 1, 2, 3, 9, 10, 11],
            #             "basic ledger, non-guaranteed scenario": [0, 1, 2, 3, 7, 8, 9]
            #         }.get(keyword.lower(), [0, 1, 2, 3])
            #         header_map = {
            #             "your policy's illustrated values": ILLUSTRATED_VALUES_HEADERS,
            #             "basic ledger, non-guaranteed scenario": headers if keyword.lower() == "basic ledger, non-guaranteed scenario" else ["Year", "Age", "Premium Outlay", "Net Outlay"]
            #         }.get(keyword.lower(), ["Year", "Age", "Premium Outlay", "Net Outlay"])
            #         headers = [header_map[i] for i in selected_indices if i < len(header_map)]

            # Create DataFrame with appropriate headers plus metadata columns
            df = pd.DataFrame(
                tables_by_text[keyword],
                columns=headers + ["Source_Text", "Page_Number"]
            )
            combined_tables[keyword] = df if not df.empty else None
        else:
            combined_tables[keyword] = None

    return combined_tables

# --------------------------- Main API ---------------------------

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        pdf_file = BytesIO(content)

        results = []
        tables_by_text = {k: [] for k in PYMUPDF_KEYWORDS}

        # Modified: Extract all text for both keywords and fields
        all_text = ""
        with fitz.open(stream=pdf_file, filetype="pdf") as doc:
            for page in doc:
                all_text += page.get_text("text") + "\n"  # Added newline for better pattern matching
        
        # Modified: Extract fields (optional, may return None for missing fields)
        extracted_fields = extract_fields(all_text)

        # Modified: Check for keywords in lowercase text
        found_keywords = [k for k in PYMUPDF_KEYWORDS + PDFPLUMBER_KEYWORDS if k.lower() in all_text.lower()]

        # Modified: Return early only if no keywords or fields are found
        if not found_keywords and not any(extracted_fields.values()):
            return JSONResponse(content={"message": "No matching keywords or fields found."}, status_code=200)

        if any(k in found_keywords for k in PYMUPDF_KEYWORDS):
            pdf_file.seek(0)
            with fitz.open(stream=pdf_file, filetype="pdf") as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text("text").lower()
                    
                    if "annual cost summary" in text:
                        data = extract_annual_cost_summary(page)
                        if data:
                            tables_by_text["Annual Cost Summary"].extend(data)

                    if "tabular detail - non guaranteed" in text:
                        data = extract_tabular_detail_non_guaranteed(page)
                        if data:
                            tables_by_text["Tabular Detail - Non Guaranteed"].extend(data)
                    
                    # --- CHATGPT ---
                    if "policy charges and other expenses" in text.lower():
                        data = extract_policy_charges_table(page, POLICY_CHARGES_HEADERS)
                        if data:
                            # Validate and add metadata columns
                            data_with_metadata = [
                                row + ["Policy Charges and Other Expenses", page_num + 1]
                                for row in data if len(row) == 1
                            ]
                            if not data_with_metadata:
                                logger.warning("No valid rows with 7 columns for Policy Charges and Other Expenses")
                            else:
                                logger.info(f"Appending {len(data_with_metadata)} rows with 9 columns for Policy Charges and Other Expenses")
                                tables_by_text["Policy Charges and Other Expenses"].extend(data_with_metadata)
                    # --- CHATGPT ---

                    if "current illustrated rate*" in text.lower():
                        data = extract_current_illustrated_rate_table(page)
                        if data:
                            tables_by_text["Current Illustrated Rate*"].extend(data)

        for keyword in PYMUPDF_KEYWORDS:
            if tables_by_text[keyword]:
                # Modified: Filter rows with no empty cells and no English words
                if keyword == "Policy Charges and Other Expenses":
                    valid_rows = [
                        row for row in tables_by_text[keyword]
                        if all(cell not in ("", None, np.nan) for cell in row[:-2])  # No empty cells
                    ]
                else:
                    valid_rows = [
                        row for row in tables_by_text[keyword]
                        if all(cell not in ("", None, np.nan) for cell in row[:-2]) and  # No empty cells
                        all(not has_english_words(cell) for cell in row[:-2])  # No English words
                    ]
                if not valid_rows:
                    logger.warning(f"No valid rows for {keyword} after filtering empty cells and English words")
                    continue
                
                # --- WEDNESDAY ---
                # Determine the number of data columns (excluding Source_Text and Page_Number)
                num_data_columns = len(valid_rows[0]) - 2  # Subtract 2 for metadata columns
                # --- WEDNESDAY ---
                
                # --- WEDNESDAY ---
                # Assign headers based on number of data columns
                if num_data_columns == 7:
                    # Use universal header for 7-column tables
                    headers = UNIVERSAL_HEADER_FOR_SEVEN_COL_TABLES
                elif num_data_columns == 1:
                    # Use universal header for 1-column tables
                    headers = UNIVERSAL_HEADER_FOR_ONE_COL_TABLES
                # else:
                #     # Fallback to existing headers for other cases
                #     if keyword == "Tabular Detail - Non Guaranteed":
                #         headers = ["Total Financial Metrics Sum"]
                #     elif keyword == "Annual Cost Summary":
                #         headers = [COST_SUMMARY_HEADERS[i] for i in [0, 1, 2, 3, 9, 10, 11]]
                #     elif keyword == "Policy Charges and Other Expenses":
                #         headers = [POLICY_CHARGES_HEADERS[i] for i in [0, 1, 2, 3, 7, 8, 9]]
                #     elif keyword == "Current Illustrated Rate*":
                #         headers = [CURRENT_ILLUSTRATED_RATE_HEADERS[9]]

                # Create DataFrame with appropriate headers plus metadata columns
                df = pd.DataFrame(
                    valid_rows,
                    columns=headers + ["Source_Text", "Page_Number"]
                )
                # --- WEDNESDAY ---

                if not df.empty:
                    print(f"\nExtracted Table: {keyword} (Combined)")
                    print(df.to_string(index=False))
                    results.append({
                        "source": keyword,
                        "page": int(df["Page_Number"].min()),
                        "keyword": keyword,
                        "extractor": "PyMuPDF",
                        "data": df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
                    })

        if any(k in found_keywords for k in PDFPLUMBER_KEYWORDS):
            pdf_file.seek(0)
            with pdfplumber.open(pdf_file) as pdf:
                tables = extract_tables_with_flexible_headers(pdf)
                for source_text, df in tables.items():
                    if df is not None and not df.empty:
                        print(f"\nExtracted Table: {source_text} (Combined)")
                        print(df.to_string(index=False))
                        results.append({
                            "source": source_text,
                            "page": int(df["Page_Number"].min()),
                            "keyword": source_text,
                            "extractor": "pdfplumber",
                            "data": df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
                        })
                        
        # Modified: Include fields in response, even if empty
        response = {
            "fields": extracted_fields,
            "tables": jsonable_encoder(results)
        }
        
        # Modified: Return message if no tables or fields are extracted
        if not results and not any(extracted_fields.values()):
            return JSONResponse(content={"message": "No tables or fields extracted."}, status_code=200)

        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")