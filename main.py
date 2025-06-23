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
    "Policy Year","Age","Premium Outlay","Net Income","Cash Value","Surrender Value","Death Benefit"
]
UNIVERSAL_HEADER_FOR_ONE_COL_TABLES = [
    "Charges"
]

HEADER_FOR_ALZ_TABLE = [
    "Age","Policy Year","Premium Outlay","Net Income","Cash Value","Surrender Value","Death Benefit"
] 
HEADER_FOR_LSW_TABLE = [
    "Policy Year","Age","Premium Outlay","Net Income","Cash Value","Surrender Value","Death Benefit"
] 
HEADER_FOR_MN_TABLE = [
    "Policy Year","Age","Premium Outlay","Net Income","Cash Value","Surrender Value","Death Benefit"
] 
HEADER_FOR_NW_TABLE = [
    "Policy Year","Age","Premium Outlay","Net Income","Cash Value","Surrender Value","Death Benefit"
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
    "Year","Age","Premium Outlay","Net Income",
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


def extract_filename(filename):
    return filename

# Added: Function to extract fields (illustration_date, insured_name, etc.)
def extract_fields(pdf_text, filename):
    """
    Extracts specified fields from PDF text, with assumed_ror conditioned on filename keywords.
    For ALZ files, extracts assumed_ror from the line below the first occurrence of
    'current scenario indexed interest rate'.
    Args:
        pdf_text: The text extracted from the PDF.
        filename: The name of the uploaded PDF file.
    Returns:
        A dictionary with field values or None if not found.
    """
    fields = {
        "illustration_date": None,
        "insured_name": None,
        "initial_death_benefit": None,
        "assumed_ror": None,
        "minimum_initial_pmt": None
    }
    
    pages = pdf_text.split("page ")[1:]
    page_1_text = pages[0].lower() if pages else text.lower()
    
    # Normalize text for case-insensitive matching
    text = pdf_text.lower()
    filename = filename.lower()
    

    # Define default pattern for assumed_ror
    default_ror_pattern = r"assumed\s*ror[:\s]*([\d.]+%)"

    # Define keyword-based patterns for assumed_ror
    ror_patterns = {
        "nationwide": r"assumed\s*[:\s]*([\d.]+%)",  # NW-specific pattern
        "lsw": r"illustrated\s*rate[:\s]*([\d.]+%)",  # LSW-specific pattern
        "mn": r"crediting\s*rate[:\s]*([\d.]+%)"  # MN-specific pattern
    }

    mn_ror_pattern = r"using\s*([\d.]+)%\s*illustrated\s*crediting\s*rate\s*and\s*current\s*charges"
    
    nw_ror_pattern = r"(?:indexed\s*interest|assumed|illustrated\s*rate)\s*[\n\r\s]*([\d.]+%)"
        
    # Handle assumed_ror for ALZ files
    if "alz" in filename:
        lines = pdf_text.splitlines()
        target_text = "Indexed interest rates"
        found = False
        for i, line in enumerate(lines):
            if target_text.lower() in line.lower():
                found = True
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    match = re.search(r"(\d+\.?\d*%)", next_line)
                    if match:
                        fields["assumed_ror"] = match.group(1)
                        break
                    
    elif "mnl" in filename:
        ror_match = re.search(mn_ror_pattern, text, re.IGNORECASE | re.DOTALL)
        if ror_match:
            fields["assumed_ror"] = f"{ror_match.group(1)}%"
        else:
            # Fallback to the existing MN pattern if the primary pattern fails
            ror_match = re.search(ror_patterns["mn"], text, re.IGNORECASE)
            if ror_match:
                fields["assumed_ror"] = ror_match.group(1).strip()
                
    elif "nationwide" in filename:
    # Search for percentage near 'indexed interest', 'assumed', or 'illustrated rate'
        ror_match = re.search(nw_ror_pattern, text, re.IGNORECASE | re.DOTALL)
        if ror_match:
            fields["assumed_ror"] = ror_match.group(1).strip()
            # Debug: Log the match
            print(f"Nationwide assumed_ror matched: {ror_match.group(1)}")
        else:
            # Fallback to broader pattern
            ror_match = re.search(r"assumed\s*[:\s]*([\d.]+%)", text, re.IGNORECASE)
            if ror_match:
                fields["assumed_ror"] = ror_match.group(1).strip()
                print(f"Nationwide fallback assumed_ror: {ror_match.group(1)}")
            else:
                print("Nationwide assumed_ror not found")
            
    else:
        # Select pattern for non-ALZ files
        selected_ror_pattern = default_ror_pattern
        for keyword, pattern in ror_patterns.items():
            if keyword in filename:
                selected_ror_pattern = pattern
                break
        ror_pattern = selected_ror_pattern
        match = re.search(ror_pattern, text, re.IGNORECASE)
        if match:
            fields["assumed_ror"] = match.group(1).strip()

    # Define patterns for other fields
    patterns = {
        "illustration_date": (
            r"(?:\billustration\s*date\b|prepared\s*on\b|issued\s*on\b|date\b)[:\s]*"
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"  # MM/DD/YYYY or MM-DD-YYYY (e.g., 3/21/2025)
            r"|\w+\s+\d{1,2},\s+\d{4}"         # Month DD, YYYY (e.g., March 21, 2025)
            r"|\d{1,2}\s+\w+\s+\d{4}"          # DD Month YYYY (e.g., 21 March 2025)
            r"|\d{4}-\d{2}-\d{2}"              # YYYY-MM-DD (e.g., 2025-03-21)
            r"|\d{1,2}/\d{1,2}/\d{2})"         # MM/DD/YY (e.g., 03/21/25)
        ),
        
        # "insured_name": r"(?:prepared\s*for|for)\s*:\s*\n*([A-Za-z][^\n:]{2,50})",
        # "insured_name": r"(?:prepared\s*for|for)\s*:?\s*\n*([A-Za-z][^\n:]{2,50})",
        # "insured_name": r"(?:prepared\s*for|for)\s*:?\s*\n*([A-Za-z][^\n]{2,50})(?=\n|$)",
        "insured_name": r"(?:prepared\s*for|for)\s*:?\s*\n*\s*([A-Za-z][^\n]{2,50})(?=\n|$)",
                
        "initial_death_benefit": r"initial\s*death\s*benefit[:\s]*[\$]?([\d,]+\.?\d*)",
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

def extract_policy_charges_table(page, POLICY_CHARGES_HEADERS):
    """
    Extracts the 9th column from a table on a PDF page containing the keyword
    "Policy Charges and Other Expenses" after transposing and reversing the table.
    Skips rows where the 9th column is empty.
    
    Args:
        page: PyMuPDF page object to process
        POLICY_CHARGES_HEADERS: List of header names for the output table
    Returns:
        List of single-column rows (column 9) with non-empty cells, or empty list if no data
    """

    # Step 1: Check if the keyword "Policy Charges and Other Expenses" is on the page
    keyword = "Policy Charges and Other Expenses"
    page_text = page.get_text().lower()
    if keyword.lower() not in page_text:    
        return []  # Skip page if keyword is missing
    
    # Step 2: Define helper function to identify numeric or currency values
    def is_numeric_or_currency(text):
        """
        Checks if text is a numeric or currency value (e.g., 123, $1,234.56, -45.67).
        Args:
            text: String to check
        Returns:
            Boolean indicating if the text matches the pattern
        """
        return bool(re.match(r'^-?\$?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$', text))

    # Step 3: Extract text spans within y-coordinate range (20 to 1000)
    lines = []
    blocks = page.get_text("dict")["blocks"]  # Get structured text blocks

    for block in blocks:
        for line in block.get("lines", []):
            for span in line["spans"]:
                y = span["bbox"][1]  # Get y-coordinate of the text span
                if 20 < y < 1000:  # Filter spans within vertical range
                    # Store (y-coordinate, x-coordinate, text) for sorting
                    lines.append((y, span["bbox"][0], span["text"].strip()))

    if not lines:
        return []  # Return empty if no text found

    # Step 4: Sort lines by y-coordinate (rounded to 1 decimal) then x-coordinate
    # This groups text into rows and orders columns left-to-right
    lines.sort(key=lambda x: (round(x[0], 1), x[1]))

    # Step 5: Group lines into rows based on y-coordinate proximity
    table_data, current_row, last_y = [], [], None
    for y, x, text in lines:
        # If no previous y or y is within 18 units, add to current row
        if last_y is None or abs(y - last_y) < 18:
            current_row.append((x, text))
        else:
            # Sort current row by x-coordinate to ensure left-to-right order
            current_row.sort()
            # Extract all values, preserving empty or non-numeric cells
            all_values = [t for _, t in current_row]
            # Include row if it has at least 3 numeric values
            row_values = [t if is_numeric_or_currency(t) else "" for t in all_values]
            if sum(1 for t in row_values if t) >= 3:  # At least 3 non-empty numeric values
                table_data.append(row_values)
            current_row = [(x, text)]  # Start new row
        last_y = y

    # Process the last row
    if current_row:
        current_row.sort()
        all_values = [t for _, t in current_row]
        row_values = [t if is_numeric_or_currency(t) else "" for t in all_values]
        if sum(1 for t in row_values if t) >= 3:
            table_data.append(row_values)

    if not table_data:
        return []  # Return empty if no valid rows

    # Step 6: Pad rows to ensure consistent length for transposition
    # Add empty strings to shorter rows to match the longest row
    max_len = max(len(row) for row in table_data)
    table_data = [row + [""] * (max_len - len(row)) for row in table_data]

    # Step 7: Transpose the table (rows become columns, columns become rows)
    transposed_data = list(map(list, zip(*table_data)))

    # Step 8: Reverse each row to make the last column the first
    reversed_data = [row[::-1] for row in transposed_data]
    for i, row in enumerate(reversed_data):
        logger.info(f"Row {i + 1}: {row}")

    # Step 9: Clean rows by excluding those with exactly 10 non-empty cells
    # Keep empty cells as they are
    cleaned_rows = [row for row in reversed_data if len([cell for cell in row if cell.strip() or cell == "0"]) != 10]
    for i, row in enumerate(cleaned_rows):
        logger.info(f"Row {i + 1}: {row}")

    # Step 10: Select the 9th column (index 8) from rows with sufficient length, skipping empty cells
    selected_columns = [[row[8]] for row in cleaned_rows if len(row) > 8 and (row[8].strip() or row[8] == "0")]
    for i, row in enumerate(selected_columns):
        logger.info(f"Row {i + 1}: {row}")

    # Step 11: Return the single-column table
    return selected_columns

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
                            data_rows = cleaned
                        else:
                            data_rows = cleaned[5:]
                            if not data_rows:
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
                        # Filter out rows with any empty cells (excluding 0)
                        cleaned = [
                            row for row in cleaned 
                            if all(
                                cell is not None and str(cell).strip() != "" 
                                for cell in row
                            )
                        ]
                        if len(cleaned) < 1:
                            data_rows = cleaned
                        else:
                            data_rows = cleaned[1:]
                            if not data_rows:
                                continue
                        data_rows = [row[:len(headers)] + [""] * (len(headers) - len(row)) for row in data_rows]
                        
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
            if keyword.lower() == "basic ledger, non-guaranteed scenario":
                headers = HEADER_FOR_ALZ_TABLE
            elif num_data_columns == 7:
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
            #             "basic ledger, non-guaranteed scenario": headers if keyword.lower() == "basic ledger, non-guaranteed scenario" else ["Year", "Age", "Premium Outlay", "Net Income"]
            #         }.get(keyword.lower(), ["Year", "Age", "Premium Outlay", "Net Income"])
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
        extracted_fields = extract_fields(all_text, file.filename)

        # Modified: Check for keywords in lowercase text
        found_keywords = [k for k in PYMUPDF_KEYWORDS + PDFPLUMBER_KEYWORDS if k.lower() in all_text.lower()]

        # Modified: Return early only if no keywords or fields are found
        if not found_keywords and not any(extracted_fields.values()):
            return JSONResponse(content={"message": "No matching keywords or fields found."}, status_code=200)

        if any(k in found_keywords for k in PYMUPDF_KEYWORDS):
            pdf_file.seek(0)
            with fitz.open(stream=pdf_file, filetype="pdf") as doc:
                policy_charges_rows = []
                
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
                            policy_charges_rows.extend(data)
                    # --- CHATGPT ---

                    if "current illustrated rate*" in text.lower():
                        data = extract_current_illustrated_rate_table(page)
                        if data:
                            tables_by_text["Current Illustrated Rate*"].extend(data)

                # Process Policy Charges and Other Expenses rows
                if policy_charges_rows:
                    # Skip every 6th row
                    filtered_policy_charges_rows = [row for i, row in enumerate(policy_charges_rows, 1) if i % 6 != 0]
                    
                    # Add metadata to filtered rows (assuming page number from first page with data)
                    # Note: If you need specific page numbers, you may need to track pages during collection
                    data_with_metadata = [
                        row + ["Policy Charges and Other Expenses", min(page_num + 1 for page_num, page in enumerate(doc) if "policy charges and other expenses" in page.get_text("text").lower())]
                        for row in filtered_policy_charges_rows if len(row) == 1
                    ]
                    
                    if not data_with_metadata:
                        logger.warning("No valid rows with 1 column for Policy Charges and Other Expenses after filtering")
                    else:
                        tables_by_text["Policy Charges and Other Expenses"].extend(data_with_metadata)
                    
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