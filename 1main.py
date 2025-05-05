from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import pandas as pd
from typing import List, Dict
import logging
import os
import tempfile
import numpy as np
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define your column headers
columns = [
    "Year", "Age", "Premium outlay", "Net outlay",
    "Surrender value (2%)", "Death benefit (2%)",
    "Cash value (4%)", "Surrender value (4%)", "Death benefit (4%)",
    "Cash value (6.65%)", "Surrender value (6.65%)", "Death benefit (6.65%)"
]

# Define a simplified version of the same structure to match the table
expected_header = [
    "Year", "Age", "Premium outlay", "Net outlay",
    "Surrender value", "Death benefit",
    "Cash value", "Surrender value", "Death benefit",
    "Cash value", "Surrender value", "Death benefit"
]

def normalize_header(header_row):
    return [str(cell).strip().lower().replace(" ", "") if cell else "" for cell in header_row]

# Clean individual cell
def clean_cell(value, row_idx: int, col_idx: int) -> str:
    try:
        if value is None:
            return ""
        value_str = str(value).strip()
        value_str = value_str.replace("$", "").replace(",", "")
        if value_str.startswith("(") and value_str.endswith(")"):
            value_str = "-" + value_str[1:-1]
        return value_str
    except Exception as e:
        logger.error(f"Error cleaning cell at Row {row_idx}, Col {col_idx}: Value='{value}', Error={str(e)}")
        raise

# Extract specific illustrated values table
def extract_supplemental_tables(pdf_path: str) -> List[Dict]:
    table_data = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                logger.info(f"Processing page {page_num + 1}")
                tables = page.extract_tables()

                if not tables:
                    logger.warning(f"No tables found on page {page_num + 1}")
                    continue

                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue

                    header = table[0]
                    header_str = " ".join(cell.strip().lower() for cell in header if cell)

                    # Match only if it's our specific table by checking keywords in header
                    if "year" in header_str and "premium" in header_str and "death benefit" in header_str:
                        logger.info(f"✅ Target table matched on page {page_num + 1}, table {table_idx + 1}")
                        for row_idx, row in enumerate(table[1:], start=1):
                            if len(row) >= 13:
                                try:
                                    cleaned_row = [
                                        clean_cell(cell, row_idx, col_idx)
                                        for col_idx, cell in enumerate(row[:13])
                                    ]
                                    table_data.append(cleaned_row)
                                except Exception as e:
                                    logger.error(f"Error processing row {row_idx} on page {page_num + 1}: {str(e)}")
                                    continue
                            else:
                                logger.warning(f"Skipping row {row_idx} on page {page_num + 1}: Expected 13 columns, got {len(row)}")

                        # Only extract the first matching table
                        break

        if not table_data:
            logger.warning("No valid table data extracted from the PDF")
            return []

        df = pd.DataFrame(table_data, columns=columns)
        for col in columns[2:]:  # convert numeric fields
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.replace({np.nan: None})
        table_dict = df.to_dict(orient="records")

        tables = [
            {
                "name": "Illustrated Policy Values Table",
                "columns": columns,
                "data": table_dict,  # or table_data if you want raw
            }
        ]

        logger.info(f"✅ Extracted {len(table_dict)} rows from the PDF")
        return tables

    except Exception as e:
        logger.error(f"Error extracting tables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/extract-tables")
async def extract_tables(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        tables = extract_supplemental_tables(temp_file_path)
        os.remove(temp_file_path)

        return JSONResponse(content={"tables": tables})

    except Exception as e:
        logger.error(f"Error in extract_tables endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
