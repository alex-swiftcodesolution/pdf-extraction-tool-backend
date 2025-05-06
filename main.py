from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import pdfplumber
import pandas as pd
import numpy as np
from io import BytesIO
import logging
from typing import List, Dict, Any
from collections import Counter

app = FastAPI()

# CORS setup for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keywords to search
search_texts: List[str] = [
    "Your policy's illustrated values",
    "Your policy's current charges summary",
]

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)) -> JSONResponse:
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        logger.info(f"Processing uploaded file: {file.filename}")
        content: bytes = await file.read()
        pdf_file = BytesIO(content)

        tables_by_text: Dict[str, List[pd.DataFrame]] = {text: [] for text in search_texts}

        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text: str = page.extract_text() or ""
                for search_text in search_texts:
                    if search_text.lower() in page_text.lower():
                        logger.info(f"Page {page.page_number}: Found '{search_text}'")

                        tables = page.extract_tables(table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "text",
                            "intersection_tolerance": 5,
                            "snap_tolerance": 3,
                        })

                        if not tables:
                            logger.info(f"Page {page.page_number}: No tables found.")
                            continue

                        for table_idx, table in enumerate(tables, 1):
                            logger.info(f"Page {page.page_number}, Table {table_idx}: Raw table data")
                            logger.info(table)

                            cleaned_table: List[List[str]] = [
                                [str(cell).strip() if cell else "" for cell in row]
                                for row in table if any(cell and str(cell).strip() for cell in row)
                            ]

                            if not cleaned_table:
                                logger.info(f"Page {page.page_number}, Table {table_idx}: Empty table skipped.")
                                continue

                            # Try to detect header rows
                            header_keywords = {"year", "age"}
                            header_row_index = -1
                            for idx, row in enumerate(cleaned_table[:6]):
                                normalized = [cell.lower() for cell in row]
                                if any(any(keyword in cell for keyword in header_keywords) for cell in normalized):
                                    header_row_index = idx
                                    break

                            if header_row_index == -1:
                                logger.warning(f"Page {page.page_number}, Table {table_idx}: No header row found.")
                                continue

                            max_header_rows = 3
                            start_header_idx = max(0, header_row_index - (max_header_rows - 1))
                            header_rows = cleaned_table[start_header_idx:header_row_index + 1]

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
                            headers = [header if header else f"Column_{i+1}" for i, header in enumerate(headers)]

                            def deduplicate_headers(header_list):
                                counts = Counter()
                                result = []
                                for h in header_list:
                                    counts[h] += 1
                                    result.append(f"{h}_{counts[h]}" if counts[h] > 1 else h)
                                return result

                            headers = deduplicate_headers(headers)
                            logger.info(f"Page {page.page_number}, Table {table_idx}: Combined headers: {headers}")

                            data_rows = cleaned_table[header_row_index + 1:]
                            data_rows = [row + [""] * (len(headers) - len(row)) for row in data_rows]
                            df = pd.DataFrame(data_rows, columns=headers)

                            year_col = next((col for col in df.columns if "year" in col.lower()), None)
                            age_col = next((col for col in df.columns if "age" in col.lower()), None)

                            if year_col:
                                df = df[df[year_col].astype(str).str.strip() != ""]
                                logger.info(f"Filtered rows with empty '{year_col}'.")

                            if age_col:
                                df = df[df[age_col].astype(str).str.strip() != ""]
                                logger.info(f"Filtered rows with empty '{age_col}'.")

                            if df.empty:
                                logger.info(f"Page {page.page_number}, Table {table_idx}: No valid data after filtering.")
                                continue

                            df["Source_Text"] = search_text
                            df["Page_Number"] = page.page_number
                            tables_by_text[search_text].append(df)

        all_tables = []
        for search_text, df_list in tables_by_text.items():
            if df_list:
                combined_df = pd.concat(df_list, ignore_index=True)

                # Replace NaN, inf, -inf with None for JSON compatibility
                combined_df = combined_df.replace([np.nan, np.inf, -np.inf], None)

                all_tables.append({
                    "source_text": search_text,
                    "page_number": int(combined_df["Page_Number"].iloc[0]),
                    "data": combined_df.to_dict(orient="records")
                })

        if not all_tables:
            logger.info("No tables found in the PDF.")
            return JSONResponse(content={"message": "No tables found in the PDF."}, status_code=200)

        logger.info(f"Returning {len(all_tables)} combined table groups.")
        return JSONResponse(content={"tables": jsonable_encoder(all_tables)}, status_code=200)

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
