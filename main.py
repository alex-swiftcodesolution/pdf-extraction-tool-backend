from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tabula
import pandas as pd
import os
import tempfile
import shutil

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract-tables")
async def extract_tables(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Extract tables using tabula
        tables = tabula.read_pdf(tmp_path, pages='all', multiple_tables=True)

        result = []
        for idx, df in enumerate(tables):
            df: pd.DataFrame
            table_json = {
                "name": f"Table {idx+1}",
                "columns": list(df.columns.astype(str)),
                "data": df.fillna("").astype(str).to_dict(orient="records"),
            }
            result.append(table_json)

        # Clean up
        os.remove(tmp_path)

        return { "tables": result }

    except Exception as e:
        return { "error": str(e) }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
