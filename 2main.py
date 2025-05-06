import tabula
import pandas as pd

tables = tabula.read_pdf("sample-alz.pdf", pages="all", multiple_tables=True)

print(f"Number of tables extracted: {len(tables)}")

for idx, table in enumerate(tables):
      print(f"\n--- table {idx+1} ---")

      if not table.empty and isinstance(table.iloc[0, 0], str):
            print("possible table name:", " | ".join(map(str, table.iloc[0])))

      print(table.head())

# print(tables)