import tabula

tables = tabula.read_pdf("sample-alz.pdf", pages="all")

print(f"Number of tables extracted: {len(tables)}")

print(tables)