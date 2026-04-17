---
name: pdf-table-extraction
description: Extract tables from PDF documents and reason over them by converting to CSV first. Use when a PDF task mentions tables, rows/columns, percentages, or numeric data arranged spatially. Avoids the common failure mode of misreading cell boundaries when reasoning over raw PDF text output.
metadata:
  consumer: deep_analyzer_agent
  skill_type: tool_usage
  source: seeded
  verified_uses: 0
  confidence: 0.8
---

# PDF Table Extraction (deep_analyzer_agent workflow)

## When to activate
- Task involves a PDF file AND references tabular data (rows, columns, percentages, financial figures, comparisons).
- You already received generic text output from the PDF but cannot align specific numbers to rows/columns.

## Workflow
1. Use `python_interpreter_tool` with `pdfplumber` to extract tables directly:
   ```python
   import pdfplumber
   with pdfplumber.open("/absolute/path/to/file.pdf") as pdf:
       for i, page in enumerate(pdf.pages):
           for j, table in enumerate(page.extract_tables() or []):
               print(f"--- page {i+1} table {j+1} ---")
               for row in table:
                   print(row)
   ```
2. Inspect the output. If rows/columns are misaligned, try `page.extract_table(table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"})`.
3. Convert the chosen table to a pandas DataFrame for row/column querying:
   ```python
   import pandas as pd
   df = pd.DataFrame(table[1:], columns=table[0])
   ```
4. Answer the question by filtering/aggregating the DataFrame (`df.query`, `df.groupby`, `df[col].sum()`, etc.) rather than by eyeballing the raw extraction.

## If pdfplumber fails
- For scanned PDFs (images of pages), switch to OCR: try `pdf2image` + `pytesseract`.
- For born-digital PDFs with unusual table formatting, fall back to `deep_analyzer_tool` with an explicit instruction ("The PDF has non-standard tables; extract the data in the table on page N as a list of (column, value) pairs").

## Avoid
- Reasoning over raw `pdfminer.extract_text()` output when the answer depends on table alignment — cell boundaries are frequently wrong.
- Using regex to "parse" table text into structure — spacing is unreliable across PDFs.
- Reporting a number without having loaded it into python first (no way to verify).

## Verification
Before returning, re-compute the key numbers in python and print them. If the user asked for "Q3 revenue", your final answer should be preceded by python output like `print(df.loc[df["Quarter"]=="Q3", "Revenue"].iloc[0])` — that gives an audit trail the manager can check.
