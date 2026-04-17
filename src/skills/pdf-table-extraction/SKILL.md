---
name: pdf-table-extraction
description: Extract and reason over PDF tables using deep_analyzer_tool first, then python_interpreter_tool only with its allowed import set (math, re, statistics, etc.). Use when a PDF task mentions tables, rows/columns, or numeric data where cell alignment matters. Avoids misreading linearized PDF text.
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

## Constraints (read first)
- **`python_interpreter_tool` allowlist:** only these imports work by default: `collections`, `datetime`, `itertools`, `math`, `queue`, `random`, `re`, `stat`, `statistics`, `time`, `unicodedata`. There is **no** `pandas`, `pdfplumber`, `pdf2image`, or `pytesseract` in the sandbox unless an operator extended the config.
- **Primary extraction** must go through **`deep_analyzer_tool`**, which reads the PDF via the stack’s converter (same path as file ingestion) and reasons over the extracted text.

## Workflow
1. Call **`deep_analyzer_tool`** with `source` set to the **absolute** PDF path and a `task` that forces structured output, for example:
   - “Extract the table on page N as a **markdown table** with header row.”
   - “List every row of the revenue table as **pipe-separated** values (no prose between rows).”
   - “Output **one JSON object per table row** with keys taken from the header.”
2. If you need programmatic checks on that structured text, call **`python_interpreter_tool`** using **only** the allowed imports — e.g. parse pipe- or tab-separated lines with `re.split`, then aggregate with `statistics.mean` / `sum` / `math.fsum`.
3. If columns are still misaligned, **re-call `deep_analyzer_tool`** with stricter instructions (name the columns you expect, ask for TSV only, or ask for “(row label, column name, cell value)” triples).

## If extraction is still poor
- **Scanned PDFs:** rely on `deep_analyzer_tool` with an explicit OCR-oriented task (“treat pages as images; transcribe the printed table”). Do not assume OCR libraries in the interpreter.
- **Born-digital but ugly layout:** repeat `deep_analyzer_tool` with page-specific instructions rather than regexing raw stream text.

## Avoid
- `import pandas`, `import pdfplumber`, or any other module outside the interpreter allowlist — the call will fail.
- Reasoning over **raw linearized PDF text** when the answer depends on which number belongs to which column.
- Using regex alone to “reconstruct” a table from badly spaced text when another `deep_analyzer_tool` pass with clearer output shape would work better.

## Verification
Recompute any reported figure inside **`python_interpreter_tool`** from the numeric literals you extracted (show prints). If you cannot parse the model’s table output reliably, fix the **`deep_analyzer_tool`** prompt before trusting the number.
