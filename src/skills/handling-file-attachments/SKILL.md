---
name: handling-file-attachments
description: Workflow for orchestrating tasks with local file attachments (PDFs, images, spreadsheets, etc.). Use when the user task references a file path or mentions attached files. Ensures absolute-path propagation and routes to `deep_analyzer_agent` first with format-appropriate extraction guidance.
metadata:
  consumer: planner
  skill_type: delegation_pattern
  source: seeded
  verified_uses: 0
  confidence: 0.8
---

# Handling File Attachments (planner workflow)

## When to activate
- The task text contains a file path, or
- The task mentions "attached", "provided file", "this document", or
- The task references a file extension such as `.pdf`, `.xlsx`, `.docx`, `.png`, `.jpg`, `.csv`.

Do NOT activate for web URLs — those are `browser_use_agent` / `deep_researcher_agent` territory.

## Workflow
1. Resolve any relative file path to an absolute path BEFORE delegation. Sub-agents run with a different working directory and relative paths fail.
2. Select the right sub-agent by file type:
   - PDF, DOCX, CSV, XLSX, JSON, TXT, MD → `deep_analyzer_agent`
   - PNG, JPG, screenshot → `deep_analyzer_agent` (it has vision support)
   - Archive (ZIP, TAR) → `deep_analyzer_agent` with explicit "extract and list contents first" instruction
3. Pass the absolute path explicitly in the delegation's `task` argument AND verbatim in the user-facing question.
4. For structured data (tables), request extraction to CSV/JSON format BEFORE reasoning — reasoning over raw PDF output is error-prone.
5. After the sub-agent responds, verify the response references specific content from the file (a direct quote, a specific value). If the response is generic ("I analyzed the file and..."), re-delegate with a more specific extraction request.

## Avoid
- Relative paths — they resolve relative to the sub-agent's CWD, not yours.
- Summarising the file before extraction completes.
- Using `browser_use_agent` for local files (it's a web tool).
- Passing the raw file binary in the task text.

## Example delegation
```json
{
  "name": "deep_analyzer_agent",
  "arguments": {
    "task": "Analyse the PDF at /absolute/path/to/report.pdf and extract the Q3 revenue figures. If the data is in a table, extract it to CSV first and then answer from the CSV."
  }
}
```
