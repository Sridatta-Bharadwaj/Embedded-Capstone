"""Render E0_Project_Report.docx to PDF with LibreOffice and record the printed page number of every heading,
table caption and figure caption (outline.json) in toc_pages.json, for the static contents lists in front.js.

  python make_toc.py <path-to-soffice.com>        (then: node build.js, and run again until "stable")
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCX, PDF = HERE / "E0_Project_Report.docx", HERE / "E0_Project_Report.pdf"
FRONT_LISTS = ["LIST OF TABLES", "LIST OF FIGURES", "LIST OF SYMBOLS AND ABBREVIATIONS"]


def norm(s):
    return re.sub(r"[^0-9a-z]", "", s.lower())


def roman(n):
    out = ""
    for v, r in [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]:
        while n >= v:
            out, n = out + r, n - v
    return out


def main():
    soffice = sys.argv[1]
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(HERE), str(DOCX)], check=True,
                   capture_output=True)
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True).stdout)[1])
    texts = [subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", str(PDF), "-"], capture_output=True,
                            text=True, encoding="utf-8").stdout for i in range(1, n + 1)]
    first_lines = [next((l.strip() for l in t.splitlines() if l.strip()), "") for t in texts]
    flat = [norm(t) for t in texts]

    pages = {}
    for title in FRONT_LISTS:
        idx = next(i for i, l in enumerate(first_lines) if norm(l) == norm(title))
        pages[title] = roman(idx + 1)

    outline = json.loads((HERE / "outline.json").read_text(encoding="utf-8"))
    body_start = next(i for i, l in enumerate(first_lines) if norm(l) == norm(outline[0]["text"]))
    cursor, missing = body_start, []
    for item in outline:
        key = norm(item["text"])
        hit = next((i for i in range(cursor, n) if key in flat[i]), None)
        if hit is None:
            missing.append(item["text"])
            continue
        cursor = hit
        pages[item["text"]] = str(hit - body_start + 1)

    old = HERE / "toc_pages.json"
    stable = old.exists() and json.loads(old.read_text(encoding="utf-8")) == pages
    old.write_text(json.dumps(pages, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{n} PDF pages, body starts at PDF page {body_start + 1}, {len(pages)} entries,"
          f" missing {missing}, {'stable' if stable else 'changed'}")


if __name__ == "__main__":
    main()
