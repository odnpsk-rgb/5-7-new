"""Извлечение текста из PDF в txt для RAG."""

import re
from pathlib import Path

import fitz

DOCS = Path(__file__).parent / "docs"


def is_page_number(line: str) -> bool:
    s = line.strip()
    return bool(re.fullmatch(r"[\dIVXLC]+", s)) and len(s) <= 6


def is_toc_dots(line: str) -> bool:
    return line.count(".") > 10 and len(line) > 40


def should_merge(prev: str, curr: str) -> bool:
    if not prev or not curr:
        return False
    if is_page_number(curr) or is_page_number(prev):
        return False
    if is_toc_dots(curr) or is_toc_dots(prev):
        return False
    if re.match(
        r"^(\d+\.?\d*\s|[А-ЯA-Z]{2,}\s*$|Приложение\s|Таблица\s|СП\s|СНиП\s)",
        curr,
    ):
        return False
    if curr[0].islower() or curr[0] in ",.;:-":
        return True
    if prev[-1] not in ".!?:;»\"'":
        return True
    return False


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    raw_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    merged = []
    buf = ""
    for line in raw_lines:
        if not line:
            if buf:
                merged.append(buf)
                buf = ""
            merged.append("")
            continue
        if not buf:
            buf = line
        elif should_merge(buf, line):
            buf = buf + " " + line
        else:
            merged.append(buf)
            buf = line
    if buf:
        merged.append(buf)

    result = [line for line in merged if not is_page_number(line)]
    text = "\n".join(result)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    parts = [page.get_text("text") for page in doc]
    doc.close()
    return clean_text("\n".join(parts))


def main() -> None:
    for pdf_path in sorted(DOCS.glob("*.pdf")):
        cleaned = extract_pdf(pdf_path)
        out_path = pdf_path.with_suffix(".txt")
        out_path.write_text(cleaned, encoding="utf-8")
        print(
            f"{pdf_path.name} -> {out_path.name}: "
            f"{len(cleaned)} символов, {len(cleaned.splitlines())} строк"
        )


if __name__ == "__main__":
    main()
