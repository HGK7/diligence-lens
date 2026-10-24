"""Export ContractNLI document texts into vendor_docs for Atlas Lens indexing."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE)
    return cleaned.strip("._") or "document"


def export_docs(
    json_path: Path,
    out_dir: Path,
    limit: int,
    prefix: str = "cnli",
) -> list[Path]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    documents = payload["documents"]
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for document in documents[:limit]:
        doc_id = document["id"]
        file_name = safe_name(document.get("file_name") or f"doc-{doc_id}")
        target = out_dir / f"{prefix}_{doc_id}_{file_name}.md"
        header = (
            f"# ContractNLI document {doc_id}\n\n"
            f"- source_file: {document.get('file_name')}\n"
            f"- document_type: {document.get('document_type')}\n"
            f"- url: {document.get('url')}\n\n"
            f"---\n\n"
        )
        body = document["text"].strip() + "\n"
        target.write_text(header + body, encoding="utf-8")
        written.append(target)
    return written


def print_hypotheses(json_path: Path) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    print("\nContractNLI hypotheses (use these as portal questions):\n")
    for key, meta in payload["labels"].items():
        print(f"{key}: {meta['short_description']}")
        print(f"  {meta['hypothesis']}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("vendor_docs/contract-nli/contract-nli/test.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("vendor_docs"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--prefix", type=str, default="cnli")
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Print the 17 hypotheses and exit",
    )
    args = parser.parse_args()

    if args.show_labels:
        print_hypotheses(args.json)
        return

    paths = export_docs(args.json, args.out, args.limit, args.prefix)
    print(f"Wrote {len(paths)} files to {args.out.resolve()}")
    for path in paths:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()