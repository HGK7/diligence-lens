"""
Export LegalBench-Deep sample data rooms into ./dataroom for Diligence Lens.

You download the dataset yourself, then point this script at the folder:

  python scripts/export_legalbench_deep.py --src datasets/legalbench-deep/sample_dataroom_50

Supported layouts (auto-detected):
  - **/contracts/*.{md,txt,pdf}
  - top-level *.{md,txt,pdf} under the room folder
  - manifest.json / documents.json listing paths or inline text
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


TEXT_SUFFIXES = {".md", ".txt", ".pdf"}


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE)
    return cleaned.strip("._") or "document"


def clear_dataroom(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.name.startswith("."):
            continue
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def copy_text_tree(src: Path, out_dir: Path) -> list[Path]:
    written: list[Path] = []
    candidates = list(src.rglob("*"))
    contract_dir = src / "contracts"
    if contract_dir.is_dir():
        candidates = list(contract_dir.rglob("*"))

    for path in sorted(candidates):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        # Skip nested tooling / docs inside the HF clone.
        if path.name.lower() in {"readme.md", "license", "license.md"}:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        target = out_dir / safe_name(path.name)
        if target.exists():
            target = out_dir / f"{path.stem}_{abs(hash(path.as_posix())) % 10_000}{path.suffix}"
        shutil.copy2(path, target)
        written.append(target)
    return written


def export_from_manifest(src: Path, out_dir: Path) -> list[Path]:
    written: list[Path] = []
    for name in ("manifest.json", "documents.json", "dataroom.json"):
        manifest_path = src / name
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        documents = payload
        if isinstance(payload, dict):
            documents = (
                payload.get("documents")
                or payload.get("contracts")
                or payload.get("files")
                or []
            )
        if not isinstance(documents, list):
            continue
        for index, item in enumerate(documents, start=1):
            if isinstance(item, str):
                rel = Path(item)
                source_file = (src / rel) if not rel.is_absolute() else rel
                if source_file.exists():
                    target = out_dir / safe_name(source_file.name)
                    shutil.copy2(source_file, target)
                    written.append(target)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("body") or item.get("content")
            file_name = (
                item.get("file_name")
                or item.get("filename")
                or item.get("path")
                or item.get("id")
                or f"contract_{index}"
            )
            if text:
                target = out_dir / f"{safe_name(str(file_name))}.md"
                header = (
                    f"# {file_name}\n\n"
                    f"- source: LegalBench-Deep\n"
                    f"- id: {item.get('id', index)}\n\n---\n\n"
                )
                target.write_text(header + str(text).strip() + "\n", encoding="utf-8")
                written.append(target)
            else:
                rel = item.get("path") or item.get("file")
                if rel:
                    source_file = src / rel
                    if source_file.exists():
                        target = out_dir / safe_name(source_file.name)
                        shutil.copy2(source_file, target)
                        written.append(target)
        if written:
            return written
    return written


def export_room(src: Path, out_dir: Path, replace: bool) -> list[Path]:
    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {src}")
    if replace:
        clear_dataroom(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    written = export_from_manifest(src, out_dir)
    if not written:
        written = copy_text_tree(src, out_dir)
    if not written:
        raise RuntimeError(
            "No contracts found. Expected contracts/*, *.md/*.txt/*.pdf, "
            "or a manifest.json with documents."
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to sample_dataroom_50 (or _20) after you download LegalBench-Deep",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataroom"),
        help="Diligence Lens active corpus folder",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Clear ./dataroom before export",
    )
    args = parser.parse_args()
    paths = export_room(args.src, args.out, replace=args.replace)
    print(f"Exported {len(paths)} files to {args.out.resolve()}")
    for path in paths[:10]:
        print(f"  - {path.name}")
    if len(paths) > 10:
        print(f"  ... and {len(paths) - 10} more")


if __name__ == "__main__":
    main()
