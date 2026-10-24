"""
Export CUAD contracts into ./dataroom for Diligence Lens.

By default this exports the **full** corpus from GitHub `data.zip` JSON
(`datasets/cuad/data/CUADv1.json` = 510 contracts). Indexing can still be
scaled later inside the app; the dataroom holds the complete text dump.

  python scripts/export_cuad.py --replace

Optional subset (only if you explicitly want a slice):

  python scripts/export_cuad.py --limit 10 --replace

CUAD is CC BY 4.0. Keep docs/ATTRIBUTION.md when redistributing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path


DEFAULT_JSON_CANDIDATES = [
    Path("datasets/cuad/data/CUADv1.json"),
    Path("datasets/cuad/CUADv1.json"),
    Path("datasets/cuad/data/cuad_v1.json"),
]

DEFAULT_TXT_CANDIDATES = [
    Path("datasets/cuad/CUAD_v1/full_contract_txt"),
    Path("datasets/cuad/full_contract_txt"),
    Path("datasets/cuad/data/full_contract_txt"),
]


def safe_stem(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE)
    return cleaned.strip("._")[:120] or "contract"


def clear_dataroom(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.name.startswith("."):
            continue
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def find_squad_json(explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.is_file() and explicit.suffix.lower() == ".json":
        return explicit
    for candidate in DEFAULT_JSON_CANDIDATES:
        if candidate.is_file():
            return candidate
    root = Path("datasets/cuad")
    if root.exists():
        for path in sorted(root.rglob("*.json")):
            if path.name.lower().startswith("cuad") and path.stat().st_size > 1_000_000:
                return path
    return None


def find_txt_root(explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.is_dir():
        return explicit
    for candidate in DEFAULT_TXT_CANDIDATES:
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    root = Path("datasets/cuad")
    if not root.exists():
        return None
    txts = [p for p in root.rglob("*.txt") if p.is_file()]
    if not txts:
        return None
    for path in txts:
        if "full_contract_txt" in path.parts:
            return path.parent
    counts: dict[Path, int] = {}
    for path in txts:
        counts[path.parent] = counts.get(path.parent, 0) + 1
    return max(counts, key=counts.get)


def load_contracts_from_json(json_path: Path) -> list[tuple[str, str]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    entries = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise RuntimeError(f"Unexpected JSON shape in {json_path}")

    contracts: list[tuple[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or f"contract_{len(contracts)+1}")
        paragraphs = entry.get("paragraphs") or []
        contexts = [
            str(para.get("context") or "").strip()
            for para in paragraphs
            if isinstance(para, dict)
        ]
        text = "\n\n".join(part for part in contexts if part)
        if text:
            contracts.append((title, text))
    if not contracts:
        raise RuntimeError(f"No contract contexts found in {json_path}")
    return contracts


def load_contracts_from_txt(src_root: Path) -> list[tuple[str, Path]]:
    files = sorted(
        path
        for path in src_root.rglob("*.txt")
        if path.is_file() and path.name.lower() not in {"readme.txt", "license.txt"}
    )
    if not files:
        raise RuntimeError(f"No .txt contracts under {src_root}")
    return [(path.name, path) for path in files]


def write_manifest(
    out_dir: Path,
    *,
    source: str,
    seed: int | None,
    shuffle: bool,
    limit: int | None,
    available: int,
    exported: list[str],
) -> None:
    manifest = {
        "dataset": "CUAD (Contract Understanding Atticus Dataset)",
        "source": source,
        "license": "CC BY 4.0",
        "attribution": "The Atticus Project; Hendrycks et al., NeurIPS 2021",
        "seed": seed,
        "shuffle": shuffle,
        "limit": limit if limit is not None else "all",
        "available_contracts": available,
        "exported_count": len(exported),
        "exported": exported,
        "citation": (
            "Hendrycks, Burns, Chen, Ball. CUAD: An Expert-Annotated NLP "
            "Dataset for Legal Contract Review. NeurIPS 2021. "
            "https://arxiv.org/abs/2103.06268"
        ),
        "homepage": "https://www.atticusprojectai.org/cuad/",
        "github": "https://github.com/The-Atticus-Project/cuad",
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def export_subset(
    *,
    json_path: Path | None,
    txt_root: Path | None,
    out_dir: Path,
    limit: int | None,
    seed: int,
    replace: bool,
    shuffle: bool,
) -> list[Path]:
    if replace:
        clear_dataroom(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    if json_path is not None:
        contracts = load_contracts_from_json(json_path)
        ordered = list(contracts)
        if shuffle:
            random.Random(seed).shuffle(ordered)
        selected = ordered if limit is None else ordered[:limit]
        for title, text in selected:
            digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
            filename = f"cuad_{digest}_{safe_stem(title)}.txt"
            target = out_dir / filename
            header = (
                f"Title: {title}\n"
                f"Source: CUAD / {json_path.as_posix()}\n"
                f"License: CC BY 4.0 (The Atticus Project)\n"
                f"{'=' * 72}\n\n"
            )
            target.write_text(header + text.strip() + "\n", encoding="utf-8")
            written.append(target)
        write_manifest(
            out_dir,
            source=str(json_path.resolve()),
            seed=seed if shuffle else None,
            shuffle=shuffle,
            limit=limit,
            available=len(contracts),
            exported=[path.name for path in written],
        )
        return written

    if txt_root is None:
        raise FileNotFoundError(
            "No CUAD source found. Expected datasets/cuad/data/CUADv1.json "
            "(from GitHub data.zip) or a full_contract_txt folder."
        )

    contracts = load_contracts_from_txt(txt_root)
    ordered = list(contracts)
    if shuffle:
        random.Random(seed).shuffle(ordered)
    selected = ordered if limit is None else ordered[:limit]
    for name, path in selected:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        target = out_dir / f"cuad_{digest}_{safe_stem(Path(name).stem)}.txt"
        shutil.copy2(path, target)
        written.append(target)
    write_manifest(
        out_dir,
        source=str(txt_root.resolve()),
        seed=seed if shuffle else None,
        shuffle=shuffle,
        limit=limit,
        available=len(contracts),
        exported=[path.name for path in written],
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Optional path to CUADv1.json or a folder of .txt contracts",
    )
    parser.add_argument("--out", type=Path, default=Path("dataroom"))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap. Omit (default) to export the full CUAD set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle before applying --limit (ignored when exporting all)",
    )
    args = parser.parse_args()

    json_path: Path | None = None
    txt_root: Path | None = None
    if args.src is not None:
        if args.src.is_file() and args.src.suffix.lower() == ".json":
            json_path = args.src
        elif args.src.is_dir():
            txt_root = args.src
        else:
            raise FileNotFoundError(f"Source not found: {args.src}")
    else:
        json_path = find_squad_json(None)
        if json_path is None:
            txt_root = find_txt_root(None)

    # Full export: no shuffle unless user asked for a limited sample.
    shuffle = bool(args.shuffle and args.limit is not None)

    written = export_subset(
        json_path=json_path,
        txt_root=txt_root,
        out_dir=args.out,
        limit=args.limit,
        seed=args.seed,
        replace=args.replace,
        shuffle=shuffle,
    )
    source = json_path or txt_root
    print(f"Source: {source.resolve() if source else 'unknown'}")
    print(f"Mode: {'SQuAD JSON' if json_path else 'TXT folder'}")
    requested = "all" if args.limit is None else str(args.limit)
    print(f"Exported {len(written)} / requested {requested} -> {args.out.resolve()}")
    print("Wrote dataroom/MANIFEST.json (keep for attribution).")
    for path in written[:8]:
        print(f"  - {path.name}")
    if len(written) > 8:
        print(f"  ... and {len(written) - 8} more")


if __name__ == "__main__":
    main()
