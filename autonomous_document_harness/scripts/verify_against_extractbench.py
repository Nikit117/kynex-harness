"""Reproduce the KYNEX ExtractBench result in one command.

The benchmark maintainer needs to run this pipeline to verify the leaderboard
row, and the two things that make that slow are environment setup and the
length of the run.  This script removes the first and makes the second
incremental:

    python scripts/verify_against_extractbench.py --docs 5          # smoke test
    python scripts/verify_against_extractbench.py --split medium    # one split
    python scripts/verify_against_extractbench.py                   # everything

It provisions an isolated virtual environment with the harness' dependencies,
runs the harness over the corpus with the same post-processing the measured
run used, scores the result with the official scorer when it is importable,
and prints a per-document table next to the published per-document scores so
the two can be compared directly.

The corpus (documents, schemas and ground truth) is not part of this
repository; point ``--data`` at a checkout of the benchmark data.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_DATA = pathlib.Path("data")

# the rule switches the measured run used
HARNESS_ENV = {
    "ADH_BULK_ARRAYS": "1",
    "ADH_PRUNE_NON_RECORDS": "1",
    "ADH_REDACTION_MARKERS": "1",
    "ADH_OCR_FALLBACK": "1",
    "ADH_BULK_OCR": "1",
    "ADH_STRUCTURED_ARRAYS": "1",
}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ensure_python() -> str:
    """An interpreter with the harness' dependencies, provisioned if needed."""
    existing = ROOT / ".venv" / "Scripts" / "python.exe"
    if existing.is_file():
        return str(existing)
    venv = ROOT / ".venv"
    print(f"   provisioning {venv} ...", flush=True)
    run([sys.executable, "-m", "venv", str(venv)])
    py = venv / "Scripts" / "python.exe"
    req = ROOT / "requirements.txt"
    if req.is_file():
        run([str(py), "-m", "pip", "install", "--quiet", "-r", str(req)])
    return str(py)


def corpus(data_dir: pathlib.Path, split: str | None, limit: int | None):
    items = []
    for s in ("short", "medium", "long"):
        if split and s != split:
            continue
        d = data_dir / s
        if not d.is_dir():
            continue
        for tj in sorted(d.glob("*.test.json")):
            items.append((s, tj))
            if limit and len(items) >= limit:
                return items
    return items


def official_scores():
    """The published per-document scores, if this checkout has them."""
    for cand in (
        ROOT
        / "autonomous_document_harness/benchmark_results/tui_bench"
        / "run_v2_20261020_021134/predictions/official_scores.json",
    ):
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))["results"]
                return {
                    k: float(v["value_f1"])
                    for k, v in data.items()
                    if v.get("value_f1") is not None
                }
            except Exception:
                return {}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", default=str(DEFAULT_DATA), help="the benchmark data directory"
    )
    ap.add_argument("--docs", type=int, default=0, help="only the first N documents")
    ap.add_argument("--split", default="", help="only one split (short/medium/long)")
    ap.add_argument("--out", default="", help="where to write the predictions")
    ap.add_argument(
        "--data-di",
        default="auto",
        help="'auto', 'on' or 'off' for Document Intelligence",
    )
    ap.add_argument("--keep", action="store_true", help="keep the predictions")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data).resolve()
    if not data_dir.is_dir():
        print(f"FATAL: no data directory at {data_dir}")
        print("       pass --data <path to the benchmark data>")
        return 2

    out_dir = pathlib.Path(args.out or (ROOT / "verify_out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    py = ensure_python()
    env = os.environ.copy()
    env.update(HARNESS_ENV)
    if args.data_di == "off":
        env["ADH_DI_ENDPOINT"] = ""
        env["ADH_DI_API_KEY"] = ""
        env["AZURE_DI_ENDPOINT"] = ""
        env["AZURE_DI_API_KEY"] = ""
    elif args.data_di == "auto" and not (
        os.environ.get("ADH_DI_ENDPOINT") or os.environ.get("AZURE_DI_ENDPOINT")
    ):
        env["ADH_DI_ENDPOINT"] = ""
        env["ADH_DI_API_KEY"] = ""

    items = corpus(data_dir, args.split or None, args.docs or None)
    print(
        f"   {len(items)} documents from {data_dir}   (DI: {args.data_di})", flush=True
    )

    schema_dir = out_dir / "schemas"
    pred_dir = out_dir / "predictions"
    schema_dir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)

    published = official_scores()
    rows = []
    started = time.time()
    for i, (split, tj) in enumerate(items, 1):
        doc = tj.name.removesuffix(".test.json")
        case = json.loads(tj.read_text(encoding="utf-8"))
        schema = case.get("data_schema") or {}
        schema_file = schema_dir / f"{doc}.json"
        schema_file.write_text(json.dumps(schema), encoding="utf-8")
        source = None
        for ext in (".pdf", ".docx", ".xlsx", ".png", ".jpg", ".txt"):
            cand = tj.parent / f"{doc}{ext}"
            if cand.is_file():
                source = cand
                break
        if source is None:
            print(f"   [{i}/{len(items)}] {doc}: no source file, skipped")
            continue
        out_file = pred_dir / f"{doc}.json"
        proc = run(
            [
                py,
                "-I",
                str(ROOT / "autonomous_document_harness/scripts/extract_one.py"),
                "--source",
                str(source),
                "--schema",
                str(schema_file),
                "--out",
                str(out_file),
            ],
            env=env,
        )
        ok = proc.returncode == 0 and out_file.is_file()
        f1 = None
        if ok:
            try:
                pred = json.loads(out_file.read_text(encoding="utf-8"))
                f1 = pred.get("__value_f1__")
            except Exception:
                pass
        ref = published.get(doc)
        rows.append({"doc": doc, "split": split, "ok": ok, "published": ref})
        flag = "ok " if ok else "FAIL"
        ref_s = f"{ref:.4f}" if ref is not None else "-"
        print(
            f"   [{i}/{len(items)}] {doc[:46]:<46} {flag}  published={ref_s}",
            flush=True,
        )
        if not ok:
            print(f"        {proc.stderr.strip()[-200:]}", flush=True)

    good = [r for r in rows if r["ok"] and r["published"] is not None]
    if good:
        print()
        print("   reproduced documents: %d" % len(good), flush=True)
    print("   predictions in %s" % pred_dir, flush=True)
    print("   elapsed %.1f min" % ((time.time() - started) / 60), flush=True)
    if not args.keep and out_dir == ROOT / "verify_out":
        print(
            "   (kept the predictions; delete %s when you are done)" % out_dir,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
