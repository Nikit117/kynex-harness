"""Provider for the KYNEX (gpt-5.4) on ExtractBench.

The harness is the local TUI agent: it stages the document, lets the model call
tools (``read_page``, ``extract_table``, Document-Intelligence layout analysis,
vision) and returns schema-shaped JSON. This provider runs one harness
extraction per request and hands the extracted JSON to the official evaluator.

It deliberately calls the *same* helpers as the benchmark runner
(``autonomous_document_harness/scripts/run_tui_benchmark_v2.py``), so the
submitted system is the system that was measured: the non-record pruning rules,
the redaction-marker rule, the bulk enumeration of long record lists and the
staging-time OCR fallback for image-only pages all apply.

The harness returns no per-field citation boxes, so ``field_citations`` is empty
and the grounding metrics are N/A for this system -- the same situation as the
other value-only entries on the leaderboard.

Like the leaderboard's coding-agent pipelines (``claude_code_extract_*``,
``codex_code_extract_*``) this provider runs the extraction as a **local
subprocess** -- ``scripts/extract_one.py`` -- so the official environment needs
no extra Python packages. The interpreter is taken from
``KYNEX_HARNESS_PYTHON`` (default: the repository's own ``.venv``), and the
repository root from ``KYNEX_HARNESS_ROOT``.

To register it with the official CLI, add to
``extract_bench/inference/providers/extract/__init__.py``::

    from .nikit_harness import KynexExtractProvider  # noqa: F401

and to ``extract_bench/inference/pipelines/extract.py``::

    register_fn(
        _pipeline_spec(
            pipeline_name="nikit_harness_gpt_5_4",
            provider_name="nikit_harness_extract",
            config={"timeout": 3600},
            per_file_timeout=3600.0,
        )
    )
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from extract_bench.inference.providers.base import (
    Provider,
    ProviderPermanentError,
)
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

DEFAULT_REPO_ROOT = Path(os.getenv("KYNEX_HARNESS_ROOT") or r"C:\path\to\kynex-harness")

# The harness rules that were enabled for the measured 370-document benchmark
# run (official ExtractBench scorer).  They are passed to the subprocess
# explicitly so the submitted system is the system that was measured: most of
# the rules default to ON in the harness, but ``ADH_BULK_ARRAYS`` is opt-in and
# would otherwise be silently OFF in the official environment, which drops the
# bulk enumeration of long record lists.
HARNESS_ENV: dict[str, str] = {
    "ADH_BULK_ARRAYS": "1",
    "ADH_PRUNE_NON_RECORDS": "1",
    "ADH_REDACTION_MARKERS": "1",
    "ADH_OCR_FALLBACK": "1",
    "ADH_BULK_OCR": "1",
    "ADH_STRUCTURED_ARRAYS": "1",
}


def _harness_python() -> str:
    """The interpreter that has the harness' dependencies."""
    explicit = os.getenv("KYNEX_HARNESS_PYTHON", "").strip()
    if explicit:
        return explicit
    candidate = DEFAULT_REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if candidate.exists():
        return str(candidate)
    return sys.executable


async def _extract_async(source: Path, schema: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Run the harness extraction as a subprocess and return the JSON."""

    def _run() -> dict[str, Any]:
        repo = Path(os.getenv("KYNEX_HARNESS_ROOT") or DEFAULT_REPO_ROOT)
        script = repo / "autonomous_document_harness" / "scripts" / "extract_one.py"
        if not script.exists():
            raise ProviderPermanentError(
                f"harness entry point not found: {script} (set KYNEX_HARNESS_ROOT)"
            )
        with tempfile.TemporaryDirectory() as td:
            schema_file = Path(td) / "schema.json"
            schema_file.write_text(json.dumps(schema), encoding="utf-8")
            out_file = Path(td) / "extraction.json"
            cmd = [
                _harness_python(),
                "-I",
                str(script),
                "--source",
                str(source),
                "--schema",
                str(schema_file),
                "--out",
                str(out_file),
            ]
            env = os.environ.copy()
            env.update(HARNESS_ENV)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(repo),
                timeout=timeout,
                env=env,
            )
            if proc.returncode != 0:
                raise ProviderPermanentError(
                    f"harness extraction failed (exit {proc.returncode}): "
                    f"{(proc.stderr or proc.stdout)[-400:]}"
                )
            if not out_file.exists():
                raise ProviderPermanentError("harness produced no extraction file")
            extraction = json.loads(out_file.read_text(encoding="utf-8"))
        return {"extraction": extraction}

    return await asyncio.get_running_loop().run_in_executor(None, _run)


class KynexExtractProvider(Provider):
    """Provider for the KYNEX on ExtractBench.

    Configuration options:
        - ``timeout`` (int, default 3600): per-document timeout in seconds. A
          long document (192 pages) needs the bulk page-wise enumeration, which
          is the slowest path in the harness.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._timeout = int(self.base_config.get("timeout", 3600))

    def run_inference(self, pipeline: Any, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                "KynexExtractProvider only supports EXTRACT product type, got "
                f"{request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError("schema_override is required for EXTRACT product type.")
        source = Path(request.source_file_path)
        if not source.exists():
            raise ProviderPermanentError(f"File not found: {source}")

        started_at = datetime.now()
        try:
            raw_output = asyncio.run(
                asyncio.wait_for(
                    _extract_async(source, request.schema_override, self._timeout),
                    timeout=self._timeout,
                )
            )
        except asyncio.TimeoutError as exc:
            raise ProviderPermanentError(
                f"harness extraction timed out after {self._timeout}s"
            ) from exc
        except ProviderPermanentError:
            raise
        except Exception as exc:  # noqa: BLE001 -- report as a document failure
            raise ProviderPermanentError(f"harness extraction failed: {exc}") from exc

        completed_at = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        extraction = raw_result.raw_output.get("extraction")
        extracted_data = extraction if isinstance(extraction, (dict, list)) else {}
        output = ExtractOutput(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data,
            field_citations=[],  # the harness emits no per-field citation boxes
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
