"""DocumentComputer: the agent's hands. Every primitive from the mission spec.

Tools return structured Observations with provenance. Anything requiring a
missing backend (OCR engine, vision model) returns status="unavailable" with
a reason — never an exception, never silence.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.schemas import BudgetUsage, ExtractionMethod, SourceType, validate_bbox
from ..verification.entailment import entails
from ..workspace.world_model import WorldModel
from . import azure_di as di_mod
from . import compute as compute_mod
from . import ocr as ocr_mod
from . import search as search_mod
from . import spreadsheets as sheet_mod
from . import tables as tables_mod
from . import text as text_mod
from . import vision as vision_mod
from . import vision_llm as vision_llm_mod
from .registry import (
    ToolError,
    ToolRegistry,
    ToolSpec,
    ToolUnavailable,
    provenance,
)

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers(text: str) -> List[float]:
    out = []
    for raw in _NUMBER_RE.findall(text or ""):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


class DocumentComputer:
    def __init__(
        self,
        world: WorldModel,
        budgets: BudgetUsage,
        render_dir: Optional[str] = None,
        config=None,
        gateway=None,
    ) -> None:
        self.world = world
        self.budgets = budgets
        self.config = config
        self.render_dir = Path(render_dir or tempfile.mkdtemp(prefix="adh_render_"))
        self.render_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ToolRegistry(budgets)
        self._di_cache: Dict[str, Any] = {}
        if gateway is None and config is not None:
            from ..models.gateway import ModelGateway

            gateway = ModelGateway(config)
        self.gateway = gateway
        self._register_all()

    # -- registration ----------------------------------------------------
    def _tool(
        self,
        name: str,
        description: str,
        func,
        cost: float = 1.0,
        needs_visual: bool = False,
        needs_model: bool = False,
        **params: str,
    ) -> None:
        self.registry.register(
            ToolSpec(
                name=name,
                description=description,
                params=dict(params),
                cost=cost,
                needs_visual=needs_visual,
                needs_model=needs_model,
            ),
            func,
        )

    def _register_all(self) -> None:
        self._tool(
            "list_documents",
            "List every document in the workspace corpus.",
            lambda: self._list_documents(),
        )
        self._tool(
            "inspect_document",
            "Profile, headings, sheets and errors of one document.",
            lambda doc_id: self._inspect_document(doc_id),
            doc_id="stable document ID",
        )
        self._tool(
            "get_document_metadata",
            "Filename, fingerprint, size, classification.",
            lambda doc_id: self._metadata(doc_id),
            doc_id="stable document ID",
        )
        self._tool(
            "search_text",
            "Lexical/semantic/hybrid search with provenance hits.",
            lambda query, top_k=10, mode="hybrid": search_mod.text_search(
                self.world, query, mode, top_k
            ),
            query="search text",
            top_k="max hits",
            mode="lexical|semantic|hybrid",
        )
        self._tool(
            "semantic_search",
            "Conceptual search (local TF-IDF vectors).",
            lambda query, top_k=10: search_mod.text_search(
                self.world, query, "semantic", top_k
            ),
            query="search text",
            top_k="max hits",
        )
        self._tool(
            "hybrid_search",
            "RRF fusion of lexical + semantic rankings.",
            lambda query, top_k=10: search_mod.text_search(
                self.world, query, "hybrid", top_k
            ),
            query="search text",
            top_k="max hits",
        )
        self._tool(
            "search_headings",
            "Search document headings/structure.",
            lambda query, top_k=10: search_mod.heading_search(self.world, query, top_k),
            query="heading text",
            top_k="max hits",
        )
        self._tool(
            "read_text",
            "Full (truncated) native text of a document.",
            lambda doc_id, max_chars=4000: text_mod.read_text(
                self.world, doc_id, max_chars
            ),
            doc_id="document ID",
            max_chars="truncate at N chars",
        )
        self._tool(
            "read_page",
            "Native text + structure of one page. Embedded images are auto-OCR'd.",
            lambda doc_id, page: self._read_page_with_image_ocr(doc_id, page),
            doc_id="document ID",
            page="1-based page number",
        )
        self._tool(
            "read_section",
            "Text under a heading, across page breaks.",
            lambda doc_id, heading: text_mod.read_section(self.world, doc_id, heading),
            doc_id="document ID",
            heading="heading text",
        )
        self._tool(
            "read_surrounding_context",
            "Pages around a target page.",
            lambda doc_id, page, window=1: text_mod.read_surrounding_context(
                self.world, doc_id, page, window
            ),
            doc_id="document ID",
            page="center page",
            window="pages each side",
        )
        self._tool(
            "render_page",
            "Render a PDF page to PNG (escalation ladder step).",
            lambda doc_id, page, dpi=110, output_dir="": self._render_page(
                doc_id, page, dpi, output_dir or None
            ),
            doc_id="document ID",
            page="1-based page",
            dpi="resolution",
        )
        self._tool(
            "render_region",
            "Render + crop a bbox region (pixels of rendered page).",
            lambda doc_id, page, bbox, dpi=150, output_dir="": self._render_region(
                doc_id, page, tuple(bbox), dpi, output_dir or None
            ),
            doc_id="document ID",
            page="page",
            bbox="[x0,y0,x1,y1] px",
            dpi="resolution",
        )
        self._tool(
            "inspect_page_visually",
            "GPT-5.4 Vision reading of a rendered page (unavailable if unconfigured).",
            lambda doc_id, page, question: self._vision_inspect(
                doc_id, page, None, question
            ),
            doc_id="document ID",
            page="page",
            question="question",
            cost=3.0,
            needs_visual=True,
            needs_model=True,
        )
        self._tool(
            "inspect_region_visually",
            "GPT-5.4 Vision reading of a targeted crop (unavailable if unconfigured).",
            lambda doc_id, page, bbox, question: self._vision_inspect(
                doc_id, page, tuple(bbox), question
            ),
            doc_id="document ID",
            page="page",
            bbox="bbox",
            question="question",
            cost=3.0,
            needs_visual=True,
            needs_model=True,
        )
        self._tool(
            "detect_layout",
            "Blocks/headings/tables/image-coverage per page.",
            lambda doc_id, page=None: tables_mod.detect_layout(
                self.world, doc_id, page
            ),
            doc_id="document ID",
            page="optional page",
        )
        self._tool(
            "detect_tables",
            "Candidate tables with dimensions + headers.",
            lambda doc_id, page=None: tables_mod.detect_tables(
                self.world, doc_id, page
            ),
            doc_id="document ID",
            page="optional page",
        )
        self._tool(
            "extract_table",
            "Full rows of one table (or a whole sheet).",
            lambda doc_id, table_index=0, page=None, sheet=None: (
                tables_mod.extract_table(self.world, doc_id, table_index, page, sheet)
            ),
            doc_id="document ID",
            table_index="0-based",
            page="optional",
            sheet="optional sheet name",
        )
        self._tool(
            "reconstruct_table",
            "Logical tables: header hierarchies, banner groups, continuations.",
            lambda doc_id, page=None, sheet=None: tables_mod.reconstruct_table(
                self.world, doc_id, page, sheet
            ),
            doc_id="document ID",
            page="optional page",
            sheet="optional sheet name",
        )
        self._tool(
            "inspect_table_visually",
            "GPT-5.4 Vision check of a table region.",
            lambda doc_id, page, table_index=0: self._vision_inspect_table(
                doc_id, page, table_index
            ),
            doc_id="document ID",
            page="page",
            table_index="0-based",
            cost=3.0,
            needs_visual=True,
            needs_model=True,
        )
        self._tool(
            "di_analyze_document",
            "Azure Document Intelligence layout/OCR/tables for target pages.",
            lambda doc_id, pages=None: self._di_analyze(doc_id, pages),
            doc_id="document ID",
            pages="explicit page list, e.g. [1, 2]",
            cost=2.0,
            needs_model=True,
        )
        self._tool(
            "vision_verify_claim",
            "GPT-5.4 visual check of one claim against a render/crop.",
            lambda doc_id, page, subject, attribute, value, bbox=None, claim_id="": (
                self._vision_verify(
                    doc_id,
                    page,
                    subject,
                    attribute,
                    value,
                    tuple(bbox) if bbox else None,
                    claim_id,
                )
            ),
            doc_id="document ID",
            page="page",
            subject="claim subject",
            attribute="claim attribute",
            value="claimed value",
            bbox="optional [x0,y0,x1,y1] px crop",
            claim_id="optional claim ID for evidence linking",
            cost=3.0,
            needs_visual=True,
            needs_model=True,
        )
        self._tool(
            "run_ocr",
            "OCR a page (unavailable when no engine installed).",
            lambda doc_id, page=None: self._ocr(doc_id, page, None),
            doc_id="document ID",
            page="optional page",
            cost=2.0,
        )
        self._tool(
            "ocr_region",
            "OCR a bbox region of a rendered page.",
            lambda doc_id, page, bbox: self._ocr(doc_id, page, tuple(bbox)),
            doc_id="document ID",
            page="page",
            bbox="[x0,y0,x1,y1] px",
            cost=2.0,
        )
        self._tool(
            "read_spreadsheet",
            "Workbook overview: sheets, dims, merged, formulas.",
            lambda doc_id: sheet_mod.read_spreadsheet(self.world, doc_id),
            doc_id="document ID",
        )
        self._tool(
            "read_sheet_range",
            "Addressed cell range from a sheet.",
            lambda doc_id, sheet, cell_range="A1:Z50": sheet_mod.read_sheet_range(
                self.world, doc_id, sheet, cell_range
            ),
            doc_id="document ID",
            sheet="sheet name",
            cell_range="e.g. A1:E10",
        )
        self._tool(
            "find_similar_sections",
            "Pages/sections most similar to a given page.",
            lambda doc_id, page, top_k=5: self._similar(doc_id, page, top_k),
            doc_id="document ID",
            page="reference page",
            top_k="max hits",
        )
        self._tool(
            "find_supporting_evidence",
            "Hybrid hits supporting a question.",
            lambda query, top_k=5: self._supporting(query, top_k),
            query="question",
            top_k="max hits",
        )
        self._tool(
            "find_contradicting_evidence",
            "Cross-document numeric disagreements.",
            lambda query, top_k=10: self._contradictions(query, top_k),
            query="question",
            top_k="max candidates",
        )
        self._tool(
            "compare_sources",
            "Side-by-side view of two evidence snippets.",
            lambda a, b: self._compare(a, b),
            a="hit dict",
            b="hit dict",
        )
        self._tool(
            "verify_claim",
            "Heuristic entailment: does evidence state the value?",
            lambda value, snippets, min_overlap=0.34: self._verify_claim(
                value, snippets, min_overlap
            ),
            value="claimed value",
            snippets="evidence strings",
            min_overlap="token overlap threshold",
            cost=2.0,
            needs_model=True,
        )
        self._tool(
            "calculate",
            "Sandboxed arithmetic (preferred over guessing).",
            lambda expression: (
                {"expression": expression, "result": compute_mod.safe_eval(expression)},
                [],
            ),
            expression="e.g. '1248 * 2 + 100'",
        )
        self._tool(
            "run_python_analysis",
            "Restricted analysis script over `data`; sets `result`.",
            lambda code, data: (compute_mod.run_analysis(code, data), []),
            code="script source",
            data="JSON input bound as `data`",
        )

    # -- read_page with embedded-image auto-OCR -----------------------------

    def _read_page_with_image_ocr(
        self, doc_id: str, page: int
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """read_page + automatic Vision OCR of significant embedded images."""
        data, prov = text_mod.read_page(self.world, doc_id, page)

        parsed = self.world.parsed.get(doc_id)
        if parsed is None:
            return data, prov
        page_obj = next((p for p in parsed.pages if p.page_no == page), None)
        if page_obj is None or not page_obj.image_bboxes:
            return data, prov

        if data.get("char_count", 0) < 30:
            return data, prov

        ref = self.world.refs.get(doc_id)
        if ref is None or ref.file_type != ".pdf":
            return data, prov

        try:
            self._vision_ready()
        except Exception:
            return data, prov

        dpi = 150
        scale = dpi / 72.0
        image_texts: list[str] = []
        for bbox_pt in page_obj.image_bboxes[:5]:
            if "visual" in self.budgets.exhausted():
                break
            try:
                page_path = self.render_dir / f"{doc_id}_p{page}_{dpi}dpi.png"
                if not page_path.exists():
                    vision_mod.render_page_to_png(
                        ref.path, page, dpi, str(page_path)
                    )
                bbox_px = (
                    bbox_pt[0] * scale,
                    bbox_pt[1] * scale,
                    bbox_pt[2] * scale,
                    bbox_pt[3] * scale,
                )
                crop_path = (
                    self.render_dir
                    / f"{doc_id}_p{page}_img_{int(bbox_pt[0])}_{int(bbox_pt[1])}.png"
                )
                vision_mod.crop_region(
                    str(page_path), bbox_px, str(crop_path)
                )
                self.budgets.consume("visual")
                self.budgets.consume("model")
                result = vision_llm_mod.inspect_images(
                    self.gateway,
                    [str(crop_path)],
                    "Read ALL text visible in this image region. "
                    "Include labels, numbers, headings, and any data.",
                    max_side=self.config.vision_image_max_side
                    if self.config
                    else 1568,
                )
                answer = (result.get("answer") or "").strip()
                if answer:
                    image_texts.append(answer)
            except Exception:
                continue

        if image_texts:
            data["embedded_image_texts"] = image_texts
            combined = "\n".join(image_texts)
            data["text"] = (
                data.get("text", "")
                + "\n\n[EMBEDDED IMAGE CONTENT]\n"
                + combined
            )

        return data, prov

    # -- corpus tools ------------------------------------------------------
    def _list_documents(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        docs = [
            {
                "doc_id": p.doc_id,
                "filename": p.filename,
                "file_type": p.file_type,
                "classification": p.classification,
                "pages": p.page_count,
                "tables": p.table_candidates,
                "sheets": [s["name"] for s in p.sheets],
                "needs_ocr": p.needs_ocr,
                "errors": p.errors,
            }
            for p in self.world.profiles.values()
        ]
        return {"count": len(docs), "documents": docs}, []

    def _inspect_document(
        self, doc_id: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        profile = self.world.profiles.get(doc_id)
        if profile is None:
            raise ToolError(f"unknown document '{doc_id}'")
        parsed = self.world.parsed.get(doc_id)
        data: Dict[str, Any] = {
            "doc_id": doc_id,
            "filename": profile.filename,
            "classification": profile.classification,
            "pages": profile.page_count,
            "text_chars": profile.text_chars,
            "tables": profile.table_candidates,
            "headings": list(profile.headings),
            "sheets": list(profile.sheets),
            "needs_ocr": profile.needs_ocr,
            "errors": list(profile.errors),
            "profile": {
                "file_type": profile.file_type,
                "classification": profile.classification,
                "avg_text_density": profile.avg_text_density,
                "image_coverage_avg": profile.image_coverage_avg,
            },
        }
        if parsed is not None:
            data["warnings"] = list(parsed.warnings)
        return data, [
            provenance(
                doc_id, source_type=SourceType.PAGE, method=ExtractionMethod.NATIVE_TEXT
            )
        ]

    def _metadata(self, doc_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        ref = self.world.refs.get(doc_id)
        profile = self.world.profiles.get(doc_id)
        if ref is None or profile is None:
            raise ToolError(f"unknown document '{doc_id}'")
        return (
            {
                "doc_id": doc_id,
                "filename": ref.filename,
                "fingerprint": ref.fingerprint,
                "size_bytes": ref.size_bytes,
                "file_type": ref.file_type,
                "classification": profile.classification,
            },
            [
                provenance(
                    doc_id,
                    source_type=SourceType.PAGE,
                    method=ExtractionMethod.NATIVE_TEXT,
                )
            ],
        )

    # -- escalation tools ----------------------------------------------------
    def _render_page(
        self, doc_id: str, page: int, dpi: int = 110, output_dir: Optional[str] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        ref = self.world.refs.get(doc_id)
        if ref is None:
            raise ToolError(f"unknown document '{doc_id}'")
        if ref.file_type != ".pdf":
            raise ToolError(f"render_page supports PDF, not '{ref.file_type}'")
        out = (
            Path(output_dir or str(self.render_dir)) / f"{doc_id}_p{page}_{dpi}dpi.png"
        )
        try:
            info = vision_mod.render_page_to_png(ref.path, page, dpi, str(out))
        except ImportError:
            raise ToolError("pypdfium2 is not installed; rendering unavailable")
        except vision_mod.RenderCrashed as exc:
            # Native renderer died on this page: mark unavailable (not fatal)
            # so the agent tries other pages/regions instead.
            raise ToolUnavailable(f"render crashed on page {page}: {exc}")
        except TimeoutError as exc:
            raise ToolError(str(exc))
        except Exception as exc:
            raise ToolError(f"render failed: {exc}")
        prov = [
            provenance(
                doc_id,
                page=page,
                source_type=SourceType.PAGE,
                method=ExtractionMethod.VISION,
                snippet=str(out),
            )
        ]
        return {"doc_id": doc_id, **info}, prov

    def _render_region(
        self,
        doc_id: str,
        page: int,
        bbox: tuple,
        dpi: int = 150,
        output_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        ref = self.world.refs.get(doc_id)
        if ref is None:
            raise ToolError(f"unknown document '{doc_id}'")
        if ref.file_type != ".pdf":
            raise ToolError(f"render_region supports PDF, not '{ref.file_type}'")
        try:
            _, _, _, _ = validate_bbox(tuple(float(v) for v in bbox))
        except ValueError as exc:
            raise ToolError(f"invalid bbox: {exc}")
        out_dir = Path(output_dir or str(self.render_dir))
        page_path = out_dir / f"{doc_id}_p{page}_{dpi}dpi.png"
        try:
            vision_mod.render_page_to_png(ref.path, page, dpi, str(page_path))
            info = vision_mod.crop_region(
                str(page_path),
                tuple(float(v) for v in bbox),
                str(out_dir / f"{doc_id}_p{page}_crop.png"),
            )
        except ImportError:
            raise ToolError("rendering backends (pypdfium2/pillow) missing")
        except vision_mod.RenderCrashed as exc:
            raise ToolUnavailable(f"render crashed on page {page}: {exc}")
        except TimeoutError as exc:
            raise ToolError(str(exc))
        except (ToolError, ValueError):
            raise
        except Exception as exc:
            raise ToolError(f"region render failed: {exc}")
        prov = [
            provenance(
                doc_id,
                page=page,
                bbox=tuple(float(v) for v in bbox),
                source_type=SourceType.REGION,
                method=ExtractionMethod.VISION,
            )
        ]
        return {"doc_id": doc_id, **info}, prov

    def _vision_ready(self) -> None:
        """Raise ToolUnavailable unless GPT-5.4 Vision is actually callable."""
        if self.config is None or self.gateway is None:
            raise ToolUnavailable(
                "vision inspection unavailable: no model configured "
                "(set ADH_MODEL_BACKEND=openai-compatible with credentials)"
            )
        if not vision_llm_mod.vision_configured(self.config):
            raise ToolUnavailable(
                "vision inspection unavailable: openai-compatible backend needs "
                "ADH_MODEL_API_KEY/ADH_MODEL_BASE_URL/ADH_MODEL_DEPLOYMENT "
                "(or AZURE_OPENAI_* equivalents)"
            )

    _IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"})

    def _render_artifact(
        self, doc_id: str, page: int, bbox: Optional[tuple], dpi: int
    ) -> Tuple[str, Dict[str, Any]]:
        """Render a page (or crop) and return (image_path, render_info)."""
        ref = self.world.refs.get(doc_id)
        if ref is None:
            raise ToolError(f"unknown document '{doc_id}'")
        if ref.file_type in self._IMAGE_EXTS:
            target = ref.path
            info = {"source": "raw_image", "file_type": ref.file_type}
        elif ref.file_type == ".pdf":
            page_path = self.render_dir / f"{doc_id}_p{page}_{dpi}dpi.png"
            try:
                info = vision_mod.render_page_to_png(
                    ref.path, page, dpi, str(page_path)
                )
            except ImportError:
                raise ToolError("pypdfium2/pillow missing; rendering unavailable")
            except Exception as exc:
                raise ToolError(f"render failed: {exc}")
            target = str(page_path)
        else:
            raise ToolError(
                f"vision inspection supports PDF and images, not '{ref.file_type}'"
            )
        extra: Dict[str, Any] = {}
        if bbox is not None:
            validate_bbox(tuple(float(v) for v in bbox))
            crop = self.render_dir / f"{doc_id}_p{page}_vision_crop.png"
            try:
                extra = vision_mod.crop_region(
                    target, tuple(float(v) for v in bbox), str(crop)
                )
            except Exception as exc:
                raise ToolError(f"crop failed: {exc}")
            target = str(crop)
        return target, {"render": info, "crop": extra}

    def _vision_inspect(
        self, doc_id: str, page: int, bbox: Optional[tuple], question: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        self._vision_ready()
        if not (question or "").strip():
            raise ToolError("vision inspection needs a question")
        image, render_info = self._render_artifact(doc_id, page, bbox, 150)
        try:
            result = vision_llm_mod.inspect_images(
                self.gateway,
                [image],
                question,
                max_side=self.config.vision_image_max_side,
            )
        except Exception as exc:
            raise ToolError(f"vision call failed: {type(exc).__name__}: {exc}")
        image_hash = vision_mod.image_fingerprint(image)
        prov = [
            provenance(
                doc_id,
                page=page,
                bbox=tuple(float(v) for v in bbox) if bbox else None,
                source_type=SourceType.REGION if bbox else SourceType.PAGE,
                method=ExtractionMethod.GPT54_VISION,
                snippet=result["answer"][:300],
                image_hash=image_hash,
                dpi=150,
            )
        ]
        return (
            {
                "doc_id": doc_id,
                "page": page,
                "question": question,
                "answer": result["answer"],
                "backend": result["backend"],
                "image_path": image,
                "image_hash": image_hash,
                "dpi": 150,
                "render": render_info,
                "extraction_method": ExtractionMethod.GPT54_VISION,
            },
            prov,
        )

    def _vision_inspect_table(
        self, doc_id: str, page: int, table_index: int
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        data, prov = tables_mod.extract_table(self.world, doc_id, table_index, page)
        question = (
            "Read this table precisely. Preserve hierarchical/merged headers: "
            "state each column's full header path and every row's values. "
            f"Native extraction saw {data.get('n_rows')} rows x "
            f"{data.get('n_cols')} cols."
        )
        return self._vision_inspect(doc_id, page, None, question)

    def _vision_verify(
        self,
        doc_id: str,
        page: int,
        subject: str,
        attribute: str,
        value: Any,
        bbox: Optional[tuple],
        claim_id: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        self._vision_ready()
        image, render_info = self._render_artifact(doc_id, page, bbox, 200)
        try:
            verdict = vision_llm_mod.verify_claim_image(
                self.gateway,
                image,
                subject,
                attribute,
                value,
                max_side=self.config.vision_image_max_side,
            )
        except Exception as exc:
            raise ToolError(f"vision verification failed: {type(exc).__name__}: {exc}")
        retried_hires = False
        if verdict.get("supported") is None:
            # Ambiguous verdict -> single bounded high-resolution retry.
            hires, render_info = self._render_artifact(doc_id, page, bbox, 300)
            try:
                verdict = vision_llm_mod.verify_claim_image(
                    self.gateway,
                    hires,
                    subject,
                    attribute,
                    value,
                    max_side=self.config.vision_image_max_side,
                )
                image, retried_hires = hires, True
            except Exception as exc:
                raise ToolError(f"vision retry failed: {type(exc).__name__}: {exc}")
        image_hash = vision_mod.image_fingerprint(image)
        dpi = 300 if retried_hires else 200
        prov = [
            provenance(
                doc_id,
                page=page,
                bbox=tuple(float(v) for v in bbox) if bbox else None,
                source_type=SourceType.REGION if bbox else SourceType.PAGE,
                method=ExtractionMethod.GPT54_VISION,
                snippet=verdict["rationale"],
                image_hash=image_hash,
                dpi=dpi,
            )
        ]
        return (
            {
                "doc_id": doc_id,
                "page": page,
                "subject": subject,
                "attribute": attribute,
                "value": value,
                "claim_id": claim_id,
                "supported": verdict["supported"],
                "rationale": verdict["rationale"],
                "confidence": verdict["confidence"],
                "backend": verdict["backend"],
                "image_hash": image_hash,
                "dpi": dpi,
                "retried_hires": retried_hires,
                "extraction_method": ExtractionMethod.GPT54_VISION,
            },
            prov,
        )

    def _di_analyze(
        self, doc_id: str, pages: Optional[list]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if self.config is None or not di_mod.di_configured(self.config):
            raise ToolUnavailable(
                "Azure Document Intelligence unavailable: needs ADH_DI_ENDPOINT + "
                "ADH_DI_API_KEY and the azure-ai-documentintelligence package"
            )
        ref = self.world.refs.get(doc_id)
        if ref is None:
            raise ToolError(f"unknown document '{doc_id}'")
        _DI_SUPPORTED = frozenset({".pdf"}) | self._IMAGE_EXTS
        if ref.file_type not in _DI_SUPPORTED:
            raise ToolError(
                f"di_analyze_document supports PDF and images, not '{ref.file_type}'"
            )
        targets = sorted({int(p) for p in (pages or [1, 2, 3]) if int(p) >= 1})
        if not targets:
            raise ToolError("di_analyze_document needs explicit target pages")
        client = di_mod.AzureDIClient(
            self.config.model.di_endpoint,
            self.config.model.di_key,
            cache=self._di_cache,
        )
        try:
            mapped = client.analyze_layout(ref.path, targets, cache_key=ref.fingerprint)
        except Exception as exc:
            raise ToolError(
                f"Document Intelligence failed: {type(exc).__name__}: {exc}"
            )
        floor = self.config.di_confidence_floor
        low = [
            i
            for i, t in enumerate(mapped.tables)
            if t.avg_confidence() is not None and t.avg_confidence() < floor
        ]
        data = mapped.to_data()
        data.update(
            {
                "doc_id": doc_id,
                "pages_analyzed": targets,
                "defaulted_pages": pages is None,
                "low_confidence_tables": low,
                "extraction_method": ExtractionMethod.AZURE_DI,
                "di_tables": [
                    {
                        "page": t.page,
                        "header": t.header(),
                        "rows": t.grid(),
                        "bbox": list(
                            di_mod.polygon_bbox(
                                [v for c in t.cells if c.polygon for v in c.polygon]
                            )
                        )
                        if any(c.polygon for c in t.cells)
                        else None,
                    }
                    for t in mapped.tables
                ],
                "di_pages": [
                    {"page": p.page_no, "text": p.text[:4000]}
                    for p in mapped.pages
                    if p.text.strip()
                ],
            }
        )
        prov: List[Dict[str, Any]] = []
        for table in mapped.tables:
            polys = [c.polygon for c in table.cells if c.polygon]
            bbox = (
                di_mod.polygon_bbox([v for poly in polys for v in poly])
                if polys
                else None
            )
            prov.append(
                provenance(
                    doc_id,
                    page=table.page,
                    bbox=tuple(bbox) if bbox else None,
                    source_type=SourceType.TABLE,
                    method=ExtractionMethod.AZURE_DI,
                )
            )
        for page in mapped.pages:
            prov.append(
                provenance(
                    doc_id,
                    page=page.page_no,
                    source_type=SourceType.PAGE,
                    method=ExtractionMethod.AZURE_DI,
                    snippet=page.text[:300],
                )
            )
        return data, prov

    def _ocr(
        self, doc_id: str, page: Optional[int], bbox: Optional[tuple]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        availability = ocr_mod.probe()
        if not availability.available:
            # Availability signal (not an error): the agent adapts its plan.
            raise ToolUnavailable(f"OCR unavailable: {availability.reason}")
        ref = self.world.refs.get(doc_id)
        if ref is None:
            raise ToolError(f"unknown document '{doc_id}'")
        if ref.file_type == ".pdf":
            if page is None:
                raise ToolError("run_ocr on PDF requires a page number")
            page_path = self.render_dir / f"{doc_id}_p{page}_ocr.png"
            vision_mod.render_page_to_png(ref.path, page, 200, str(page_path))
            target = str(page_path)
            if bbox is not None:
                validate_bbox(tuple(float(v) for v in bbox))
                crop = self.render_dir / f"{doc_id}_p{page}_ocr_crop.png"
                vision_mod.crop_region(target, tuple(float(v) for v in bbox), str(crop))
                target = str(crop)
        else:
            target = ref.path
        text = ocr_mod.run_on_image(target)
        prov = [
            provenance(
                doc_id,
                page=page,
                bbox=tuple(float(v) for v in bbox) if bbox else None,
                source_type=SourceType.REGION if bbox else SourceType.PAGE,
                method=ExtractionMethod.OCR,
                snippet=text[:300],
            )
        ]
        return (
            {
                "doc_id": doc_id,
                "engine": ocr_mod.probe().engine,
                "text": text,
                "available": True,
            },
            prov,
        )

    # -- reasoning helpers -----------------------------------------------------
    def _similar(
        self, doc_id: str, page: int, top_k: int
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        _, parsed = text_mod._require_doc(self.world, doc_id)
        ref_page = next((p for p in parsed.pages if p.page_no == page), None)
        if ref_page is None or not ref_page.text.strip():
            raise ToolError(f"page {page} of '{doc_id}' has no text to compare")
        hits = [
            h
            for h in self.world.tfidf.search(ref_page.text[:2000], top_k=top_k + 3)
            if not (h.provenance_doc == doc_id and h.page == page)
        ][:top_k]
        dicts = [
            {
                "provenance": provenance(
                    h.provenance_doc,
                    page=h.page,
                    cell=h.cell,
                    source_type=SourceType.TEXT,
                    method="semantic",
                    snippet=h.snippet,
                ),
                "snippet": h.snippet,
                "score": h.score,
            }
            for h in hits
        ]
        return (
            {"doc_id": doc_id, "page": page, "similar": dicts},
            [d["provenance"] for d in dicts],
        )

    def _supporting(
        self, query: str, top_k: int
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        data, prov = search_mod.text_search(self.world, query, "hybrid", top_k)
        return ({"query": query, "hits": data["hits"]}, prov)

    def _contradictions(
        self, query: str, top_k: int
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        data, _ = search_mod.text_search(self.world, query, "hybrid", max(top_k, 10))
        observations = []
        for h in data["hits"]:
            for number in _numbers(h["snippet"]):
                observations.append((h, number))
        candidates = []
        prov = []
        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                (ha, na), (hb, nb) = observations[i], observations[j]
                if (
                    na != nb
                    and ha["provenance"]["document_id"]
                    != hb["provenance"]["document_id"]
                ):
                    candidates.append(
                        {
                            "a": {
                                "snippet": ha["snippet"],
                                "provenance": ha["provenance"],
                                "number": na,
                            },
                            "b": {
                                "snippet": hb["snippet"],
                                "provenance": hb["provenance"],
                                "number": nb,
                            },
                        }
                    )
                    prov.extend([ha["provenance"], hb["provenance"]])
                    if len(candidates) >= top_k:
                        break
            if len(candidates) >= top_k:
                break
        return ({"query": query, "candidates": candidates}, prov)

    @staticmethod
    def _compare(
        a: Dict[str, Any], b: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise ToolError("compare_sources needs two hit dicts {snippet, provenance}")
        pa, pb = a.get("provenance", {}), b.get("provenance", {})
        return (
            {
                "a_snippet": str(a.get("snippet", ""))[:1000],
                "b_snippet": str(b.get("snippet", ""))[:1000],
                "same_document": pa.get("document_id") == pb.get("document_id"),
                "a_len": len(str(a.get("snippet", ""))),
                "b_len": len(str(b.get("snippet", ""))),
            },
            [pa, pb],
        )

    @staticmethod
    def _verify_claim(
        value: Any, snippets: List[str], min_overlap: float
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if not snippets:
            raise ToolError("verify_claim needs at least one evidence snippet")
        # Single source of truth with the entailment gate (verification/entailment.py).
        supported, score, matched_idx = entails(value, snippets, float(min_overlap))
        return (
            {
                "value": value,
                "supported": supported,
                "score": round(score, 4),
                "matched_snippet": matched_idx,
                "method": "heuristic-entailment",
            },
            [],
        )
