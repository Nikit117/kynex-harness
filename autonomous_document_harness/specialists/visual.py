"""Visual specialist: bounded render/OCR/visual-check requests.

Honest about capability: without a vision model it reports what it *could*
do and returns the render artifacts it *did* produce.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..tools.computer import DocumentComputer


def inspect_visually(
    computer: DocumentComputer,
    doc_id: str,
    page: int,
    bbox: Optional[List[float]] = None,
    question: str = "",
) -> Dict[str, Any]:
    rendered = computer.registry.call("render_page", doc_id=doc_id, page=page)
    artifacts = [rendered.data] if rendered.status == "ok" else []
    if bbox:
        crop = computer.registry.call(
            "render_region", doc_id=doc_id, page=page, bbox=bbox
        )
        if crop.status == "ok":
            artifacts.append(crop.data)
    if bbox:
        visual = computer.registry.call(
            "inspect_region_visually",
            doc_id=doc_id,
            page=page,
            bbox=bbox,
            question=question,
        )
    else:
        visual = computer.registry.call(
            "inspect_page_visually", doc_id=doc_id, page=page, question=question
        )
    ocr = computer.registry.call("run_ocr", doc_id=doc_id, page=page)
    return {
        "doc_id": doc_id,
        "page": page,
        "question": question,
        "visual_status": visual.status,
        "visual_message": visual.message,
        "ocr_status": ocr.status,
        "ocr_text": ocr.data.get("text", "") if ocr.status == "ok" else "",
        "artifacts": artifacts,
    }
