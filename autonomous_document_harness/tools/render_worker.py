"""Isolated page renderer (subprocess worker).

Renders one PDF page to PNG via pypdfium2 in a SEPARATE process so a native
PDFium crash on pathological pages can never take down the investigation
run. Minimal imports (stdlib + pypdfium2 + pillow only) for fast startup.
No harness imports — this file must stay dependency-free.

Usage:
    python render_worker.py <pdf_path> <page_no> <dpi> <output_png>

Prints a JSON object to stdout on success; exits non-zero on any failure.
"""

import json
import sys


def main(argv):
    if len(argv) != 5:
        print(json.dumps({"error": "usage: pdf page dpi out_png"}))
        return 2
    _, pdf_path, page_no, dpi, output_path = argv
    try:
        import pypdfium2 as pdfium
    except ImportError:
        print(json.dumps({"error": "pypdfium2 is not installed"}))
        return 3
    from pathlib import Path

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            page_no = int(page_no)
            dpi = int(dpi)
            if not (1 <= page_no <= len(pdf)):
                print(json.dumps({"error": f"page {page_no} out of range"}))
                return 4
            bitmap = pdf[page_no - 1].render(scale=dpi / 72.0)
            pil = bitmap.to_pil()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            pil.save(output_path)
            print(
                json.dumps(
                    {
                        "image_path": output_path,
                        "width": pil.width,
                        "height": pil.height,
                        "dpi": dpi,
                    }
                )
            )
            return 0
        finally:
            pdf.close()
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
