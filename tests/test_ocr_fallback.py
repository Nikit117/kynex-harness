"""RC-9: the staging-time OCR fallback for image-only pages.

The degraded re-captures (`*_corrupted`) have no native text at all, so the
agent never sees their content. Document Intelligence reads them; this checks
that the fallback actually fills the pages and that the text it produces
contains the ground-truth values.
"""

from __future__ import annotations

import pathlib

import pytest

from autonomous_document_harness.parsers.pdf import PdfParser

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _ref(doc_id: str):
    from autonomous_document_harness.workspace.corpus import DocumentRef

    pdf = next(ROOT.glob(f"data/**/{doc_id}.pdf"), None)
    if pdf is None:
        pytest.skip(f"{doc_id}.pdf not in the corpus")
    return DocumentRef(
        doc_id=doc_id,
        path=str(pdf),
        filename=pdf.name,
        fingerprint=doc_id,
        file_type=".pdf",
    )


class TestOcrFallback:
    def test_image_only_page_gets_text(self, monkeypatch):
        """The real degraded document: 0 native characters, DI supplies them."""
        monkeypatch.setenv("ADH_OCR_FALLBACK", "1")
        monkeypatch.setenv("ADH_DI_ENDPOINT", "https://example.invalid")
        monkeypatch.setenv("ADH_DI_API_KEY", "not-a-real-key")

        calls: list = []

        class FakeDiPage:
            def __init__(self, page_no, text):
                self.page_no = page_no
                self.text = text
                self.line_count = len(text.splitlines())

        class FakeTable:
            def __init__(self, page):
                self.page = page

            def grid(self):
                return [["PROVIDER NUMBER", "0123456789"], ["WARRANT NO", "39248026"]]

        class FakeLayout:
            pages = [FakeDiPage(1, "CA MEDI-CAL\nPROVIDER NUMBER\n0123456789")]
            tables = [FakeTable(1)]

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            def analyze_layout(self, path, pages=None, cache_key=""):
                calls.append(list(pages or []))
                return FakeLayout()

        import autonomous_document_harness.tools.azure_di as di_mod

        monkeypatch.setattr(di_mod, "AzureDIClient", FakeClient)

        parser = PdfParser()
        doc = parser.parse(_ref("ca_rad_crossover_outpatient_p5_corrupted"))

        assert calls, "the DI client was never called"
        assert doc.pages[0].char_count > 0, "the empty page was not filled"
        assert "0123456789" in doc.pages[0].text
        assert doc.pages[0].tables, "the DI table was not attached"

    def test_fallback_is_disabled_by_the_env_switch(self, monkeypatch):
        monkeypatch.setenv("ADH_OCR_FALLBACK", "0")
        monkeypatch.setenv("ADH_DI_ENDPOINT", "https://example.invalid")
        monkeypatch.setenv("ADH_DI_API_KEY", "not-a-real-key")

        class Boom:
            def __init__(self, *a, **kw):
                raise AssertionError("the DI client must not be constructed")

        import autonomous_document_harness.tools.azure_di as di_mod

        monkeypatch.setattr(di_mod, "AzureDIClient", Boom)

        parser = PdfParser()
        doc = parser.parse(_ref("ca_rad_crossover_outpatient_p5_corrupted"))
        assert doc.pages[0].char_count == 0, "the page must be left untouched"

    def test_a_failure_never_breaks_the_parse(self, monkeypatch):
        monkeypatch.setenv("ADH_OCR_FALLBACK", "1")
        monkeypatch.setenv("ADH_DI_ENDPOINT", "https://example.invalid")
        monkeypatch.setenv("ADH_DI_API_KEY", "not-a-real-key")

        class Boom:
            def __init__(self, *a, **kw):
                pass

            def analyze_layout(self, *a, **kw):
                raise RuntimeError("service unavailable")

        import autonomous_document_harness.tools.azure_di as di_mod

        monkeypatch.setattr(di_mod, "AzureDIClient", Boom)

        parser = PdfParser()
        doc = parser.parse(_ref("ca_rad_crossover_outpatient_p5_corrupted"))
        assert doc is not None
        assert doc.pages[0].char_count == 0
        assert any("no selectable text" in w for w in doc.warnings)

    def test_native_text_documents_are_not_sent_to_di(self, monkeypatch):
        """A document with a text layer must not pay for an OCR call."""
        monkeypatch.setenv("ADH_OCR_FALLBACK", "1")
        monkeypatch.setenv("ADH_DI_ENDPOINT", "https://example.invalid")
        monkeypatch.setenv("ADH_DI_API_KEY", "not-a-real-key")

        calls: list = []

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            def analyze_layout(self, path, pages=None, cache_key=""):
                calls.append(list(pages or []))
                raise RuntimeError("must not be called")

        import autonomous_document_harness.tools.azure_di as di_mod

        monkeypatch.setattr(di_mod, "AzureDIClient", FakeClient)

        parser = PdfParser()
        parser.parse(_ref("real_enotes_deviations"))
        assert not calls, f"a text document was sent to DI: {calls}"


class TestCorpusScan:
    def test_image_only_documents_are_detected(self):
        if not list((ROOT / "data").glob("**/*.pdf")):
            pytest.skip("no corpus in this checkout")
        """How many documents in the corpus have no native text at all?

        The degraded re-captures must all be in this set -- they are the
        documents the OCR fallback exists for.
        """
        import fitz

        empty_docs = []
        for pdf in ROOT.glob("data/**/*.pdf"):
            try:
                doc = fitz.open(pdf)
            except Exception:
                continue
            chars = sum(len(p.get_text().strip()) for p in doc)
            doc.close()
            if chars == 0:
                empty_docs.append(pdf.stem)
        assert empty_docs, "expected image-only documents in the corpus"
        degraded = [d for d in empty_docs if d.endswith("_corrupted")]
        assert len(degraded) >= 30, f"only {len(degraded)} degraded documents found"
        for name in (
            "ca_rad_crossover_outpatient_p5_corrupted",
            "real_enotes_deviations_corrupted",
            "brand_price_list_lifescience_2024_corrupted",
        ):
            assert name in empty_docs, name
