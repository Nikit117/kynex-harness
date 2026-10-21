"""The submission path must be the measured path.

The official ExtractBench submission does not run the benchmark runner -- it
runs ``scripts/extract_one.py`` in a subprocess (see
``eval/extractbench/kynex_provider.py``).  If that entry point drifts from
the runner, the submitted system is NOT the system that was measured and the
leaderboard number does not describe the code in this repository.

Two things are pinned here:

  1. the post-processing chain of ``extract_one.py`` is the runner's chain, in
     the runner's order -- including the structured-table step (RC-12), which
     was missing and silently dropped every table-derived row from the
     submission;
  2. the provider passes the measured rule switches to the subprocess, so the
     opt-in rules (``ADH_BULK_ARRAYS``) are not silently off in the official
     environment.

The provider module itself cannot be imported in this venv (``extract_bench``
is only installed in ``.venv-extractbench``), so both files are inspected
syntactically with ``ast``.
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXTRACT_ONE = ROOT / "autonomous_document_harness/scripts/extract_one.py"
PROVIDER = ROOT / "autonomous_document_harness/eval/extractbench/kynex_provider.py"

# The benchmark runner's pipeline, in the order it applies it
# (run_tui_benchmark_v2.py: answer_to_json_v2 -> fill_massive_arrays ->
#  prune_non_record_items -> normalize_redaction_markers -> _apply_structured_arrays).
RUNNER_ORDER = [
    "answer_to_json_v2",
    "fill_massive_arrays",
    "prune_non_record_items",
    "normalize_redaction_markers",
    "_apply_structured_arrays",
]


def _called_names(path: pathlib.Path) -> list[str]:
    """Names of the functions called in the module's ``main``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.append(func.id)
        elif isinstance(func, ast.Attribute):
            names.append(func.attr)
    return names


class TestExtractOnePipeline:
    def test_chain_matches_the_runner(self):
        called = _called_names(EXTRACT_ONE)
        chain = [name for name in called if name in RUNNER_ORDER]
        # the last occurrence of each step defines the order
        chain = [name for name in RUNNER_ORDER if name in called]
        assert chain == RUNNER_ORDER, (
            "extract_one.py must apply the runner's post-processing chain in the "
            f"runner's order; got {chain}"
        )

    def test_structured_arrays_step_is_present(self):
        called = _called_names(EXTRACT_ONE)
        assert "_apply_structured_arrays" in called, (
            "the submission path must apply the structured-table rows (RC-12); "
            "without it the submitted system drops every table-derived row"
        )

    def test_structured_arrays_is_the_last_step(self):
        called = _called_names(EXTRACT_ONE)
        positions = {name: i for i, name in enumerate(called)}
        for step in RUNNER_ORDER[:-1]:
            assert positions[step] < positions["_apply_structured_arrays"], (
                f"{step} must run before the structured-table step"
            )


class TestProviderEnv:
    def _provider_tree(self) -> ast.Module:
        return ast.parse(PROVIDER.read_text(encoding="utf-8"))

    def test_harness_env_declares_the_measured_switches(self):
        tree = self._provider_tree()
        env: dict[str, str] = {}
        for node in tree.body:
            if (
                isinstance(node, ast.AnnAssign)
                and getattr(node.target, "id", "") == "HARNESS_ENV"
            ):
                for key, value in zip(node.value.keys, node.value.values):
                    env[key.value] = value.value
        assert env == {
            "ADH_BULK_ARRAYS": "1",
            "ADH_PRUNE_NON_RECORDS": "1",
            "ADH_REDACTION_MARKERS": "1",
            "ADH_OCR_FALLBACK": "1",
            "ADH_BULK_OCR": "1",
            "ADH_STRUCTURED_ARRAYS": "1",
        }, "the provider must enable exactly the rules the measured run used"

    def test_subprocess_receives_the_env(self):
        tree = self._provider_tree()
        run_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ]
        assert run_calls, "the provider must call subprocess.run"
        for call in run_calls:
            kwargs = {kw.arg for kw in call.keywords}
            assert "env" in kwargs, (
                "subprocess.run must receive env= so the measured rule switches "
                "reach the harness"
            )
