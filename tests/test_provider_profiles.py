"""The provider profile is a convenience, so the measured default must not move.

The published result was measured on Azure with ``gpt-5.4``.  ``--provider``
exists so the same pipeline can be run on a machine without that resource; if it
ever changed the default, the reproduction would silently stop reproducing.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "autonomous_document_harness"
    / "scripts"
    / "verify_against_extractbench.py"
)


@pytest.fixture(scope="module")
def verify():
    spec = importlib.util.spec_from_file_location("verify_against_extractbench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_default_is_the_measured_configuration(verify):
    assert verify.build_parser().parse_args([]).provider == "azure"
    assert verify.provider_env("azure") == {}


def test_azure_declares_no_key_of_its_own(verify):
    assert verify.PROVIDERS["azure"]["key_env"] == ""
    assert verify.provider_key("azure", {"OPENAI_API_KEY": "x"}) == ""


def test_openai_switches_to_standard_chat_completions(verify):
    env = verify.provider_env("openai")
    assert env["ADH_MODEL_BACKEND"] == "openai"
    assert env["ADH_MODEL_BASE_URL"] == "https://api.openai.com/v1"
    assert env["ADH_MODEL_DEPLOYMENT"] == "gpt-5.4"


def test_the_overrides_win(verify):
    env = verify.provider_env("openai", "https://models.example/v1", "some-model")
    assert env["ADH_MODEL_BASE_URL"] == "https://models.example/v1"
    assert env["ADH_MODEL_DEPLOYMENT"] == "some-model"


def test_the_profile_reads_the_key_its_ecosystem_uses(verify):
    assert verify.provider_key("openai", {"OPENAI_API_KEY": " sk-1 "}) == "sk-1"
    assert verify.provider_key("openai", {}) == ""


def test_every_profile_is_selectable_from_the_command_line(verify):
    parser = verify.build_parser()
    for provider in verify.PROVIDERS:
        assert parser.parse_args(["--provider", provider]).provider == provider


def test_an_unknown_provider_is_rejected(verify):
    with pytest.raises(SystemExit):
        verify.build_parser().parse_args(["--provider", "nope"])


def test_no_profile_loads_a_key_from_the_command_line(verify):
    """A key typed as a flag lands in the shell history, so it is never accepted."""
    assert not hasattr(verify.build_parser().parse_args([]), "api_key")
