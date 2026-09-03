"""Build a real rails config folder for each test, the way a user would.

The `.co` files are COPIED out of `flows_path()` rather than written here. That
makes every load test a test of the install step too: a wheel that ships no
flows, or a flow file with a Colang syntax error, fails these tests rather than
failing in somebody's config folder.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from jamjet_guardrails_nemo import flows_path

# NeMo's own convention, and deliberately not shipped inside the flow files: a
# config that already defines `bot refuse to respond` keeps its own wording.
REFUSAL_MESSAGE = "I am sorry, I cannot respond to that."

_MESSAGES_CO = f'define bot refuse to respond\n  "{REFUSAL_MESSAGE}"\n'


def build_rails_folder(root: Path, config_yaml: str) -> Path:
    """Write a config folder holding `config_yaml`, the shipped flows and a refusal."""
    folder = root / "rails"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yml").write_text(config_yaml, encoding="utf-8")
    (folder / "config.py").write_text(
        "from jamjet_guardrails_nemo import init  # noqa: F401\n", encoding="utf-8"
    )
    (folder / "messages.co").write_text(_MESSAGES_CO, encoding="utf-8")
    copied = 0
    for source in sorted(flows_path().glob("*.co")):
        shutil.copy(source, folder / source.name)
        copied += 1
    assert copied == 2, f"expected two shipped .co files, copied {copied} from {flows_path()}"
    return folder


@pytest.fixture
def rails_folder(tmp_path: Path) -> Callable[[str], Path]:
    """A factory bound to this test's tmp_path."""

    def make(config_yaml: str) -> Path:
        return build_rails_folder(tmp_path, config_yaml)

    return make
