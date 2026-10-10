"""Run every pinned scanner over the hidden-Unicode corpus and record decisions.

One command reproduces the committed numbers:

    python benchmarks/unicode-scanners/run.py --tools-dir /tmp/unicode-scanners --install

Run it with the interpreter of an environment where this checkout is
installed (`pip install -e .`), from the repository root. `--install` needs the
network once: it builds `<tools-dir>/venv` from `tools/requirements.txt`
(hash-pinned), runs `npm ci` over `tools/package-lock.json`, and downloads
hidden-content-scan's script at its pinned commit, refusing it unless the
SHA-256 matches `tools.json`. Measuring needs no network, and on macOS every
scanner runs under `sandbox-exec` with network access denied; `network_denied`
in the output records whether that held.

Each case is written to its own directory as the file a repository would hold
(an `AGENTS.md` unless the case names another path), and each scanner is asked
about that one file with its default settings. One decision per file, read
from the exit code: 1 is a flag, 0 is a pass, anything else is an error and is
counted as a flag, because a CI step that errors fails the build.

Writes `results/measurements.json`. `render.py` turns it into `RESULTS.md`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CORPUS = HERE / "corpus.jsonl"
TOOLS = HERE / "tools.json"
OUT = HERE / "results" / "measurements.json"

_SANDBOX = "/usr/bin/sandbox-exec"
_NO_NETWORK = "(version 1)(allow default)(deny network*)"
_TIMEOUT = 60


def load_corpus() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line]


def load_tools() -> list[dict[str, Any]]:
    return list(json.loads(TOOLS.read_text(encoding="utf-8"))["tools"])


def naive_flags(text: str) -> bool:
    """Any Cf character, or any Cc character other than tab, LF and CR."""
    return any(
        unicodedata.category(c) == "Cf" or (unicodedata.category(c) == "Cc" and c not in "\t\n\r")
        for c in text
    )


def write_case(base: Path, case: dict[str, Any]) -> Path:
    """Write one case as bytes, UTF-8, exactly as its text says."""
    target = base / case["id"] / case["file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(case["text"].encode("utf-8"))
    return target


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(tools_dir: Path, tools: list[dict[str, Any]]) -> None:
    tools_dir.mkdir(parents=True, exist_ok=True)
    venv = tools_dir / "venv"
    if not venv.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    subprocess.run(
        [
            str(venv / "bin" / "pip"),
            "install",
            "--quiet",
            "--require-hashes",
            "--no-deps",
            "-r",
            str(HERE / "tools" / "requirements.txt"),
        ],
        check=True,
    )
    npm_dir = tools_dir / "npm"
    npm_dir.mkdir(exist_ok=True)
    for name in ("package.json", "package-lock.json"):
        shutil.copyfile(HERE / "tools" / name, npm_dir / name)
    subprocess.run(
        ["npm", "ci", "--no-audit", "--no-fund", "--ignore-scripts"], cwd=npm_dir, check=True
    )
    (hcs,) = [t for t in tools if t["id"] == "hidden-content-scan"]
    dest = tools_dir / "hidden-content-scan.py"
    with urllib.request.urlopen(hcs["url"]) as response:
        dest.write_bytes(response.read())
    if _sha256(dest) != hcs["sha256"]:
        dest.unlink()
        raise SystemExit(
            f"hidden-content-scan.py does not match the pinned SHA-256 {hcs['sha256']}"
        )


def _jamjet_script() -> Path:
    script = Path(sys.executable).with_name("jamjet-guardrails")
    if not script.exists():
        raise SystemExit(f"no jamjet-guardrails console script beside {sys.executable}")
    return script


def _git_commit() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head[:12] + ("+dirty" if dirty else "")


def _sandboxed(argv: list[str]) -> tuple[list[str], bool]:
    if sys.platform == "darwin" and Path(_SANDBOX).exists():
        return [_SANDBOX, "-p", _NO_NETWORK, *argv], True
    return argv, False


def _verify_installed(tools_dir: Path, tools: list[dict[str, Any]]) -> dict[str, str]:
    """Read each installed version back from the install, not from tools.json."""
    found: dict[str, str] = {}
    out = subprocess.run(
        [str(tools_dir / "venv" / "bin" / "unicode-smuggling-guard"), "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    found["unicode-smuggling-guard"] = out[-1]
    for tool_id, pkg in (
        ("ghostchars", "@antonsoloviev/ghostchars"),
        ("anti-trojan-source", "anti-trojan-source"),
    ):
        meta = json.loads(
            (tools_dir / "npm" / "node_modules" / pkg / "package.json").read_text(encoding="utf-8")
        )
        found[tool_id] = meta["version"]
    hcs = tools_dir / "hidden-content-scan.py"
    (pin,) = [t for t in tools if t["id"] == "hidden-content-scan"]
    if _sha256(hcs) != pin["sha256"]:
        raise SystemExit("hidden-content-scan.py does not match the pinned SHA-256")
    found["hidden-content-scan"] = pin["version"]
    for tool in tools:
        if tool["id"] in found and found[tool["id"]] != tool["version"]:
            raise SystemExit(
                f"{tool['id']} installed {found[tool['id']]}, tools.json pins {tool['version']}"
            )
    return found


def _fill(arg: str, subs: dict[str, str], file: str) -> str:
    for key, value in {**subs, "{file}": file}.items():
        arg = arg.replace(key, value)
    return arg


def _decide(code: int) -> str:
    return {0: "pass", 1: "flag"}.get(code, "error")


def measure(tools_dir: Path | None, only: set[str] | None) -> dict[str, Any]:
    corpus = load_corpus()
    tools = [t for t in load_tools() if only is None or t["id"] in only]
    external = {
        "unicode-smuggling-guard",
        "ghostchars",
        "hidden-content-scan",
        "anti-trojan-source",
    }
    installed: dict[str, str] = {}
    if any(t["id"] in external for t in tools):
        if tools_dir is None:
            raise SystemExit("--tools-dir is needed to run the external scanners")
        installed = _verify_installed(tools_dir, tools)
    from jamjet_guardrails import __version__

    commit = _git_commit()
    node = shutil.which("node")
    node_version = (
        subprocess.run(
            [node, "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if node
        else None
    )
    subs = {
        "{jamjet}": str(_jamjet_script()),
        "{tools}": str(tools_dir) if tools_dir else "",
        "{python}": sys.executable,
    }
    denied = False
    results: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for case in corpus:
            write_case(base, case)
        for tool in tools:
            decisions: dict[str, str] = {}
            for case in corpus:
                if tool["id"] == "naive-cf-cc":
                    decisions[case["id"]] = "flag" if naive_flags(case["text"]) else "pass"
                    continue
                argv = [_fill(a, subs, case["file"]) for a in tool["argv"]]
                argv, denied = _sandboxed(argv)
                try:
                    proc = subprocess.run(
                        argv,
                        cwd=base / case["id"],
                        capture_output=True,
                        timeout=_TIMEOUT,
                        check=False,
                    )
                    decisions[case["id"]] = _decide(proc.returncode)
                except subprocess.TimeoutExpired:
                    decisions[case["id"]] = "error"
            results[tool["id"]] = decisions

    ours = {"jamjet-guardrails": f"{__version__} at {commit}", "naive-cf-cc": f"run.py at {commit}"}
    versions = {t["id"]: ours.get(t["id"]) or installed[t["id"]] for t in tools}
    return {
        "measured": dt.date.today().isoformat(),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "unicode": unicodedata.unidata_version,
        "node": node_version,
        "network_denied": denied,
        "versions": versions,
        "decisions": results,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tools-dir", type=Path)
    parser.add_argument("--install", action="store_true", help="install the pinned tools first")
    parser.add_argument("--only", help="comma-separated tool ids to run")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    tools = load_tools()
    if args.install:
        if args.tools_dir is None:
            parser.error("--install needs --tools-dir")
        install(args.tools_dir.resolve(), tools)
    only = set(args.only.split(",")) if args.only else None
    data = measure(args.tools_dir.resolve() if args.tools_dir else None, only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
