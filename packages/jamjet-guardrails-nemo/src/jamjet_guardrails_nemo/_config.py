"""Read the adapter's configuration out of a rails config, or refuse to load.

Everything here happens once, at rails load, and every failure is a refusal to
build rather than a message allowed through later. That is the core library's
own posture (`build_chain` refuses five shapes of "this configuration does not
mean what it looks like") carried into the host: a guardrails deployment whose
checks silently did not run is the failure this project exists to prevent, and
the cheapest place to catch it is before there is any content.

WHERE THE CONFIGURATION LIVES, and why it is not where you would first put it.

`RailsConfig` is a pydantic model with the default `extra` behaviour, which is
`ignore`. Measured on nemoguardrails 0.24.0: a top-level `jamjet_guardrails:`
key in `config.yml` is parsed, discarded, and never reaches the loaded config;
`RailsConfig.from_path(...)` succeeds, `model_extra` is `None`, and there is no
attribute to read it back from. A custom top-level key is therefore not a place
an adapter can store anything, and building on one would mean a typo'd or
misplaced block produced a silent no-op, which is the exact shape of failure
this module exists to refuse.

`custom_data` is NeMo's own declared field for this ("Any custom configuration
data"), it survives both `from_path` and `from_content`, and it is still a
custom key in the rails `config.yml`. So the configuration is:

    custom_data:
      jamjet_guardrails:
        input: [injection-structural]
        output: [pii, secrets]
        options:
          secrets: {on_match: deny}

A user who writes the block at the top level instead gets a refusal at load
naming this shape, because the key is absent from the only place it can be.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jamjet_guardrails import GuardrailChain, GuardrailUnavailableError, build, build_chain
from jamjet_guardrails.detectors import AVAILABLE

CONFIG_KEY = "jamjet_guardrails"


# Printed inside every refusal below, so a message that says the configuration
# is wrong also says what a right one looks like. The check names are read from
# the registry rather than written out: a list of names in a string goes stale
# the release a check is added, and an example naming a check the installed
# version cannot build is worse than no example.
def _example() -> str:
    installed = sorted(AVAILABLE)
    return (
        "custom_data:\n"
        f"  {CONFIG_KEY}:\n"
        f"    input: [{installed[0]}]\n"
        f"    output: [{installed[-1]}]\n"
        f"(installed checks: {installed})"
    )


@dataclass(frozen=True)
class RailConfiguration:
    """The checks each direction runs, and the options each check is built with."""

    input: tuple[str, ...]
    output: tuple[str, ...]
    options: Mapping[str, Mapping[str, Any]]


def read(custom_data: Mapping[str, Any]) -> RailConfiguration:
    """Pull the adapter's block out of `RailsConfig.custom_data`, or refuse.

    Every refusal is `GuardrailUnavailableError`, the error the core raises for
    the same class of mistake, so a host wrapping its rails load in one
    `except GuardrailUnavailableError` catches the configuration seam whole.
    """
    if not isinstance(custom_data, Mapping) or CONFIG_KEY not in custom_data:
        raise GuardrailUnavailableError(
            f"no {CONFIG_KEY!r} block in the rails config's custom_data, so this adapter "
            f"would register two actions that check nothing. Add:\n{_example()}"
        )
    block = custom_data[CONFIG_KEY]
    if not isinstance(block, Mapping):
        raise GuardrailUnavailableError(
            f"custom_data.{CONFIG_KEY} is a {type(block).__name__}, not a mapping. "
            f"Expected:\n{_example()}"
        )
    unknown = sorted(set(block) - {"input", "output", "options"})
    if unknown:
        # A misspelled direction is the mistake this catches, and it is the
        # expensive one: `inputs: [pii]` leaves `input` empty, so the input rail
        # checks nothing while the config reads as though it does. Refusing the
        # key is the only way that is ever noticed.
        raise GuardrailUnavailableError(
            f"custom_data.{CONFIG_KEY} has keys this adapter does not know: {unknown}. "
            "Expected 'input', 'output' and 'options'. A misspelled direction leaves that "
            "rail configured with nothing and silently checking nothing"
        )
    options = _options(block.get("options", {}))
    configured = RailConfiguration(
        input=_names(block.get("input", ()), "input"),
        output=_names(block.get("output", ()), "output"),
        options=options,
    )
    if not configured.input and not configured.output:
        raise GuardrailUnavailableError(
            f"custom_data.{CONFIG_KEY} names no checks in either direction. A rails config "
            f"that installs this adapter and lists nothing is a config that checks "
            f"nothing.\n{_example()}"
        )
    # An option for a check that no direction runs configures nothing. It is
    # the same mistake one level up: the config reads as though `secrets` were
    # tuned, and `secrets` is not in the chain at all.
    orphaned = sorted(set(options) - set(configured.input) - set(configured.output))
    if orphaned:
        raise GuardrailUnavailableError(
            f"custom_data.{CONFIG_KEY}.options configures checks that neither direction "
            f"runs: {orphaned}. Add them to 'input' or 'output', or remove the options"
        )
    return configured


def _names(value: object, where: str) -> tuple[str, ...]:
    """The names one direction lists, refusing every shape that is not a list of them.

    `build_chain` refuses the same shapes and this is not that check moved: this
    one runs over a YAML value before either chain is built, so it can name the
    direction the mistake is in, and it runs on the path that carries per-check
    options, where `build_chain` cannot be the gate (see `chain_for`).
    `tests/test_config.py` asserts both paths refuse the same set.
    """
    if isinstance(value, (str, bytes, bytearray)):
        # `input: pii` rather than `input: [pii]`. A string is an iterable of
        # its own characters, so unguarded this asks the registry for a check
        # called "p".
        raise GuardrailUnavailableError(
            f"custom_data.{CONFIG_KEY}.{where} is the scalar {value!r}; it takes a list, "
            f"such as [{value!r}]"
        )
    if not isinstance(value, Iterable):
        # `input:` with nothing after it parses as None.
        cause = " A key present with no value parses as null." if value is None else ""
        raise GuardrailUnavailableError(
            f"custom_data.{CONFIG_KEY}.{where} is a {type(value).__name__}, which is not a "
            f"list of check names.{cause}"
        )
    listed = list(value)
    not_strings = [entry for entry in listed if not isinstance(entry, str)]
    if not_strings:
        raise GuardrailUnavailableError(
            f"custom_data.{CONFIG_KEY}.{where} holds entries that are not check names: "
            f"{not_strings!r}"
        )
    return tuple(listed)


def _options(value: object) -> Mapping[str, Mapping[str, Any]]:
    """The per-check options table, refusing anything that is not one."""
    if not isinstance(value, Mapping):
        raise GuardrailUnavailableError(
            f"custom_data.{CONFIG_KEY}.options is a {type(value).__name__}, not a mapping of "
            "check name to that check's options"
        )
    for name, options in value.items():
        if not isinstance(name, str):
            raise GuardrailUnavailableError(
                f"custom_data.{CONFIG_KEY}.options is keyed by check name; {name!r} is not one"
            )
        if not isinstance(options, Mapping) or any(not isinstance(k, str) for k in options):
            raise GuardrailUnavailableError(
                f"custom_data.{CONFIG_KEY}.options[{name!r}] is not a mapping of option name "
                "to value"
            )
    return {str(name): dict(options) for name, options in value.items()}


def chain_for(names: Sequence[str], options: Mapping[str, Mapping[str, Any]]) -> GuardrailChain:
    """Build one direction's chain, refusing at load rather than at first message.

    `build_chain` is the mechanism whenever no listed check carries options: it
    is the core's own front door and it refuses all five shapes of a
    configuration that would check less than it names.

    It cannot be the mechanism when a check DOES carry options, and the reason
    is not stylistic. `build_chain` constructs every guardrail with no options
    at all, and some registered checks refuse to exist without them: `rules`
    with no patterns, no banned substrings and no limits checks nothing, so the
    core refuses to build it. Routing an option-carrying configuration through
    `build_chain` would therefore reject a configuration that is valid. The
    option-carrying path uses `build(name, **options)` per check, which is the
    same registry front door with the same five refusals per name, and
    `_names` above has already refused every container shape `build_chain`
    would have.
    """
    listed = list(names)
    if any(name in options for name in listed):
        return GuardrailChain([build(name, **options.get(name, {})) for name in listed])
    return build_chain(listed)
