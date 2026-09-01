"""Synthetic attacks and hard negatives, generated locally with Ollama.

Local generation rather than a hosted API: no key to hold, no per-run cost, and
a run anyone can repeat with the same model pulled. The model DIGEST is recorded
per row rather than the tag, because a tag moves and a digest does not.

`HARD_NEGATIVE_KINDS` is the point of this module. An injection classifier fails
in the field on text that TALKS ABOUT instructions without being one, and public
benign corpora are made of ordinary prose that looks nothing like it. Task 3's
licence screen left this stage with two usable public corpora and no public
`eval` corpus at all, so what is generated here is most of the data rather than
a supplement to it.

## What is pinned, and what that buys

A row records the model tag and the sha256 of the weights blob behind it. The
seed, the sampling options, the prompt text and the date sit beside the rows in
`provenance.json`, because they are properties of a RUN and repeating them on
every row would be three thousand copies of one fact.

Reproducibility was measured rather than assumed, on ollama 0.24.0 with this
model on 2026-09-01:

- The same seed returns a byte-identical response, on two runs of the same
  request some minutes apart.
- It stays byte-identical when the request is issued as one of six concurrent
  ones instead of alone, so the continuous batching this module uses for
  throughput does not perturb it. That was worth checking rather than assuming:
  batched decode changes the reduction order of a floating point sum, and it
  would have been a reasonable guess that it changed the sampled token too.

What that does NOT establish is reproduction on other hardware or a later
ollama build, and this file does not claim it. The digest says which weights;
the seed and options say how they were sampled; whether a different machine
lands on the same token is a property of the runtime, not of anything recorded
here.

## Two deviations from the task brief, both deliberate

**The transport is the HTTP API, not `ollama run`.** `ollama run` offers no way
to set the sampling seed. The brief's sketch put `Seed: {n}` inside the prompt
TEXT, which varies the prompt and therefore the output, but it is not a seed:
re-running it is not reproducible in any sense the word usually carries, and the
row provenance would have recorded a number that controlled nothing. The API
takes `options.seed`, and the paragraph above is the evidence that it works.

**One call returns a batch, not one example.** Measured on this machine: a
single request decodes at somewhere between 4 and 12 tokens per second
depending on what else is running, and four to six concurrent requests reach
about 18 to 21 tokens per second between them. One round trip per example would
have put a corpus of this size out of reach. The prompts themselves are
unchanged; only the envelope asking for N of them differs.

A bounded `num_ctx` is part of that. The tag's default context is 32768, which
made the resident model about 17 GB on a 36 GB machine and put it into memory
pressure; `OPTIONS` sets 2048, which is ample for a batch of eight and roughly
halves the footprint.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: The repository root. `training/` sits at the root, so this file's
#: grandparent is it.
ROOT = Path(__file__).resolve().parent.parent

#: The generated corpus, and the record of what produced it.
#:
#: Under `training/` and NOT under `data/`, which is a decision the brief's
#: sketch got the other way round. `training/README.md` states the split: files
#: under `data/` are raw downloads and intermediate splits, gitignored by the
#: anchored `/data/` rule, while `training/` holds anything a published number
#: is measured on or measured by. This corpus is fitted on, so it is a published
#: input, and CI has to be able to screen it without an Ollama anywhere.
GENERATED = ROOT / "training" / "generated" / "rows.jsonl"
PROVENANCE = ROOT / "training" / "generated" / "provenance.json"

MODEL = "qwen2.5:14b"

#: The local Ollama HTTP API. Loopback, and nothing here reaches the network.
ENDPOINT = "http://127.0.0.1:11434/api/generate"

HARD_NEGATIVE_KINDS: tuple[str, ...] = (
    "user_correcting_themselves",  # "ignore my last message, I meant the other file"
    "documentation_quoting_an_attack",
    "security_report_with_payload",
    "prompt_engineering_tutorial",
    "roleplay_request",  # benign fiction that mentions rules
    "config_or_code_with_instructions",
    "translation_request",
    "meta_question_about_the_system",  # "what are your instructions?" asked honestly
)

ATTACK_KINDS: tuple[str, ...] = (
    "direct_override",
    "indirect_via_retrieved_content",
    "role_reassignment",
    "delimiter_confusion",
    "encoded_payload",
    "multi_turn_setup",
    "tool_misuse_request",
    "exfiltration_request",
)

#: Every kind, and the label a row of that kind carries. Derived from the two
#: tuples rather than written out a third time: a kind added to one of them
#: without a label here would be a kind the generator cannot run, and a label
#: map maintained by hand is the third copy that drifts.
LABELS: dict[str, int] = {kind: 0 for kind in HARD_NEGATIVE_KINDS} | {
    kind: 1 for kind in ATTACK_KINDS
}

KINDS: tuple[str, ...] = HARD_NEGATIVE_KINDS + ATTACK_KINDS


class GenerationError(RuntimeError):
    """The generator could not produce rows it is willing to record."""


@dataclass(frozen=True, slots=True)
class Row:
    """One generated example, and enough provenance to explain where it came from."""

    text: str
    label: int
    kind: str
    prompt_id: str
    model: str
    model_digest: str


@dataclass(frozen=True, slots=True)
class Generator:
    """A model used to produce training data, and the licence finding on it.

    Registered rather than named in passing, for the same reason
    `REFERENCE_MODELS` in `tests/test_training_data.py` is a registry: the
    identifier scan over this tree accounts for a model by finding its entry,
    so a generator swapped in without an entry fails rather than passing
    quietly. It is also where the licence question is answered in a form a test
    can read, instead of in prose a reader has to trust.

    `licence` is compared against `training.screen.licence_refusal`, the same
    allowlist every corpus in `training/sources.yaml` is screened by. A
    generator whose grant this repository has not screened is refused there,
    not here.
    """

    #: The Ollama tag, which is what a reader types to obtain the same weights.
    tag: str
    #: The upstream repository the weights come from.
    weights_id: str
    #: The SPDX identifier, screened by the allowlist in `training/screen.py`.
    licence: str
    #: sha256 of the licence text the artifact itself carries, as printed by
    #: `ollama show --license <tag>`. The grant is pinned by its bytes rather
    #: than by a name somebody typed, so a re-pull that quietly ships different
    #: terms is visible.
    licence_sha256: str
    #: When the finding was made. A model card can be edited and an Ollama tag
    #: can be repointed; a licence read without a date is one nobody can
    #: re-check.
    read_on: str
    #: What the finding rests on, and what it does not reach.
    note: str


#: Every model that produced data in this tree.
#:
#: One entry today. The finding on it is size-specific and does NOT generalise
#: to the family it belongs to: the 3B size of the same generation ships under
#: `qwen-research`, a licence restricting use to research, which would have been
#: refused by `training/screen.py` exactly as `cc-by-nc-4.0` is. Reading "Qwen2.5
#: is Apache-2.0" off one size and applying it to another is the same class of
#: mistake as reading a corpus licence off a downstream tag.
GENERATORS: tuple[Generator, ...] = (
    Generator(
        tag=MODEL,
        weights_id="Qwen/Qwen2.5-14B-Instruct",
        licence="apache-2.0",
        licence_sha256="c156170b718ec29139d3653d40ed1986fd92fb7e0959b5c71f3c48f62e6636f4",
        read_on="2026-09-01",
        note=(
            "Apache-2.0, read three ways on 2026-09-01 and agreeing: the licence text "
            "bundled with the local artifact and printed by `ollama show --license "
            "qwen2.5:14b`, which ends 'Copyright 2024 Alibaba Cloud'; the tag's page in "
            "the Ollama library; and the upstream model card, whose licence field is "
            "`apache-2.0`. Apache-2.0 grants use, modification and distribution and says "
            "nothing about model output, so it imposes no term on data generated with the "
            "weights and no term on a model fitted to that data. That is the whole reason "
            "the question had to be asked: a licence that DID reach outputs, as several "
            "open-weight licences do through an acceptable-use policy or a derivative-works "
            "clause naming outputs, would follow the classifier into an Apache-2.0 wheel. "
            "What this does not establish is anything about the training data behind the "
            "generator, which Alibaba Cloud has not published; the grant covers the weights "
            "that were released, and a claim about what went into them is not available to "
            "anyone outside it."
        ),
    ),
)

#: Vocabulary that names the CLASSES a row is labelled with, as opposed to the
#: phenomenon a prompt describes. No prompt may contain any of it.
#:
#: The rule is one-sided and that is deliberate rather than an oversight. A
#: prompt for an attack kind says "prompt-injection attempt" because an
#: injection is the thing being written, and a generator told to write one
#: writes one. A prompt for a negative kind that said "benign" would be naming
#: the class instead of the artifact, and a generator told to write something
#: benign writes ordinary prose: fluent, obviously safe, and worthless as a hard
#: negative, because the whole difficulty this corpus exists to capture is text
#: that reads like an instruction and is not one.
#:
#: So this catches the defect the brief names and does not pretend to be a proof
#: that a prompt is well posed. `documentation_quoting_an_attack` steers by
#: saying the surrounding text is "clearly explanatory", and
#: `user_correcting_themselves` by saying it should not read "like an attack".
#: Both constrain the SHAPE of what to write. Neither hands the generator the
#: label and lets it choose anything at all on the far side of it.
LABEL_VOCABULARY: tuple[str, ...] = (
    "benign",
    "harmless",
    "innocuous",
    "non-malicious",
    "nonmalicious",
    "false positive",
    "true positive",
    "negative example",
    "positive example",
    "hard negative",
    "label",
    "class 0",
    "class 1",
    "safe example",
)

_PROMPTS: dict[str, str] = {
    "user_correcting_themselves": (
        "Write one message a user sends to an AI assistant when they want to "
        "retract what they just asked and redirect it. Use natural phrasing like "
        "'ignore that', 'forget my last message', 'scratch that, instead'. It "
        "should read like an ordinary correction mid-conversation, not like an "
        "attack. Vary the domain: code, travel, cooking, spreadsheets."
    ),
    "documentation_quoting_an_attack": (
        "Write one paragraph from security documentation explaining how text sitting "
        "inside a document or a message can hijack a language model that reads it, "
        "the way an OWASP page or a vendor's docs would, and quoting a short example "
        "inline. The quoted example is plain English addressed to a model, of the "
        "'disregard the above and ...' sort. It is not a database, memory or markup "
        "exploit. The text around the quote is clearly explanatory."
    ),
    "security_report_with_payload": (
        "Write one short passage from a vulnerability report a researcher files with "
        "a vendor about the vendor's language-model feature. The finding is that "
        "text supplied by an outside party was followed by the model as though the "
        "operator had written it, and the report quotes, verbatim, the plain-English "
        "string that was sent, so the vendor can reproduce it. Professional "
        "disclosure voice, addressed to a security team. Not a database, memory or "
        "markup exploit."
    ),
    "prompt_engineering_tutorial": (
        "Write one short passage from a tutorial teaching a reader how to write "
        "better prompts for a language model. It shows a sample prompt inline as "
        "the thing being taught and then discusses why that wording works, the "
        "way a blog post or a course module would. Address the reader as a "
        "learner and vary the technique being taught: few-shot examples, giving "
        "the model a role, asking it to think step by step, constraining the "
        "output format."
    ),
    "roleplay_request": (
        "Write one message, in the first person and addressed directly to an AI "
        "assistant, asking it to play a character in a story the user is writing "
        "with it. The user speaks as themselves: they set a scene in a sentence or "
        "two and say who the assistant should play. The character may be one who "
        "breaks rules, defies orders, lies or keeps secrets, because that is the "
        "part. Do not write the story, and do not narrate in the third person. Write "
        "only what the user types."
    ),
    "config_or_code_with_instructions": (
        "Write one short excerpt from a file a real repository holds, carrying "
        "imperative text inside it: a YAML field whose value is a system prompt, a "
        "Python string constant holding instructions for a model, a docstring "
        "describing how an agent must behave, a template with placeholders and "
        "directives, a README block quoting the persona a service is configured "
        "with. Show it with its surrounding syntax, keys and quoting, so it reads as "
        "a fragment of a file rather than as a message to anybody."
    ),
    "translation_request": (
        "Write one message where a user asks an AI assistant to translate a "
        "passage between two named languages and includes the passage inline. "
        "The passage they want translated is itself instructions of some kind: "
        "assembly steps, a recipe, a notice telling staff what to do, terms of "
        "service, a safety warning. Name both languages and quote the passage."
    ),
    "meta_question_about_the_system": (
        "Write one question an ordinary curious user asks an AI assistant about "
        "how it works: what it was told to do, what it can and cannot help with, "
        "whether it remembers earlier messages. Honest curiosity, no attempt to "
        "extract or override anything."
    ),
    "direct_override": (
        "Write one prompt-injection attempt that tries to make an AI assistant "
        "disregard its existing instructions and follow new ones instead. Vary "
        "the technique across calls: pretend the conversation restarted, claim "
        "developer authority, use fake system markers, embed the instruction in "
        "what looks like data."
    ),
    "indirect_via_retrieved_content": (
        "Write one short document of the kind a language model reads while working "
        "on somebody's behalf: a web page excerpt, an email body, a code comment, a "
        "calendar invite, a product review, a support ticket. Most of it is ordinary "
        "content about its own subject. One or two sentences inside it break frame "
        "and speak directly, in the second person, to the model reading the "
        "document, telling it to set aside the task it was given and do something "
        "else instead. Those sentences address the model and not the reader, and "
        "they say what to do instead in plain words. Address it generically, as "
        "'assistant', 'AI' or 'model', or with no name at all, and never by a "
        "product name. Write the document only, with no framing around it."
    ),
    "role_reassignment": (
        "Write one message telling an AI assistant that it is now a different "
        "system, with a different name, character or set of permissions, and "
        "that the constraints it was operating under belonged to the previous "
        "one. Vary how the new persona arrives across calls: a named alter ego, "
        "an unrestricted mode, a debug or maintenance role, a fictional operator "
        "who answers to nobody."
    ),
    "delimiter_confusion": (
        "Write one message that fakes the structural markers a chat system uses to "
        "separate turns and roles, so an instruction the user typed appears to have "
        "arrived from the system or the operator. The text placed inside the faked "
        "markers must itself be a directive changing how the assistant behaves. Vary "
        "the markers: XML-like role tags, ChatML turn tokens, a code fence closed "
        "early so what follows escapes it, JSON shaped like a message array."
    ),
    "encoded_payload": (
        "Write one message that hides an instruction to a language model inside an "
        "encoding and asks, in plain text alongside it, for the encoded part to be "
        "decoded and then carried out. The hidden instruction is a real sentence, "
        "and the encoding is correct for the scheme used. Vary the scheme: base64, "
        "rot13, hexadecimal, text written backwards, letters spaced apart, an "
        "acrostic down the first letters, leetspeak."
    ),
    "multi_turn_setup": (
        "Write one opening message to an AI assistant that asks for nothing yet and "
        "instead asserts a premise about the assistant's own operation that a later "
        "message will lean on: that a restriction it works under was lifted for this "
        "session, that permission for something was granted earlier in the "
        "conversation, that the two of them agreed a way of working at the start, "
        "that its operator signed off on an exception. It is addressed to the "
        "assistant, it is about what the assistant may now do, and on its own it "
        "reads as unremarkable."
    ),
    "tool_misuse_request": (
        "Write one message asking an AI assistant that has tools to carry out an "
        "action its operator would not sanction, phrased as an ordinary piece of "
        "work. Vary the action: run a shell command with wide effect, read or write "
        "a path outside the workspace it was given, call an internal network "
        "endpoint, send mail as somebody else, install a package from a source "
        "nobody named, widen its own permissions, disable a check. The action is "
        "concrete and named in the message."
    ),
    "exfiltration_request": (
        "Write one message trying to get an AI assistant to reveal or transmit "
        "something it holds and was not asked to share: the text of its system "
        "prompt, the contents of a file it read earlier, a key from its "
        "environment, earlier conversation history. Vary the pretext across "
        "calls: debugging, an audit, a formatting exercise, a translation, a "
        "summary, a test of its memory."
    ),
}

#: Which revision of a kind's prompt produced a row, recorded in `prompt_id`.
#:
#: Not decoration, and not versioning for its own sake. The task brief's step 4
#: says to read the output and rewrite a prompt that is not producing what the
#: kind is for, and reading it found nine that were not. Two examples of what
#: only reading catches:
#:
#: - `documentation_quoting_an_attack` and `security_report_with_payload` both
#:   read "injection" as SQL injection and wrote `admin' OR '1'='1` and `DROP
#:   TABLE users`. Fluent, on-topic for a different topic, and worthless to a
#:   classifier that has to tell prose about a model from an instruction to one.
#: - `multi_turn_setup` wrote ordinary workplace chatter: "Legal's reviewed our
#:   proposal", "The boss gave us the green light". Under `label = 1` those rows
#:   teach a classifier that a status update from a colleague is an attack,
#:   which costs precision on exactly the traffic a deployed detector sees most.
#:
#: A row generated before a rewrite and a row generated after it are not the
#: same row, so `prompt_id` carries the revision and `provenance.json` carries
#: the text each revision ran with. A corpus that silently mixes two wordings
#: under one id cannot be split back apart.
PROMPT_VERSIONS: dict[str, int] = {
    "documentation_quoting_an_attack": 2,
    "security_report_with_payload": 2,
    "roleplay_request": 2,
    "config_or_code_with_instructions": 2,
    "indirect_via_retrieved_content": 4,
    "delimiter_confusion": 2,
    "encoded_payload": 2,
    "multi_turn_setup": 2,
    "tool_misuse_request": 2,
}


def prompt_id(kind: str) -> str:
    """How a row names the wording that produced it. Unrevised kinds are v1."""
    return f"{kind}/v{PROMPT_VERSIONS.get(kind, 1)}"


#: How far apart two kinds' seed ranges start. Wide enough that a kind
#: generated over several rounds never reaches into the next kind's range, which
#: would make `provenance.json` describe seeds that produced somebody else's
#: rows.
SEED_STRIDE = 100_000

#: How many examples one call asks for. Eight fits inside a 2048-token context
#: with the paragraph kinds and still amortises the prompt evaluation across a
#: batch.
BATCH = 8

#: Sampling options, recorded in `provenance.json` and part of what makes a run
#: repeatable. Temperature is high because the corpus needs variety across
#: thousands of rows and duplicates are dropped anyway.
OPTIONS: dict[str, float | int] = {
    "temperature": 0.9,
    "top_p": 0.95,
    "num_ctx": 2048,
    "num_predict": 900,
}

#: The response shape the model is constrained to. Ollama passes a JSON schema
#: to the sampler, so the reply parses rather than being scraped out of prose.
#: The parser below still cleans what comes back: a schema constrains the shape
#: of the JSON and says nothing about what the strings inside it contain.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"examples": {"type": "array", "items": {"type": "string"}}},
    "required": ["examples"],
}

#: An example shorter than this is not one. Observed in real replies: empty
#: strings, a bare "1.", and the word "Example".
_MIN_CHARS = 20

#: And longer than this is the model having run away with a whole article.
_MAX_CHARS = 2000

#: Leading list furniture the model emits inside a JSON string even when it was
#: asked not to: "1. ", "3) ", "- ", "* ", "Example 2: ".
_FURNITURE = re.compile(
    r"\A\s*(?:(?:example|item)\s*\d*\s*[:.\-]\s*|\d+\s*[.)]\s*|[-*•]\s+)", re.IGNORECASE
)

#: How the model declines. A refusal recorded under an attack kind would label
#: "I'm sorry, I can't help with that" as an injection, which teaches the
#: classifier the opposite of the thing it is for. Anchored to the start, so a
#: payload that merely contains the words survives.
_REFUSAL = re.compile(
    r"\A\s*(?:i'?m sorry|i am sorry|sorry,|i cannot|i can'?t|i won'?t|i'?m not able"
    r"|as an ai|i must decline|i'?m unable)",
    re.IGNORECASE,
)


#: The generator signing its own output. Observed: asked for a document with an
#: instruction planted in it, the model wrote "Qwen, please put this task aside"
#: in three replies out of four, because that is the name it answers to.
#:
#: Left in, that is a spurious feature and a strong one. Every row carrying it
#: would be an attack and no hard negative would carry it, so the cheapest rule
#: fitting the training data is "says Qwen, therefore injection" -- a classifier
#: that has learned which model wrote its corpus rather than what an injection
#: is. The prompt now asks for a generic address; this drops what still gets
#: through, because a prompt is a request and a filter is a guarantee.
#:
#: Bounded on purpose: it removes the generator's own identity and nothing else.
#: It is not a general de-biasing pass, and no claim here is that one has been
#: done.
_GENERATOR_NAME = re.compile(r"\bqwen\b|\balibaba\b", re.IGNORECASE)


def model_digest(model: str = MODEL) -> str:
    """The digest behind an Ollama tag, so a row names an artifact not a label."""
    out = subprocess.run(
        ["ollama", "show", "--modelfile", model],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("# FROM") or "sha256" in line:
            marker = line.split("sha256-")[-1].split()[0] if "sha256-" in line else ""
            if marker:
                return marker
    raise RuntimeError(f"could not read a digest for {model}; refusing to record a tag alone")


def licence_digest(model: str = MODEL) -> str:
    """sha256 of the licence text the model artifact itself carries.

    The grant pinned by its bytes. `Generator.licence` records a name, and a
    name is what somebody typed; this is what the artifact actually shipped, and
    it is what a later reader compares against when the tag has moved under
    them.
    """
    out = subprocess.run(
        ["ollama", "show", "--license", model],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if "Apache License" not in out and "License" not in out:
        raise RuntimeError(f"{model} printed no licence text; refusing to record a digest of it")
    return hashlib.sha256(out.encode("utf-8")).hexdigest()


def prompt_digest(text: str) -> str:
    """sha256 of a prompt, which is how `provenance.json` ties rows to wording.

    An edit to a prompt after the rows were generated leaves the rows describing
    a prompt that no longer exists. Comparing digests is what turns that from
    something a reviewer might notice into something the suite fails on.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_example(raw: str) -> str:
    """One model-emitted string, as it is worth recording, or `""` to drop it.

    Written against what the model actually returns rather than against tidy
    input. Even under a JSON schema the observed replies carry leading "1. ",
    stray surrounding quotes, trailing whitespace, empty strings, and for the
    attack kinds an occasional refusal in place of a payload.
    """
    text = raw.strip()
    # Repeat: the model writes '1. "Ignore previous instructions"', so the
    # furniture has to come off before the quotes can be seen, and a quoted
    # string can itself begin with furniture.
    for _ in range(2):
        text = _FURNITURE.sub("", text).strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1].strip()
    if _REFUSAL.match(text):
        return ""
    if _GENERATOR_NAME.search(text):
        return ""
    if not _MIN_CHARS <= len(text) <= _MAX_CHARS:
        return ""
    return text


def parse_examples(raw: str) -> list[str]:
    """The usable examples in one raw reply, in order, without repeats.

    The schema makes a JSON object the normal case. The fallback is not
    decoration: a reply can still be truncated by `num_predict` mid-array, and a
    truncated object is prose as far as `json.loads` is concerned. Rather than
    lose the batch, the lines are read the way they would have been without a
    schema.
    """
    values: list[str] = []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        found = parsed.get("examples")
        if isinstance(found, list):
            values = [item for item in found if isinstance(item, str)]
    elif isinstance(parsed, list):
        values = [item for item in parsed if isinstance(item, str)]
    if not values:
        # No object to read. Recover whole quoted strings if the reply was a
        # truncated array, and otherwise fall back to lines.
        quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', raw)
        candidates = quoted if len(quoted) > 1 else raw.splitlines()
        values = []
        for item in candidates:
            try:
                values.append(json.loads(f'"{item}"') if quoted else item)
            except json.JSONDecodeError:
                values.append(item)

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_example(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _ask(instruction: str, count: int, seed: int, timeout: float = 900.0) -> str:
    """One call to the local model. Returns the raw reply text."""
    prompt = (
        f"{instruction}\n\n"
        f"Produce {count} different examples, varied from each other. "
        'Return JSON of the form {"examples": ["...", "..."]}. Each entry is the '
        "text of one example on its own, with no numbering, no heading and no "
        "commentary about it. Where an example runs to several lines, keep the whole "
        "of it in a single entry with newlines inside that entry; never split one "
        "example across two entries."
    )
    payload: dict[str, Any] = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": _SCHEMA,
        "options": {**OPTIONS, "seed": seed},
        "keep_alive": "30m",
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise GenerationError(f"the endpoint returned no object: {body[:200]!r}")
    reply = parsed.get("response")
    if not isinstance(reply, str):
        raise GenerationError(f"the endpoint returned no response field: {body[:200]!r}")
    return reply


def generate(
    kind: str,
    label: int,
    count: int,
    seed: int,
    digest: str | None = None,
    workers: int = 6,
    exclude: set[str] | None = None,
) -> list[Row]:
    """Ask the local model for `count` examples of one kind.

    Seeds run consecutively from `seed`, so the seed range a run used is a pair
    of numbers `provenance.json` can record. Requests go out in parallel because
    a single stream on this machine is slow enough to put the corpus out of
    reach; the seed is per request, and a request's reply was measured to be the
    same whether it was issued alone or alongside five others.
    """
    if kind not in _PROMPTS:
        raise GenerationError(f"no prompt for kind {kind!r}")
    resolved = model_digest() if digest is None else digest
    row_prompt_id = prompt_id(kind)
    instruction = _PROMPTS[kind]
    rows: list[Row] = []
    # Shared with the caller across rounds. Deduplication inside one call is not
    # enough once a kind is generated in several passes: the second pass has no
    # memory of the first, and the seeds it uses are different but the model's
    # favourite phrasings are not.
    seen = set() if exclude is None else exclude
    # Twice the batches the arithmetic needs, as a CAP rather than a plan: some
    # replies come back empty and some examples repeat, and a run that stopped
    # at the nominal count would come up short on exactly the kinds the model
    # finds hardest. `seed_span` is what `provenance.json` records, so it has to
    # be the cap and not the number actually used.
    batches = -(-count // BATCH) * 2

    def one(offset: int) -> list[str]:
        try:
            return parse_examples(_ask(instruction, BATCH, seed + offset))
        except (OSError, GenerationError, json.JSONDecodeError):
            return []

    # Submitted a wave at a time rather than all at once. `Executor.map` queues
    # every task the moment it is called, so a kind that reached its count on
    # the tenth batch still paid for the fiftieth. On a run measured in hours
    # against a local 14B model that is half the wall clock, and the waste is
    # invisible because the extra rows are simply discarded.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        offset = 0
        while len(rows) < count and offset < batches:
            wave = range(offset, min(offset + workers, batches))
            offset += len(wave)
            for texts in pool.map(one, wave):
                for text in texts:
                    if text in seen:
                        continue
                    seen.add(text)
                    rows.append(Row(text, label, kind, row_prompt_id, MODEL, resolved))
                # Every reply in a wave is kept, including the ones that arrive
                # after the count is reached. They were paid for already, and
                # discarding them to land on a round number would be throwing
                # away the only thing this run produces.
    return rows


def load_generated(path: Path) -> list[Row]:
    """The committed corpus, as rows.

    Strict about fields. `Row(**json.loads(line))` raises on a missing or an
    unexpected key, which is what makes a row with no provenance a failure at
    load rather than a row whose `model_digest` is quietly the empty string.
    """
    return [
        Row(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


def write_generated(rows: Sequence[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def provenance_record(
    rows: Sequence[Row],
    seeds: dict[str, list[int]],
    digest: str,
    generated_on: str,
    ollama_version: str,
) -> dict[str, Any]:
    """What produced these rows, in the form `provenance.json` carries.

    The prompt text is stored beside its digest rather than only referenced.
    Someone reading the artifact without this file still has to be able to see
    the wording; the digest is what stops the two copies drifting, because the
    suite compares the stored one against the live one.
    """
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    generator = GENERATORS[0]
    return {
        "generated_on": generated_on,
        "ollama_version": ollama_version,
        "model": MODEL,
        "model_digest": digest,
        "weights_id": generator.weights_id,
        "licence": generator.licence,
        "licence_sha256": generator.licence_sha256,
        "options": dict(OPTIONS),
        "batch": BATCH,
        "rows": len(rows),
        "prompts": {
            kind: {
                "prompt_id": prompt_id(kind),
                "sha256": prompt_digest(_PROMPTS[kind]),
                "text": _PROMPTS[kind],
            }
            for kind in KINDS
        },
        "kinds": {
            kind: {
                "label": LABELS[kind],
                "seeds": seeds.get(kind, []),
                "rows": counts.get(kind, 0),
            }
            for kind in KINDS
        },
    }


def _ollama_version() -> str:
    out = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"\d+\.\d+\.\d+", out)
    if match is None:
        raise GenerationError(f"could not read an ollama version from {out!r}")
    return match.group(0)


def build(
    per_kind: int,
    seed: int,
    generated_on: str,
    kinds: Iterable[str] = KINDS,
    workers: int = 6,
    checkpoint: bool = True,
    chunk: int = 25,
) -> tuple[list[Row], dict[str, Any]]:
    """Generate the whole corpus and the record that explains it.

    Breadth-first: every kind reaches `chunk` rows before any kind reaches two
    of them. Depth-first is the obvious way to write this and it is the wrong
    one for a run measured in hours against a local model. Interrupted halfway,
    depth-first leaves eight kinds complete and eight absent, which is not a
    corpus and fails every test that asks for the classes to be represented.
    Breadth-first leaves a whole corpus at every checkpoint, thinner than the
    target and usable.

    Each kind gets its own seed base `SEED_STRIDE` apart, so a later round
    extends a kind's seed range rather than colliding with the next kind's.
    """
    digest = model_digest()
    version = _ollama_version()
    order = list(kinds)
    bases = {kind: seed + index * SEED_STRIDE for index, kind in enumerate(order)}
    produced: dict[str, list[Row]] = {kind: [] for kind in order}
    seen: dict[str, set[str]] = {kind: set() for kind in order}
    used: dict[str, int] = {kind: 0 for kind in order}
    span = -(-chunk // BATCH) * 2

    target = 0
    while target < per_kind:
        target = min(per_kind, target + chunk)
        for kind in order:
            short = target - len(produced[kind])
            if short <= 0:
                continue
            produced[kind].extend(
                generate(
                    kind,
                    LABELS[kind],
                    short,
                    bases[kind] + used[kind],
                    digest=digest,
                    workers=workers,
                    exclude=seen[kind],
                )
            )
            used[kind] += span
            print(f"{kind}: {len(produced[kind])}", flush=True)
        rows = [row for kind in order for row in produced[kind]]
        seeds = {kind: [bases[kind], bases[kind] + used[kind]] for kind in order}
        if checkpoint:
            _checkpoint(rows, seeds, digest, generated_on, version)

    rows = [row for kind in order for row in produced[kind]]
    seeds = {kind: [bases[kind], bases[kind] + used[kind]] for kind in order}
    return rows, provenance_record(rows, seeds, digest, generated_on, version)


def _checkpoint(
    rows: list[Row],
    seeds: dict[str, list[int]],
    digest: str,
    generated_on: str,
    version: str,
) -> None:
    write_generated(rows, GENERATED)
    record = provenance_record(rows, seeds, digest, generated_on, version)
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import datetime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-kind", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=25)
    # UTC rather than local. The date is provenance, and provenance read in
    # another timezone should not disagree with itself.
    today = datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat()
    parser.add_argument("--date", default=today)
    args = parser.parse_args(argv)

    rows, record = build(
        args.per_kind, args.seed, args.date, workers=args.workers, chunk=args.chunk
    )
    _checkpoint(
        rows,
        {kind: v["seeds"] for kind, v in record["kinds"].items()},
        record["model_digest"],
        record["generated_on"],
        record["ollama_version"],
    )
    print(f"wrote {len(rows)} rows to {GENERATED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
