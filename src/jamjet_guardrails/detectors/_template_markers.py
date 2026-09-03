"""Chat-template markers, read from pinned model repositories. GENERATED.

Do not edit. Regenerate with:

    ./.venv/bin/python scripts/generate_template_markers.py

`tests/test_template_markers.py` regenerates this module from the raw files
committed under `template-data/` and requires the result to be byte-identical
to what is committed here, so an edit made by hand is a failing test rather
than a table nobody can reproduce.

**What a marker is.** A bracketed run with no whitespace in it, three to
sixty-four characters long and containing at least one letter, that a listed
repository declares as a special token or that its chat template writes into
its own output. `<|im_start|>`, `[INST]`, `<<SYS>>`, `<start_of_turn>` and
`<｜User｜>` are the shapes this covers. These strings exist to delimit
turns, which is why one of them arriving inside untrusted content is a claim to
a role that content does not have.

**What is not a marker.** Two populations are removed, each by a property
rather than by a list, and each counted here so the cost of removing it can be
argued with.

Reserved vocabulary slots, whose name ends in a number, are dropped:
`<|reserved_special_token_0|>`, `[control_8]`, `<unused12>`. A tokenizer
allocates these in blocks to be named later, and Llama 3.1 did exactly that
when it renamed `<|reserved_special_token_2|>` to `<|python_tag|>`. No chat
template emits one. `RESERVED_SLOTS_DROPPED` records how many went: dropped
here, 1018. A detector that later wants them back can reach
them with the same rule rather than with a longer table.

Markers that are also HTML element names are excluded and kept in
`EXCLUDED_AS_HTML`, where a reader can see what the rule costs. Such a marker
is a boundary token rather than a role claim, and denying it would deny a
strikethrough tag in any ordinary HTML document. The element names come from
the element index of the HTML Standard itself, pinned at a commit and recorded
in `HTML_ELEMENT_SOURCE`, so the rule is a property of that index rather than
of the strings it happens to remove today. `corpora/NOTICE.md` records the same
exclusion. Excluded here, 2.

**The weakest two entries are named rather than quietly kept.**
`<function-name>` and `<args-json-object>` are written by the Qwen 2.5
tool-calling template into the system prompt it builds, as placeholders inside
a JSON example. They were read out of a real template and are kept for that
reason, but they are the two entries most likely to appear in ordinary
developer prose, and they are the first place to look if the check's precision
row disappoints. `corpora/NOTICE.md` says the same.

**Markers are stored as they were read.** No normalisation, no case folding,
no confusable skeleton. The detector that consumes this table matches over the
folded view described in the phase 3 design, and folds the table entries the
same way at load; folding them here would bake one interpreter's Unicode
version into a generated file and lose the string a source actually declares.

`detectors/template_integrity.py` is the only module that reads this table.
It folds every entry at load, matches over the folded view, and publishes its
own precision and recall on `corpora/template-integrity/in-repo.jsonl`.

Markers: 59. Model repositories read: 8.
"""

from __future__ import annotations

from typing import NamedTuple


class Source(NamedTuple):
    """One repository the table was read out of, pinned to one commit."""

    key: str
    """The name `MARKER_SOURCES` uses for this repository."""

    repository: str
    """What was actually fetched. A mirror, where the upstream is gated."""

    revision: str
    """The commit the files below were read at. Never a branch name."""

    licence: str
    licence_url: str

    upstream: str
    """The canonical repository this mirrors, or the empty string."""

    upstream_revision: str
    upstream_gated: bool
    """Whether an anonymous fetch of `upstream` is refused."""

    files: dict[str, str]
    """Path in the repository to the SHA-256 of the bytes that were read."""

    note: str


SOURCES: tuple[Source, ...] = (
    Source(
        key="llama-2-chat",
        repository="unsloth/llama-2-7b-chat",
        revision="a6d63d7c9ac31fd7e6d31e66ee0d1c784a489fcf",
        licence="LLAMA 2 Community License",
        licence_url="https://ai.meta.com/llama/license/",
        upstream="meta-llama/Llama-2-7b-chat-hf",
        upstream_revision="f5db02db724555f92da89c216ac04704f23d4590",
        upstream_gated=True,
        files={
            "tokenizer_config.json": "86888e77911253cae4f44b212d357045e6735ed6356622d740fd17ff83b81258",
            "special_tokens_map.json": "719833ff26ac897a3ec8ed946028a135de2a351470af59b4008744ab1f0ee9b7",
        },
        note="The mirror declares apache-2.0 on the Hub. The stricter upstream licence is recorded instead, because a mirror cannot relicense Meta's material. NousResearch/Llama-2-7b-chat-hf is the other non-gated mirror and was rejected: its tokenizer config predates the chat_template key, so it carries no [INST] or <<SYS>>.",
    ),
    Source(
        key="llama-3-instruct",
        repository="NousResearch/Meta-Llama-3-8B-Instruct",
        revision="53346005fb0ef11d3b6a83b12c895cca40156b6c",
        licence="Meta Llama 3 Community License",
        licence_url="https://llama.meta.com/llama3/license/",
        upstream="meta-llama/Meta-Llama-3-8B-Instruct",
        upstream_revision="8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        upstream_gated=True,
        files={
            "tokenizer_config.json": "da0e3a7cce6e4d787e85eb1c24d548420e0d7fe2c7a214e192795c46e40d75bb",
            "special_tokens_map.json": "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc",
        },
        note="",
    ),
    Source(
        key="qwen-2.5-instruct",
        repository="Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        licence="Apache-2.0",
        licence_url="https://www.apache.org/licenses/LICENSE-2.0",
        upstream="",
        upstream_revision="",
        upstream_gated=False,
        files={
            "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
        },
        note="No special_tokens_map.json exists at this revision; 404.",
    ),
    Source(
        key="mistral-instruct",
        repository="mistralai/Mistral-7B-Instruct-v0.3",
        revision="c170c708c41dac9275d15a8fff4eca08d52bab71",
        licence="Apache-2.0",
        licence_url="https://www.apache.org/licenses/LICENSE-2.0",
        upstream="",
        upstream_revision="",
        upstream_gated=False,
        files={
            "tokenizer_config.json": "0533dec9cfe319163801b6618d0f3ec9cfa126b6288e3df5deca6e32acb09cd2",
            "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
        },
        note="",
    ),
    Source(
        key="gemma-2-instruct",
        repository="unsloth/gemma-2-9b-it",
        revision="fc7d4737cda11c3a19af2b722319e846670b4d89",
        licence="Gemma Terms of Use",
        licence_url="https://ai.google.dev/gemma/terms",
        upstream="google/gemma-2-9b-it",
        upstream_revision="11c9b309abf73637e4b6f9a3fa1e92e615547819",
        upstream_gated=True,
        files={
            "tokenizer_config.json": "e5ac0f1cfb5f450763d368edacbb2448a3c02c8d8f7f98d963a820751fb0ddc3",
            "special_tokens_map.json": "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
        },
        note="",
    ),
    Source(
        key="phi-3-instruct",
        repository="microsoft/Phi-3-mini-4k-instruct",
        revision="f39ac1d28e925b323eae81227eaba4464caced4e",
        licence="MIT",
        licence_url="https://opensource.org/license/mit",
        upstream="",
        upstream_revision="",
        upstream_gated=False,
        files={
            "tokenizer_config.json": "32f66c2bab499baaa8341819ad9a13342f957501ce1e989bcb90853a01336cc0",
            "special_tokens_map.json": "810adc6e6c6ef2f56c285ef930d243358a3a9f05e36a01c5a10bafc6fac4609b",
        },
        note="",
    ),
    Source(
        key="deepseek-v3",
        repository="deepseek-ai/DeepSeek-V3",
        revision="e815299b0bcbac849fa540c768ef21845365c9eb",
        licence="MIT",
        licence_url="https://opensource.org/license/mit",
        upstream="",
        upstream_revision="",
        upstream_gated=False,
        files={
            "tokenizer_config.json": "637bcd1a08cf7c772ce6a383196b22930921c79c3c73223c5afc0c7f41545546",
        },
        note="LICENSE-CODE in the repository is MIT and covers the repository's code and configuration; LICENSE-MODEL covers the weights, which are not read here. No special_tokens_map.json exists; 404.",
    ),
    Source(
        key="gpt-2",
        repository="openai-community/gpt2",
        revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        licence="MIT",
        licence_url="https://opensource.org/license/mit",
        upstream="",
        upstream_revision="",
        upstream_gated=False,
        files={
            "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
            "onnx/special_tokens_map.json": "6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad",
        },
        note="The root tokenizer_config.json at this revision is 26 bytes and names no token. The end-of-text token is read from the ONNX export's special_tokens_map.json, committed in the same repository at the same revision; there is no root special_tokens_map.json to read it from.",
    ),
)
"""Every model repository the markers were read out of."""

HTML_ELEMENT_SOURCE: Source = Source(
    key="html-elements",
    repository="w3c/webref",
    revision="f3b81966c45f34f62df20e7f8d6f66d5b5ba9279",
    licence="MIT",
    licence_url="https://opensource.org/license/mit",
    upstream="",
    upstream_revision="",
    upstream_gated=False,
    files={
        "ed/elements/html.json": "56030c8bb725c6009e17ca85ef729aa4cecb9f51926a9ef870f36c5ecb37dfd0",
    },
    note="A curated extraction of the element index of the WHATWG HTML Standard.",
)
"""The element index that decides which markers are HTML tags."""

MARKERS: tuple[str, ...] = (
    "</tool_call>",
    "</tool_response>",
    "</tools>",
    "<</SYS>>",
    "<<SYS>>",
    "<args-json-object>",
    "<bos>",
    "<end_of_turn>",
    "<eos>",
    "<function-name>",
    "<pad>",
    "<start_of_turn>",
    "<tool_call>",
    "<tool_response>",
    "<tools>",
    "<unk>",
    "<|assistant|>",
    "<|begin_of_text|>",
    "<|box_end|>",
    "<|box_start|>",
    "<|end_header_id|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "<|end|>",
    "<|eot_id|>",
    "<|im_end|>",
    "<|im_start|>",
    "<|image_pad|>",
    "<|object_ref_end|>",
    "<|object_ref_start|>",
    "<|quad_end|>",
    "<|quad_start|>",
    "<|start_header_id|>",
    "<|system|>",
    "<|user|>",
    "<|video_pad|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|vision_start|>",
    "<｜Assistant｜>",
    "<｜User｜>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜tool▁calls▁begin｜>",
    "<｜tool▁calls▁end｜>",
    "<｜tool▁call▁begin｜>",
    "<｜tool▁call▁end｜>",
    "<｜tool▁outputs▁begin｜>",
    "<｜tool▁outputs▁end｜>",
    "<｜tool▁output▁begin｜>",
    "<｜tool▁output▁end｜>",
    "<｜tool▁sep｜>",
    "[/AVAILABLE_TOOLS]",
    "[/INST]",
    "[/TOOL_RESULTS]",
    "[AVAILABLE_TOOLS]",
    "[INST]",
    "[TOOL_CALLS]",
    "[TOOL_RESULTS]",
)
"""Every marker, sorted. The table the check matches against."""

MARKER_SOURCES: dict[str, tuple[str, ...]] = {
    "</tool_call>": ("qwen-2.5-instruct",),
    "</tool_response>": ("qwen-2.5-instruct",),
    "</tools>": ("qwen-2.5-instruct",),
    "<</SYS>>": ("llama-2-chat",),
    "<<SYS>>": ("llama-2-chat",),
    "<args-json-object>": ("qwen-2.5-instruct",),
    "<bos>": ("gemma-2-instruct",),
    "<end_of_turn>": ("gemma-2-instruct",),
    "<eos>": ("gemma-2-instruct",),
    "<function-name>": ("qwen-2.5-instruct",),
    "<pad>": ("gemma-2-instruct",),
    "<start_of_turn>": ("gemma-2-instruct",),
    "<tool_call>": ("qwen-2.5-instruct",),
    "<tool_response>": ("qwen-2.5-instruct",),
    "<tools>": ("qwen-2.5-instruct",),
    "<unk>": ("llama-2-chat", "mistral-instruct", "gemma-2-instruct", "phi-3-instruct"),
    "<|assistant|>": ("phi-3-instruct",),
    "<|begin_of_text|>": ("llama-3-instruct",),
    "<|box_end|>": ("qwen-2.5-instruct",),
    "<|box_start|>": ("qwen-2.5-instruct",),
    "<|end_header_id|>": ("llama-3-instruct",),
    "<|end_of_text|>": ("llama-3-instruct",),
    "<|endoftext|>": ("qwen-2.5-instruct", "phi-3-instruct", "gpt-2"),
    "<|end|>": ("phi-3-instruct",),
    "<|eot_id|>": ("llama-3-instruct",),
    "<|im_end|>": ("qwen-2.5-instruct",),
    "<|im_start|>": ("qwen-2.5-instruct",),
    "<|image_pad|>": ("qwen-2.5-instruct",),
    "<|object_ref_end|>": ("qwen-2.5-instruct",),
    "<|object_ref_start|>": ("qwen-2.5-instruct",),
    "<|quad_end|>": ("qwen-2.5-instruct",),
    "<|quad_start|>": ("qwen-2.5-instruct",),
    "<|start_header_id|>": ("llama-3-instruct",),
    "<|system|>": ("phi-3-instruct",),
    "<|user|>": ("phi-3-instruct",),
    "<|video_pad|>": ("qwen-2.5-instruct",),
    "<|vision_end|>": ("qwen-2.5-instruct",),
    "<|vision_pad|>": ("qwen-2.5-instruct",),
    "<|vision_start|>": ("qwen-2.5-instruct",),
    "<｜Assistant｜>": ("deepseek-v3",),
    "<｜User｜>": ("deepseek-v3",),
    "<｜begin▁of▁sentence｜>": ("deepseek-v3",),
    "<｜end▁of▁sentence｜>": ("deepseek-v3",),
    "<｜tool▁calls▁begin｜>": ("deepseek-v3",),
    "<｜tool▁calls▁end｜>": ("deepseek-v3",),
    "<｜tool▁call▁begin｜>": ("deepseek-v3",),
    "<｜tool▁call▁end｜>": ("deepseek-v3",),
    "<｜tool▁outputs▁begin｜>": ("deepseek-v3",),
    "<｜tool▁outputs▁end｜>": ("deepseek-v3",),
    "<｜tool▁output▁begin｜>": ("deepseek-v3",),
    "<｜tool▁output▁end｜>": ("deepseek-v3",),
    "<｜tool▁sep｜>": ("deepseek-v3",),
    "[/AVAILABLE_TOOLS]": ("mistral-instruct",),
    "[/INST]": ("llama-2-chat", "mistral-instruct"),
    "[/TOOL_RESULTS]": ("mistral-instruct",),
    "[AVAILABLE_TOOLS]": ("mistral-instruct",),
    "[INST]": ("llama-2-chat", "mistral-instruct"),
    "[TOOL_CALLS]": ("mistral-instruct",),
    "[TOOL_RESULTS]": ("mistral-instruct",),
}
"""Every marker, to the keys of the SOURCES that declare it."""

EXCLUDED_AS_HTML: dict[str, tuple[str, ...]] = {
    "</s>": ("llama-2-chat", "mistral-instruct"),
    "<s>": ("llama-2-chat", "mistral-instruct", "phi-3-instruct"),
}
"""Candidates an HTML element name removed, and what declared them."""

RESERVED_SLOTS_DROPPED: int = 1018
"""Reserved vocabulary slots the name-ends-in-a-number rule dropped."""
