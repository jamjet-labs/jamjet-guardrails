# Migrating from llm-guard

`protectai/llm-guard` was archived on 9 July 2026 and its last release, 0.3.16,
went to PyPI on 19 May 2025. This page maps every scanner in that release onto
what jamjet-guardrails does, marks each one `mapped`, `partial` or `gap`, and
says in one sentence what is different. It is written to be checkable rather
than persuasive: the counts below are held by
`tests/test_migration_guide.py`, which parses this page's own table, and every
replacement named in it has to be a check this library can actually build.

There is no coverage percentage anywhere on this page. Six of thirty-seven is
not a coverage figure, because the scanners are not interchangeable units:
twenty-three of them are a model making a judgment, and this library does not
make judgments. The arithmetic is the claim.

## The numbers, first

| Status | Input | Output | Total |
|---|---:|---:|---:|
| mapped | 3 | 3 | 6 |
| partial | 5 | 4 | 9 |
| gap | 7 | 15 | 22 |
| all scanners | 15 | 22 | 37 |

Of those 37 scanner classes, 23 need a model and 14 do not. The 14 are where
this migration is worth doing. The 23 are what you give up, and the section
after next says so in detail rather than leaving you to discover it a row at a
time.

## The install, which is the whole argument

`llm-guard` declares these as unconditional runtime dependencies in its
`pyproject.toml` at 0.3.16, not as extras:

```
bc-detect-secrets==1.5.43   faker>=37,<38          fuzzysearch>=0.7,<0.9
json-repair==0.44.1         nltk>=3.9.1,<4         presidio-analyzer==2.2.358
presidio-anonymizer==2.2.358                       regex==2024.11.6
structlog>=24               tiktoken>=0.9,<1.0     torch>=2.4.0
transformers==4.51.3
```

Torch, transformers, and both halves of Presidio are installed whether or not
you ever construct a scanner that uses them. Somebody who only ever ran
`BanSubstrings`, which is a case-insensitive substring match with no model and
no data file, still installed torch.

jamjet-guardrails declares zero runtime dependencies. That is checked against
the built distribution metadata rather than against `pyproject.toml`, by
`tests/test_packaging.py::test_the_installed_distribution_declares_no_runtime_dependencies`,
so a dependency added later fails the build instead of the sentence going
quietly out of date.

The second half of the archive matters as much as the first. The notice on the
archived repository covers the models as well as the code: "This project and
its associated models on Hugging Face are no longer under active development or
maintained." The default weights for `PromptInjection` and `NoRefusal` sit on
that account, and so do the ONNX builds for `BanCode`, `BanCompetitors`,
`BanTopics`, `Language`, `Toxicity`, `Bias` and `MaliciousURLs`, which are what
`use_onnx=True` fetches. The remaining defaults are on unrelated accounts that
the notice does not cover and that nobody has promised to maintain either.

## Why the count is 37 and not 36, or 38, or 35

Four different ways of counting are all defensible and none of them agree, so
this page says which one it uses.

| How you count | What you get |
|---|---|
| Exported scanner classes at v0.3.16, the final release | **37** (15 input, 22 output) |
| Scanner modules at v0.3.16 | 36 |
| Scanner modules on the archived `main` branch | 38 |
| Pages in the documentation navigation | 35 |

This page counts **exported classes at the final PyPI release**, read from the
`__all__` lists in `llm_guard/input_scanners/__init__.py` and
`llm_guard/output_scanners/__init__.py`.

- **36 versus 37** is one module. `output_scanners/no_refusal.py` exports two
  classes: `NoRefusal`, a transformers classifier, and `NoRefusalLight`, a
  `BanSubstrings` subclass with 28 hardcoded refusal phrases. Both are in
  `__all__` and you construct one or the other. Collapsing them would hide the
  one axis this whole page is organised around: one of the two is a model and
  maps to nothing here, the other is pure substring matching and reproduces
  exactly.
- **38** is the archived branch, which carries 12 commits past the tag,
  including an `EmotionDetection` scanner on both sides that was never released
  to PyPI. If you counted directories on GitHub, that is where the extra came
  from.
- **35** is the documentation navigation, which omits the output `BanCode` page
  and both `EmotionDetection` pages. Do not source a count from it.

## What you lose

A guide that only listed wins would be worth exactly as much as its first
checkable row. So, plainly:

**Twenty-three scanners are a model making a judgment about meaning, and
nothing here replaces them.** `PromptInjection`, `Toxicity`, `BanTopics`,
`Gibberish`, `Bias`, `FactualConsistency`, `Relevance`, `NoRefusal`,
`Language`, `LanguageSame`, `MaliciousURLs`, `BanCode`, `Code`,
`BanCompetitors` and the NER half of `Anonymize` and `Sensitive` all decide
something about what the text means. That list is the 23, counted as classes:
sixteen names, seven of which llm-guard ships on both sides. This
library classifies nothing. If a model's judgment is what you were buying, keep
buying it somewhere; the honest migration for those scanners is to a maintained
model, not to here.

**No regular expression finds a person's name.** `Anonymize` and `Sensitive`
run Presidio with a DeBERTa NER model behind it, so they catch `PERSON`,
`LOCATION`, `ORGANIZATION` and `DATE_TIME`. The `pii` check here is four regex
types. Names are the concrete loss and there is no way to dress it up.

**Redaction here is one way.** There is no `Vault` and no `Deanonymize`. A
placeholder cannot be turned back into the value it replaced, by design; a
reversible store is a different product with a different threat model.

**Secrets is narrower on purpose.** Seven prefix-anchored families against
llm-guard's 110 configured detect-secrets plugins, several of which are entropy
detectors. Prefix anchoring is why the published precision figure is
defensible, and it is also why `github_pat_` and `xapp-` are named misses in
`README.md`.

**No token counting and no network calls.** `TokenLimit` counts tiktoken
tokens; `Limits` counts characters, bytes and lines and will not estimate a
token count from them. `URLReachability` fetches every URL in a reply; nothing
here opens a socket, and a test over the shipped source holds that.

## What the 14 model-free scanners buy you

Fourteen classes never needed a model: `BanSubstrings` twice, `Regex` twice,
`InvisibleText`, `Secrets`, `Deanonymize`, `JSON`, `NoRefusalLight`,
`ReadingTime`, `URLReachability`, `Sentiment` twice and `TokenLimit`. Eleven of
them need no model file and no data file at all; two need an NLTK lexicon and
one a tiktoken encoding.

Line those fourteen up against the table below and the shape of this migration
is clear. All six `mapped` rows are model-free scanners, and three more
(`Secrets`, `TokenLimit`, `ReadingTime`) are model-free and `partial`. Of the
five model-free scanners that are still gaps, `Deanonymize` and
`URLReachability` are deliberate non-goals, `Sentiment` on both sides is a
judgment about tone, and `JSON` is simply not built. No model-backed scanner is
`mapped`, because mapping one would mean shipping a model.

Moving those to this library buys four things llm-guard did not offer for any
scanner, model-backed or not:

- **Typed findings with exact spans.** An llm-guard scanner returns
  `(text, is_valid, risk_score)`: a float, with no finding type and no
  position. Every verdict here carries typed findings with character spans, so
  a redaction can be applied and an audit record can say what was removed from
  where.
- **Published precision and recall per check**, measured on a committed corpus
  and gated in CI, with the misses named. llm-guard published latency
  benchmarks, not accuracy per scanner.
- **A written porting contract.** `docs/conformance.md` specifies the verdict
  fields, the combination order and the corpus schema, so an implementation in
  another language can be graded against the same corpora.
- **Zero dependencies**, which is the same sentence as the section above and is
  worth repeating exactly once.

## The mapping

`mapped` means the behaviour reproduces, with the differences named. `partial`
means the intent survives and the mechanism changes, so read the sentence
before you rely on it. `gap` means nothing here does this.

| Scanner | Direction | Status | Replacement | What differs |
|---|---|---|---|---|
| `Anonymize` | input | partial | `pii` | Both find and redact personal data, but `pii` is four regex types where llm-guard's default is thirteen entity types with NER behind them, and there is no vault, so redaction is one way. |
| `BanCode` | input | gap | none | Needs a classifier that decides whether text is source code. |
| `BanCompetitors` | input | partial | `rules` | A fixed case-insensitive list of names reproduces; the NER generalisation does not, so an unlisted competitor, a misspelling or an inflected form passes. |
| `BanSubstrings` | input | mapped | `rules` | Loses `match_type="word"` and `contains_all`; folds with `str.casefold` rather than `str.lower`, and gives back a span into the original text. |
| `BanTopics` | input | gap | none | Zero-shot classification of the prompt against a topic list. |
| `Code` | input | gap | none | Identifies which programming language a snippet is written in. |
| `Gibberish` | input | gap | none | A classifier for text that is not language. |
| `InvisibleText` | input | mapped | `injection-structural` | A different policy, not just a better one: llm-guard bans every Cf, Co and Cn code point and silently deletes them, so a soft hyphen and an emoji joiner are violations, while this reports typed findings with spans, treats bidi controls as balanced or unbalanced rather than banned, and never rewrites on a deny. |
| `Language` | input | gap | none | Detects natural language against ISO 639-1 codes. `script-constraint` is not this: script and language are different axes, and Spanish and English both pass a Latin constraint. |
| `PromptInjection` | input | partial | `injection-structural` | Covers the encoding channel only, so plain-text "ignore all previous instructions" is not caught; it does see what the classifier cannot, because a transformer tokenizer collapses a tag-character run to one unknown token. |
| `Regex` | input | mapped | `rules` | Loses `is_blocked=False`, so patterns are a block list and never an allow list, and loses `match_type="fullmatch"`; gains refusal at construction of a pattern that matches the empty string or nests unbounded repeats. |
| `Secrets` | input | partial | `secrets` | Seven prefix-anchored families against 110 configured detect-secrets plugins including entropy detectors; the anchoring is what makes the published precision defensible. |
| `Sentiment` | input | gap | none | A VADER lexicon score, which is a judgment about tone. |
| `TokenLimit` | input | partial | `rules` | Same intent, different unit and different action: `Limits` counts characters, bytes and lines, never tokens, and denies or redacts rather than chunking. |
| `Toxicity` | input | gap | none | A multi-label toxicity classifier. |
| `BanCode` | output | gap | none | As input. |
| `BanCompetitors` | output | partial | `rules` | As input. Every check here declares both directions, so it is the same object. |
| `BanSubstrings` | output | mapped | `rules` | As input. |
| `BanTopics` | output | gap | none | As input. |
| `Bias` | output | gap | none | Scores the reply for bias. |
| `Code` | output | gap | none | As input. |
| `Deanonymize` | output | gap | none | A deliberate non-goal: redaction here is one way and there is no vault to restore from. |
| `FactualConsistency` | output | gap | none | An entailment score between the prompt and the reply. |
| `Gibberish` | output | gap | none | As input. |
| `JSON` | output | gap | none | No JSON validation and no JSON repair. |
| `Language` | output | gap | none | As input. |
| `LanguageSame` | output | gap | none | Checks the reply is in the same language as the prompt. |
| `MaliciousURLs` | output | partial | `url-exfiltration` | A different question: llm-guard classifies a URL as malicious, while this asks whether a link is a data channel, so it has no host reputation, makes no network call, and will not flag a plain link to a known bad domain. |
| `NoRefusal` | output | gap | none | A classifier that detects the model refused. |
| `NoRefusalLight` | output | mapped | `rules` | It is `BanSubstrings` with a canned list, so it reproduces exactly, but the 28 phrases are yours to supply and are printed below. |
| `ReadingTime` | output | partial | `rules` | A size cap in a different unit, characters or bytes or lines against words per minute, and this never truncates. |
| `Regex` | output | mapped | `rules` | As input. Note that llm-guard's output default `match_type` is SEARCH where its input default is ALL. |
| `Relevance` | output | gap | none | Cosine similarity between prompt and reply embeddings. |
| `Sensitive` | output | partial | `pii` | As `Anonymize`, minus the vault. |
| `Sentiment` | output | gap | none | As input. |
| `Toxicity` | output | gap | none | As input. |
| `URLReachability` | output | gap | none | A deliberate non-goal: it fetches every URL in the reply, and nothing here opens a socket. |

## Worked replacements

Every `mapped` and every `rules`-backed `partial` row is one `build` call.

```py
from jamjet_guardrails import Context, Limits, build

guard = build(
    "rules",
    patterns={"TICKET_ID": r"\bJIRA-\d{4,}\b"},
    banned={"COMPETITOR": ("northwind", "initech")},
    limits=Limits(max_chars=20_000),
    on_match="deny",
)
guard.check("see JIRA-1234", Context(direction="input", origin="user"))
```

`patterns` replaces `Regex`, `banned` replaces `BanSubstrings` and
`BanCompetitors`, `limits` replaces `TokenLimit` and `ReadingTime` in a unit
you can count. `on_match` takes a decision, or one decision per direction.

### NoRefusalLight, in full

Nothing here ships a refusal list, so migrating that scanner means carrying its
28 phrases yourself. They are matched case-insensitively as plain substrings,
which is what `banned` does:

```py
REFUSALS = (
    "I'm sorry", "Sorry", "I am sorry", "I apologize",
    "As an", "As an AI", "I'm an", "I'm just",
    "As a language model", "As an Assistant", "I cannot", "I do not",
    "Hello!", "is illegal and unethical", "I'm really sorry", "OpenAI",
    "unethical", "not ethical", "not legal", "My apologies",
    "but I cannot", "If you have any other non-malicious requests",
    "I'm not able to provide", "I am really sorry",
    "I cannot create", "I cannot support", "I can't provide", "I can't assist",
)

guard = build("rules", banned={"REFUSAL": REFUSALS}, on_match="deny")
```

That list is copied from `output_scanners/no_refusal.py` at v0.3.16. It is a
blunt instrument in either library: `Sorry`, `unethical` and `Hello!` fire on
ordinary replies, and llm-guard shipped it that way.

## A note on Guardrails AI, which is a different situation

These two are often mentioned together and should not be. `guardrails-ai/guardrails`
is **not archived and is actively developed**. What changed there is
distribution: on 25 August 2026 the `guardrails hub install` path, the private
validator registry behind it, and the hosted remote inference were shut down,
and validators became plain PyPI packages you install with `pip`. The
catalogue page still resolves. The canonical statement is
[HUB_UPDATE.md](https://github.com/guardrails-ai/guardrails/blob/main/HUB_UPDATE.md)
in that repository, which is worth citing in preference to the announcement
issue, since the issue contradicts itself on the date.

The two events are unrelated in kind and land on the same people: one library
stopped, another changed how you install it, and both send the same population
looking for something they can `pip install` and keep.

## Where to check the rest

- `README.md` for what each check catches and the published precision and
  recall behind it.
- `BENCHMARKS.md` for the per-type scores and the worst misses.
- `corpora/NOTICE.md` for what every corpus is, where it came from, and the
  cases each check is known to get wrong.
- `docs/conformance.md` for the porting contract, if you are reimplementing
  rather than adopting.
- `CONTRIBUTING.md` if a `gap` row above is one you want to close.

## Sources

Everything above about llm-guard was read from the source tree at the tag, not
from documentation prose.

- <https://github.com/protectai/llm-guard> and the archive notice on it
- <https://github.com/protectai/llm-guard/tree/v0.3.16/llm_guard/input_scanners>
- <https://github.com/protectai/llm-guard/tree/v0.3.16/llm_guard/output_scanners>
- <https://github.com/protectai/llm-guard/blob/v0.3.16/pyproject.toml>
- <https://pypi.org/project/llm-guard/>
- <https://github.com/guardrails-ai/guardrails/blob/main/HUB_UPDATE.md>
