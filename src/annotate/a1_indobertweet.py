"""A1 - IndoBERTweet fine-tuned on Ibrohim & Budi's multilabel corpus.

**A1 EMITS A HarmSignal AND NEVER A Label.** (AGENTS.md TRAP 1 / D29.)

Ibrohim & Budi's 12 columns are `HS`, `Abusive` and ten `HS_*` subtypes. There is no
class for non-hateful negative content, so "not hate" is an *abstention on the semantic
label* that still carries negative evidence on the harm axis. An earlier version of this
project mapped A1 into the 5-label schema anyway; "no harm found" became a positive vote
for `benign_other`, outvoted A3 two-to-one, made `C(f)` identically zero and made demo
scene 2 unproducible. That is the single most important correction in the project's
history and it must not be re-introduced.

`Label` is deliberately not imported anywhere in this module, and
`tests/test_annotate.py` asserts that it never enters the module namespace.

R4: the native binary-column dict is the only thing this class produces on its own; the
mapping to a `HarmSignal` happens exclusively in `adapters.adapt_a1`, and the raw dict is
stored beside the mapped signal on every vote.

Backends
--------
`transformers`  a real fine-tuned checkpoint, batched on the GPU (lazy import).
`stub`          a deterministic keyword stand-in so the pipeline runs with no model on
                disk. It prints a loud banner: its output is NOT model output.

What a real run needs is documented on `A1IndoBERTweet` and in `docs/05-build-contract.md`.
"""
from __future__ import annotations

import hashlib
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..schemas import Comment, HarmSignal, HarmVote
from .adapters import A1_HARM_KEYS, adapt_a1

#: IndoBERTweet's OWN expected input format (its model card, verbatim): "lower-case all
#: words", "converting user mentions and URLs into @USER and HTTPURL respectively", and
#: "translating emoticons into text using the emoji package". This is deliberately
#: SEPARATE from `src/normalize/text.py`'s pipeline-wide normalizer, which keeps emoji
#: as unicode on purpose (it is tone-bearing evidence the whole pipeline needs) and
#: only strips LEADING mentions, not every mention. `tweet_preprocess` is applied here,
#: once, immediately before tokenization - never written back to `text_normalized` -
#: and the fine-tuning script imports this exact function so training and inference
#: see the same transformation. Two preprocessing layers with different jobs; drifting
#: them apart would mean the classifier trains on one distribution and serves another.
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w{1,30}")
#: The source dataset anonymises handles as the literal word `USER` (no `@`) rather
#: than a real mention - see `docs/06-a1-indobertweet.md`. Folding it to `@USER` aligns
#: training with the token IndoBERTweet actually saw in pretraining. On real pipeline
#: text this is a safe no-op: "USER" as a bare capitalised word does not occur in
#: normal Indonesian comment text.
_ANON_USER_RE = re.compile(r"(?<![@\w])USER(?![\w])")


def tweet_preprocess(text: str) -> str:
    """IndoBERTweet's documented input format. See the module-level note above."""
    if not text:
        return ""
    s = _ANON_USER_RE.sub("@USER", text)
    s = _URL_RE.sub("HTTPURL", s)
    s = _MENTION_RE.sub("@USER", s)
    try:
        import emoji                                    # lazy: only needed for A1
        s = emoji.demojize(s, delimiters=(" :", ": "))
    except ImportError:
        pass                                            # degrades to unicode emoji, not fatal
    return s.lower()

#: The native Ibrohim & Budi multilabel columns, in the canonical dataset order.
#: The first nine are `A1_HARM_KEYS`; the three strength columns are severity, not
#: additional harm evidence, so `adapt_a1` ignores them.
IBROHIM_BUDI_COLUMNS: tuple[str, ...] = (
    "HS", "Abusive",
    "HS_Individual", "HS_Group",
    "HS_Religion", "HS_Race", "HS_Physical", "HS_Gender", "HS_Other",
    "HS_Weak", "HS_Moderate", "HS_Strong",
)

STUB_MODEL_ID = "stub://a1-deterministic-keyword-stand-in"

_BANNER = """
################################################################################
##                                                                            ##
##   A1 IS RUNNING ON THE  * S T U B *  BACKEND                               ##
##   THESE ARE NOT MODEL PREDICTIONS. NOTHING HERE IS A REAL RESULT.          ##
##                                                                            ##
##   A hand-written keyword rule is standing in for IndoBERTweet so the       ##
##   pipeline can run end to end with no checkpoint on disk. Do not quote,    ##
##   screenshot or report any number derived from this run.                   ##
##                                                                            ##
##   Real run:  --backend transformers  with                                  ##
##   annotators.a1_indobertweet.checkpoint pointing at a checkpoint that has  ##
##   been FINE-TUNED on Ibrohim & Budi. The bare base model has a randomly    ##
##   initialised classification head and emits noise.                         ##
##                                                                            ##
################################################################################
"""

#: Fake. Used only by the stub backend. Deliberately small and obvious so that nobody
#: mistakes stub output for a classifier's decision boundary.
_STUB_HARM_SEEDS: tuple[str, ...] = (
    "goblok", "tolol", "bego", "bodoh", "idiot", "dungu",
    "anjing", "anjir", "bangsat", "babi", "monyet", "kampret", "bajingan",
    "kontol", "memek", "ngentot", "bispak",
    "sampah", "tai", "taik", "bacot", "mampus",
    "mati aja", "mending mati", "bunuh",
)
_STUB_INDIVIDUAL_MARKERS: tuple[str, ...] = ("lu ", "lo ", "elu", "kamu", "kau", "situ", "dia ", " mu ")
_STUB_GROUP_MARKERS: tuple[str, ...] = ("kalian", "mereka", "kaum", "orang-orang")


def _print_stub_banner() -> None:
    """Loud on stdout *and* stderr, so it survives a redirect and lands in screenshots."""
    print(_BANNER, flush=True)
    print(_BANNER, file=sys.stderr, flush=True)


def _sha256_path(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        read = 0
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            read += len(chunk)
            if limit_bytes is not None and read >= limit_bytes:
                break
    return h.hexdigest()[:16]


def _hash_checkpoint_dir(path: Path) -> str | None:
    """Identity of a local checkpoint: sha over per-file (name, sha) pairs."""
    if not path.exists():
        return None
    if path.is_file():
        return _sha256_path(path)
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(f.relative_to(path).as_posix().encode("utf-8"))
        h.update(_sha256_path(f).encode("ascii"))
    return h.hexdigest()[:16]


class A1IndoBERTweet:
    """IndoBERTweet fine-tuned on Ibrohim & Budi. Implements `base.HarmAnnotator`.

    What a *real* run requires:

    * a checkpoint fine-tuned on the Ibrohim & Budi multilabel set with a 12-output
      sigmoid head (`indolem/indobertweet-base-uncased` is the base, ~440 MB; the bare
      base model is refused here because its head is randomly initialised);
    * `torch` + `transformers` installed (neither is imported unless this backend runs);
    * ~1.5 GB of VRAM at `batch_size=32`, `max_length=128`, fp16 - trivial on a 16 GB
      4080. Throughput is roughly 1-2k comments/second, so a 25k-comment run is seconds.

    A1 never sees `Label`. It produces the native binary dict and hands it to `adapt_a1`.
    """

    can_vote_criticism = False          # TRAP 1. Structural, not a configuration choice.

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        backend: str = "transformers",
        *,
        annotator_id: str = "a1",
        batch_size: int = 32,
        max_length: int = 128,
        threshold: float = 0.5,
        device: str | None = None,
        revision: str | None = None,
        base_model: str = "indolem/indobertweet-base-uncased",
        hash_weights: bool = True,
    ) -> None:
        if backend not in ("transformers", "stub"):
            raise ValueError(f"A1 backend must be 'transformers' or 'stub', got {backend!r}")
        self.annotator_id = annotator_id
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.backend = backend
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.threshold = float(threshold)
        self.device = device
        self.revision = revision
        self.base_model = base_model
        self.hash_weights = hash_weights
        self._resolved_revision: str | None = revision
        self._model: Any = None
        self._tokenizer: Any = None
        self._columns: tuple[str, ...] = IBROHIM_BUDI_COLUMNS

        if self.backend == "stub":
            _print_stub_banner()
        elif not self.checkpoint:
            raise ValueError(
                "A1 needs a FINE-TUNED checkpoint. Set "
                "annotators.a1_indobertweet.checkpoint in the config to a checkpoint "
                "trained on Ibrohim & Budi, or run with --backend stub. The bare base "
                f"model ({self.base_model}) has a randomly initialised classification "
                "head and would emit noise into the record."
            )

    # -- the annotator protocol ------------------------------------------------

    def annotate(self, comments: Sequence[Comment]) -> list[HarmVote]:
        texts = [_text_of(c) for c in comments]
        predictions = self.predict_raw(texts)
        votes: list[HarmVote] = []
        for comment, (raw, confidence) in zip(comments, predictions, strict=True):
            # R4: the adapter is the ONLY path from native output to a signal, and the
            # raw dict travels with the vote.
            votes.append(HarmVote(
                annotator_id=self.annotator_id,
                comment_id=comment.comment_id,
                signal=adapt_a1(raw),
                raw=raw,
                confidence=confidence,
            ))
        return votes

    # -- native prediction -----------------------------------------------------

    def predict_raw(self, texts: Sequence[str]) -> list[tuple[dict[str, int], float | None]]:
        """Native Ibrohim & Budi binary columns, one dict per text. No PoC labels here."""
        if self.backend == "stub":
            return [self._stub_one(t) for t in texts]
        return self._transformers_predict(texts)

    def _stub_one(self, text: str) -> tuple[dict[str, int], float | None]:
        """Deterministic keyword stand-in. NOT a classifier (see the banner)."""
        low = f" {text.lower()} "
        hit = any(seed in low for seed in _STUB_HARM_SEEDS)
        raw = dict.fromkeys(IBROHIM_BUDI_COLUMNS, 0)
        if hit:
            raw["HS"] = 1
            raw["Abusive"] = 1
            if any(m in low for m in _STUB_GROUP_MARKERS):
                raw["HS_Group"] = 1
            else:
                raw["HS_Individual"] = 1
            raw["HS_Moderate"] = 1
        # No confidence: a keyword rule has no calibrated probability and inventing one
        # would be exactly the kind of number R5 forbids.
        return raw, None

    def _transformers_predict(self, texts: Sequence[str]) -> list[tuple[dict[str, int], float | None]]:
        import torch                                    # lazy: pytest runs without it

        model, tokenizer = self._load()
        device = next(model.parameters()).device
        out: list[tuple[dict[str, int], float | None]] = []
        harm_idx = [i for i, c in enumerate(self._columns) if c in A1_HARM_KEYS]
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                # tweet_preprocess() is applied HERE, right before tokenizing, and
                # nowhere else - it never touches the stored pipeline text. The
                # fine-tuning script applies the identical function to its training
                # texts, so this model always sees the input format it was trained on.
                batch = [tweet_preprocess(t) for t in texts[start:start + self.batch_size]]
                enc = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(device)
                logits = model(**enc).logits.float()
                probs = torch.sigmoid(logits).cpu()
                for row in probs:
                    raw = {
                        col: int(row[i].item() >= self.threshold)
                        for i, col in enumerate(self._columns)
                    }
                    # Confidence = P(any harm evidence), reported as the strongest harm
                    # column. It is the model's number, not a calibrated accuracy claim.
                    conf = max((row[i].item() for i in harm_idx), default=None)
                    out.append((raw, None if conf is None else float(conf)))
        return out

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._tokenizer
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        kwargs = {"revision": self.revision} if self.revision else {}
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, **kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint, problem_type="multi_label_classification", **kwargs,
        )
        self._columns = self._resolve_columns(model.config)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda"):
            model = model.half()
        model.to(device).eval()
        self._resolved_revision = (
            getattr(model.config, "_commit_hash", None) or self.revision
        )
        self._model, self._tokenizer = model, tokenizer
        return model, tokenizer

    def _resolve_columns(self, config: Any) -> tuple[str, ...]:
        """Trust the checkpoint's own `id2label` when it names the dataset columns.

        A checkpoint whose head is not the 12-column Ibrohim & Budi head is refused:
        silently reinterpreting some other head's outputs as `HS`/`Abusive` would write
        fabricated harm evidence into the record.
        """
        id2label = dict(getattr(config, "id2label", {}) or {})
        names = [id2label.get(i) for i in range(len(id2label))]
        if names and all(n in IBROHIM_BUDI_COLUMNS for n in names):
            return tuple(names)                                   # type: ignore[arg-type]
        n = int(getattr(config, "num_labels", 0) or 0)
        if n == len(IBROHIM_BUDI_COLUMNS):
            return IBROHIM_BUDI_COLUMNS
        raise ValueError(
            f"checkpoint {self.checkpoint!r} exposes {n} outputs and an unusable "
            f"id2label; A1 requires the {len(IBROHIM_BUDI_COLUMNS)}-column Ibrohim & "
            "Budi multilabel head."
        )

    # -- identity for models.lock.json (R3/D31) --------------------------------

    def model_identity(self) -> dict[str, Any]:
        if self.backend == "stub":
            return {
                "model_id": STUB_MODEL_ID,
                "revision": None,
                "quantization": None,
                "backend": "stub",
                "NOT_A_REAL_MODEL": True,
                "warning": "keyword stand-in; A1 signals in this run are not predictions",
            }
        identity: dict[str, Any] = {
            "model_id": self.checkpoint,
            "revision": self._resolved_revision,
            "quantization": None,
            "backend": self.backend,
            "base_model": self.base_model,
            "train_set": "ibrohim_budi_multilabel",
            "threshold": self.threshold,
        }
        local = Path(self.checkpoint) if self.checkpoint else None
        if self.hash_weights and local is not None and local.exists():
            identity["checkpoint_sha256"] = _hash_checkpoint_dir(local)
        return identity


def _text_of(comment: Comment) -> str:
    """Annotate normalized text; fall back to the original if normalization is absent.

    R7: `text_original` is never overwritten, only read.
    """
    return comment.text_normalized or comment.text_original


def harm_counts(votes: Iterable[HarmVote]) -> dict[str, int]:
    counts = {s.value: 0 for s in HarmSignal}
    for v in votes:
        counts[v.signal.value] += 1
    return counts
