"""A3 - the local Indonesian LLM. The ONLY annotator that can vote `legitimate_criticism`.

**A3 is the only annotator this lane lets import `Label`.** TRAP 1 (AGENTS.md §3): A1 and
A2 structurally cannot see criticism, so on the criticism-vs-harassment boundary the
ensemble is n=1, and A3 is that one. If demo scene 2 looks wrong, fix `prompts/a3_annotator.md`
(read VERBATIM below, never paraphrased), not this module and not the ensemble.

R4: this class produces exactly one thing on its own - the raw generated string. The
mapping to a `Label` happens exclusively in `adapters.adapt_a3`, which raises on an
off-schema label. That raise is deliberate: coercing an off-schema string to a default
label would write an invented label into the record, which is worse than dropping the
vote. So `annotate()` catches the raise, counts it, keeps up to
`max_off_schema_examples` samples for the operator to read, and does NOT emit a
`LabelVote` for that comment. `src/consensus.py`'s CLI already has the matching half of
this contract: "a comment missing its A3 vote is skipped with a counted warning."

Backends
--------
`llamacpp`      GGUF file via llama-cpp-python. Simplest on Windows - no CUDA toolchain
                needed, a prebuilt wheel with cuBLAS support is enough.
`transformers`  a local 4-bit AWQ checkpoint (lazy import).
`stub`          a deterministic keyword labeller so the pipeline runs with no model on
                disk. It prints a loud banner: its output is NOT model output. It is
                intentionally naive - it exists to exercise the pipeline shape, not to
                stand in for a labelling judgement.

8-9B models only (AGENTS.md): 27B at 4-bit is ~13.5 GB and too tight with KV cache on a
16 GB 4080.
"""
from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..schemas import Comment, Label, LabelVote
from .adapters import adapt_a3

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = Path("prompts/a3_annotator.md")

STUB_MODEL_ID = "stub://a3-deterministic-keyword-labeller"

_BANNER = """
################################################################################
##                                                                            ##
##   A3 IS RUNNING ON THE  * S T U B *  BACKEND                               ##
##   THESE ARE NOT LLM PREDICTIONS. NOTHING HERE IS A REAL LABEL.             ##
##                                                                            ##
##   A hand-written keyword rule stands in for the local LLM so the pipeline  ##
##   can run end to end with no model downloaded. It cannot see nuance and    ##
##   its legitimate_criticism / harassment_personal split is a guess, not a   ##
##   judgement. Do not quote, screenshot or report any number derived from    ##
##   this run.                                                                ##
##                                                                            ##
##   Real run:  --backend llamacpp  with annotators.a3_llm.model pointing at  ##
##   a GGUF file (8-9B, e.g. a quantized Indonesian-capable Llama/Qwen/SEA-   ##
##   LION variant), or  --backend transformers  with a local 4-bit AWQ        ##
##   checkpoint. NOT 27B - too tight with KV cache on a 16 GB 4080.           ##
##                                                                            ##
################################################################################
"""

MAX_NEW_TOKENS_CAP = 24  # AGENTS.md / build contract: A3 generates <=24 new tokens.

#: Cached in place of a real generation when the model could not even be run for a
#: comment (context-window overflow, most often). Guaranteed to fail `adapt_a3` like
#: any other off-schema string, so it flows through the existing drop-and-count path
#: with no special-casing downstream - and because it IS cached, a resumed run replays
#: the failure instantly instead of re-attempting the same doomed prompt.
_GENERATION_ERROR_SENTINEL = "__GENERATION_ERROR__"


def _print_stub_banner() -> None:
    """Loud on stdout *and* stderr, so it survives a redirect and lands in screenshots."""
    print(_BANNER, flush=True)
    print(_BANNER, file=sys.stderr, flush=True)


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    if p.exists():
        return p
    candidate = _REPO_ROOT / p
    return candidate if candidate.exists() else p


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


_cuda_dll_dirs_registered = False


def _register_cuda_dll_dirs() -> None:
    """Windows-only fixup so the `llamacpp` backend's CUDA build actually finds CUDA.

    The prebuilt CUDA wheel for llama-cpp-python (no local CUDA Toolkit needed to
    install it - see docs/06-a1-indobertweet.md sibling note in the A3 doc) still
    dynamically links `ggml-cuda.dll` against the CUDA Runtime/cuBLAS shared
    libraries at load time. Without the CUDA Toolkit installed, those are missing
    from every normal search path, and `from llama_cpp import Llama` fails with an
    opaque "could not find module (or one of its dependencies)" - it never says
    which dependency, because Windows doesn't tell you.

    `torch`'s CUDA build already ships its own copies of exactly those libraries
    (it needs them for the same reason), and this project already depends on torch
    for A1. So this points Windows' DLL search at torch's `lib/` directory before
    `llama_cpp` loads, rather than requiring a second multi-gigabyte CUDA Toolkit
    install just to satisfy one more package's copy of the same runtime.

    A no-op on any platform without this problem: non-Windows resolves shared
    library dependencies differently, and if torch is not installed this simply
    does nothing (llama_cpp then either finds CUDA itself or raises its own error).
    """
    global _cuda_dll_dirs_registered
    if _cuda_dll_dirs_registered or sys.platform != "win32":
        return
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
    except Exception:
        pass  # best-effort: let llama_cpp's own import raise if this didn't help
    _cuda_dll_dirs_registered = True


def _hash_model_path(path: Path) -> str | None:
    """Identity of a local model file/dir: sha over bytes (file) or per-file (dir)."""
    if not path.exists():
        return None
    if path.is_file():
        return _sha256_path(path)
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(f.relative_to(path).as_posix().encode("utf-8"))
        h.update(_sha256_path(f).encode("ascii"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------------------
# stub labeller - deterministic keyword rule, NOT a model. See the banner.
# --------------------------------------------------------------------------------------

_THREAT_MARKERS: tuple[str, ...] = (
    "alamat lu", "alamat rumah", "gue sebar", "nomor lu", "gue datengin",
    "siap siap gue", "gue samperin", "kantor lu", "data lu udah", "foto keluarga lu",
    "gue laporin ke kantor",
)
_SEXUAL_MARKERS: tuple[str, ...] = (
    "modal body", "buka baju", "jual paha", "jualan sensasi", "baju ketat", "bispak",
)
_DEATH_MARKERS: tuple[str, ...] = (
    "mending mati", "mati aja", "mendingan mati", "bunuh diri", "gantung diri",
)
_OTHER_INSULTS: tuple[str, ...] = (
    "goblok", "tolol", "bego", "bodoh", "idiot", "dungu", "anjing", "bangsat",
    "kampret", "bajingan", "kontol", "memek", "ngentot", "sampah", "taik", "bacot",
    "otak udang", "sarap",
)
_WORK_MARKERS: tuple[str, ...] = (
    "video", "konten", "editing", "editan", "audio", "produk", "kualitas", "warna",
    "desain", "kamera", "packaging", "sinyal", "naskah", "layout", "jahitan", "cetak",
    "baterai", "battery", "mixing", "efek suara", "thumbnail", "streaming", "siaran",
)
_PERSON_MARKERS: tuple[str, ...] = ("lu", "lo", "elu", "kamu", "kau", "situ", "muka", "orangnya")


def _stub_label(text: str) -> str:
    """Deterministic keyword rule standing in for A3's judgement. Pure function of
    `text`: same input always produces the same output. NOT a classifier - see `_BANNER`.
    """
    low = f" {(text or '').lower()} "

    if any(m in low for m in _THREAT_MARKERS):
        return Label.THREAT_PRIVACY.value
    if any(m in low for m in _SEXUAL_MARKERS):
        return Label.SEXUAL_HARM.value
    if any(m in low for m in _DEATH_MARKERS):
        return Label.HARASSMENT_PERSONAL.value

    if "jelek" in low:
        # The scene-2 boundary: criticism of a WORK/PRODUCT vs. of a PERSON.
        if any(m in low for m in _WORK_MARKERS):
            return Label.LEGITIMATE_CRITICISM.value
        if any(m in low for m in _PERSON_MARKERS) or "muka" in low:
            return Label.HARASSMENT_PERSONAL.value
        return Label.LEGITIMATE_CRITICISM.value

    if any(m in low for m in _OTHER_INSULTS):
        return Label.HARASSMENT_PERSONAL.value

    return Label.BENIGN_OTHER.value


# --------------------------------------------------------------------------------------
# the annotator
# --------------------------------------------------------------------------------------


class A3LocalLLM:
    """Local Indonesian LLM. Implements `base.LabelAnnotator`.

    The only annotator that can emit `Label.LEGITIMATE_CRITICISM` (TRAP 1). Every raw
    generation goes through `adapters.adapt_a3`, which validates against the frozen
    5-label schema and raises on anything else; `annotate()` counts those raises rather
    than inventing a label for them (see the module docstring).
    """

    can_vote_criticism = True

    def __init__(
        self,
        model: str | Path | None = None,
        backend: str = "stub",
        *,
        annotator_id: str = "a3",
        max_new_tokens: int = 24,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        quantization: str | None = None,
        revision: str | None = None,
        device: str | None = None,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        hash_model: bool = True,
        max_off_schema_examples: int = 20,
        generate_fn: Callable[[str], str] | None = None,
    ) -> None:
        if backend not in ("llamacpp", "transformers", "stub"):
            raise ValueError(f"A3 backend must be 'llamacpp', 'transformers' or 'stub', got {backend!r}")
        self.annotator_id = annotator_id
        self.model = str(model) if model is not None else None
        self.backend = backend
        if int(max_new_tokens) > MAX_NEW_TOKENS_CAP:
            print(
                f"[a3] max_new_tokens={max_new_tokens} exceeds the {MAX_NEW_TOKENS_CAP}-token "
                f"contract cap; clamping to {MAX_NEW_TOKENS_CAP}.",
                file=sys.stderr,
            )
        self.max_new_tokens = min(int(max_new_tokens), MAX_NEW_TOKENS_CAP)
        self.quantization = quantization
        self.revision = revision
        self.device = device
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.n_threads = n_threads
        self.hash_model = hash_model
        self._max_off_schema_examples = int(max_off_schema_examples)
        self._generate_fn = generate_fn

        self.prompt_path = Path(prompt_path)
        self.prompt_text = self._load_prompt()

        self._model: Any = None
        self._tokenizer: Any = None

        # -- per-`annotate()`-call bookkeeping, reset at the top of annotate() ---------
        self.n_total = 0
        self.n_off_schema = 0
        self.off_schema_examples: list[tuple[str, str]] = []

        if self.backend == "stub":
            _print_stub_banner()
        elif not self.model and self._generate_fn is None:
            raise ValueError(
                f"A3 backend={self.backend!r} needs annotators.a3_llm.model pointing at "
                + (
                    "a GGUF checkpoint file (8-9B)."
                    if self.backend == "llamacpp"
                    else "a local 4-bit AWQ checkpoint directory (8-9B)."
                )
                + " Not 27B - too tight with KV cache on a 16 GB 4080. Use --backend stub "
                "to run the pipeline with no model on disk."
            )

    # -- prompt (FROZEN, R3) ----------------------------------------------------

    def _load_prompt(self) -> str:
        p = _resolve(self.prompt_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"A3's prompt not found at {p}. prompts/a3_annotator.md is frozen (R3) "
                "and must exist verbatim before A3 can run."
            )
        return p.read_text(encoding="utf-8")

    def build_prompt(self, text: str) -> str:
        """The frozen prompt, verbatim, with the comment to label appended.

        `prompts/a3_annotator.md` is instructions only - it has no placeholder for the
        comment itself, so this is the one place that concatenation happens. The prompt
        text above the delimiter is never rewritten.
        """
        return f'{self.prompt_text}\n\nComment: "{text}"\nLabel:'

    # -- the annotator protocol ------------------------------------------------

    def annotate(
        self,
        comments: Sequence[Comment],
        *,
        cache: Mapping[str, str] | None = None,
        on_generate: Callable[[str, str], None] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[LabelVote]:
        """Label every comment. Optionally resumable across process restarts.

        A full development run is ~16k sequential generations - over an hour of GPU
        time in one process, with nothing on disk until the very end. `cache` and
        `on_generate` let a caller checkpoint that: `on_generate(comment_id, raw)` is
        invoked for each *newly generated* string, and on a later run those strings come
        back in `cache` and are replayed instead of regenerated.

        What is cached is the RAW generation, never a `Label`. Replaying it still sends
        it through `adapt_a3` below, so R4 holds (the adapter remains the only path from
        a string to a Label) and a resumed run reports exactly the off-schema count a
        single-shot run would have. Greedy decoding makes the replay faithful, but only
        for the same model and prompt - guarding the cache against a changed
        `model_identity()` is the caller's job, and `run.py` does it.
        """
        self.n_total = 0
        self.n_off_schema = 0
        self.off_schema_examples = []

        votes: list[LabelVote] = []
        total = len(comments)
        for comment in comments:
            text = comment.text_normalized or comment.text_original
            if cache is not None and comment.comment_id in cache:
                raw = cache[comment.comment_id]
            else:
                try:
                    raw = self._generate_one(text)
                except Exception as exc:
                    # A per-comment generation failure (most often: one unusually long
                    # real comment overflows the context window) must never take down
                    # a run that may be hours into thousands of other comments -
                    # confirmed the hard way on a real 122k-comment corpus (2026-09-08).
                    # Treated exactly like an off-schema output: no vote is invented,
                    # it is counted and logged, and the SENTINEL is cached via
                    # `on_generate` so a resumed run does not retry the same doomed
                    # prompt against the model every time.
                    print(
                        f"[a3] generation failed for {comment.comment_id!r} ({exc}) - "
                        "dropping this vote, not the run.",
                        file=sys.stderr,
                    )
                    raw = _GENERATION_ERROR_SENTINEL
                if on_generate is not None:
                    on_generate(comment.comment_id, raw)
            self.n_total += 1
            if progress is not None:
                progress(self.n_total, total)
            try:
                # R4: the adapter is the ONLY path from a raw string to a Label. It
                # raises on anything off-schema.
                label = adapt_a3(raw)
            except ValueError:
                self.n_off_schema += 1
                if len(self.off_schema_examples) < self._max_off_schema_examples:
                    self.off_schema_examples.append((comment.comment_id, raw))
                # Never coerce to a default label - that would write an invented label
                # into the record. Drop the vote; consensus.py's CLI counts and warns
                # about a comment with a missing A3 vote.
                continue
            votes.append(LabelVote(
                annotator_id=self.annotator_id,
                comment_id=comment.comment_id,
                label=label,
                raw=raw,
                confidence=None,  # greedy decoding has no calibrated probability (R5)
            ))
        return votes

    # -- generation --------------------------------------------------------------

    def _generate_one(self, text: str) -> str:
        if self._generate_fn is not None:
            return self._generate_fn(text)
        if self.backend == "stub":
            return _stub_label(text)
        prompt = self.build_prompt(text)
        if self.backend == "llamacpp":
            return self._llamacpp_generate(prompt)
        return self._transformers_generate(prompt)

    def _load_llamacpp(self) -> Any:
        """`n_gpu_layers` defaults to -1 (offload everything - trivial for an 8-9B GGUF
        on 16 GB) but is a constructor argument, not hardcoded, because it is the
        documented workaround for a real hardware issue: some prebuilt CUDA wheels for
        llama-cpp-python crash with an illegal-instruction fault (Windows error
        0xc000001d) on 12th/13th-gen Intel CPUs with hybrid P/E cores, REGARDLESS of
        `n_gpu_layers` - confirmed on this project's own dev machine (i9-13900K) on
        `n_gpu_layers=-1` and `=0` alike, so it is llama-cpp-python's ggml-cpu dispatch
        picking an AVX-512 code path the E-cores cannot execute, not a CUDA problem.
        `models.lock.json`'s `n_gpu_layers` field (see `model_identity`) records which
        path a run actually used. See docs/07-a3-local-llm.md for the full story,
        the diagnosis steps, and the GPU-throughput follow-up.
        """
        if self._model is not None:
            return self._model
        _register_cuda_dll_dirs()
        from llama_cpp import Llama  # lazy: pytest runs without llama-cpp-python

        kwargs: dict[str, Any] = dict(
            model_path=self.model,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads
        self._model = Llama(**kwargs)
        return self._model

    def _llamacpp_generate(self, prompt: str) -> str:
        """`prompt` is `build_prompt()`'s output, byte-for-byte - the frozen prompt
        text plus the comment (R3). What changes here is only HOW that string reaches
        the model, not its content.

        Sent as the single user turn of a chat completion, not raw text completion.
        `Llama(chat_format=None)` (the default - see `_load_llamacpp`) makes
        llama-cpp-python read the chat template baked into the GGUF's own metadata and
        wrap the turn in it. For an instruction-tuned model that is the difference
        between "the model reliably emits one of five words" and "the model completes
        the text however a base model would" - an un-templated instruct model tends to
        ramble past the label instead of stopping, which would show up as elevated
        off-schema drops, not a crash, so it is worth getting right rather than
        debugging after the fact from `off_schema_summary()` output.

        Falls back to raw completion if the checkpoint has no usable chat template
        (some GGUF conversions omit it) - `create_chat_completion` raises rather than
        silently misformatting the prompt when that happens.
        """
        llm = self._load_llamacpp()
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=0.0,  # deterministic-as-possible; this is a labelling task
                stop=["\n", "Comment:"],
            )
            return out["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            print(
                f"[a3] chat template unavailable ({exc}); falling back to raw "
                "completion for this checkpoint.",
                file=sys.stderr,
            )
            out = llm(
                prompt,
                max_tokens=self.max_new_tokens,
                temperature=0.0,
                stop=["\n", "Comment:"],
                echo=False,
            )
            return out["choices"][0]["text"]

    def _load_transformers(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._tokenizer
        import torch  # lazy: pytest runs without torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"revision": self.revision} if self.revision else {}
        tokenizer = AutoTokenizer.from_pretrained(self.model, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            self.model, device_map=self.device or "auto", **kwargs,
        )
        model.eval()
        self._model, self._tokenizer = model, tokenizer
        return model, tokenizer

    def _transformers_generate(self, prompt: str) -> str:
        import torch  # lazy: pytest runs without torch

        model, tokenizer = self._load_transformers()
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    # -- identity for models.lock.json (R3/D31) --------------------------------

    def model_identity(self) -> dict[str, Any]:
        prompt_sha256 = _sha256_bytes(self.prompt_text.encode("utf-8"))
        if self.backend == "stub":
            return {
                "model_id": STUB_MODEL_ID,
                "revision": None,
                "quantization": None,
                "backend": "stub",
                "prompt_sha256": prompt_sha256,
                "NOT_A_REAL_MODEL": True,
                "warning": "keyword stand-in; A3 labels in this run are not predictions",
            }
        identity: dict[str, Any] = {
            "model_id": self.model,
            "revision": self.revision,
            "quantization": self.quantization,
            "backend": self.backend,
            "max_new_tokens": self.max_new_tokens,
            "prompt_sha256": prompt_sha256,
        }
        if self.backend == "llamacpp":
            # Not cosmetic: n_gpu_layers=0 on this backend usually means the CPU
            # dispatch workaround documented on `_load_llamacpp`, not a deliberate
            # performance choice, and it changes throughput by roughly an order of
            # magnitude. Worth knowing which one produced a given run's numbers.
            identity["n_gpu_layers"] = self.n_gpu_layers
            # Recorded 2026-09-08 after a real crash: a comment whose prompt overflowed
            # n_ctx took down a 122k-comment run. n_ctx doesn't change output for any
            # prompt that already fits within it (greedy decoding, temperature=0), but
            # it determines whether an unusually long comment succeeds or gets dropped
            # as a generation error - worth knowing which window produced a given run.
            identity["n_ctx"] = self.n_ctx
        else:
            identity["device"] = self.device
        local = Path(self.model) if self.model else None
        if self.hash_model and local is not None and local.exists():
            identity["model_sha256"] = _hash_model_path(local)
        return identity


def off_schema_summary(annotator: "A3LocalLLM") -> dict[str, Any]:
    """Small, printable summary of what `annotate()` just counted. Never silently
    dropped: `run.py` prints this at the end of every run that includes A3."""
    return {
        "n_total": annotator.n_total,
        "n_votes": annotator.n_total - annotator.n_off_schema,
        "n_off_schema": annotator.n_off_schema,
        "off_schema_examples": list(annotator.off_schema_examples),
    }
