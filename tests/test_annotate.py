"""Tests for the three annotators and their CLI (`src/annotate/`).

AGENTS.md TRAP 1 / D29 is the organizing fact of this whole lane: A1 and A2 structurally
cannot vote `legitimate_criticism`, because Ibrohim & Budi's 12 labels have no class for
non-hateful negative content. Every test here either guards that boundary directly or
guards something R4 (raw retained), R5 (no fabricated confidence), or R3 (frozen prompt,
model identity) depends on.

No torch, transformers or llama_cpp import anywhere in this file - the stub backend is
the only one exercised, and the real backends are only reachable through lazy imports
inside a1_indobertweet.py / a2_rules.py / a3_llm.py.
"""
from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

from src.annotate import a1_indobertweet as a1_mod
from src.annotate import a2_rules as a2_mod
from src.annotate import a3_llm as a3_mod
from src.annotate import run as run_mod
from src.annotate.a1_indobertweet import A1_HARM_KEYS as A1_KEYS_FROM_A1  # re-export check
from src.annotate.a1_indobertweet import A1IndoBERTweet, IBROHIM_BUDI_COLUMNS
from src.annotate.a2_rules import A2Rules, FAMILY_PRIORITY, RuleLoadError
from src.annotate.a3_llm import A3LocalLLM, _stub_label, off_schema_summary
from src.annotate.adapters import A1_HARM_KEYS, A2_HARM_FAMILIES, adapt_a1, adapt_a2, adapt_a3
from src.schemas import Comment, HarmSignal, HarmVote, Label


def _comment(cid: str, text: str) -> Comment:
    return Comment(
        comment_id=cid, channel_id="c", video_id="v", author_hash="h",
        published_at="2026-01-01T00:00:00Z", text_original=text, text_normalized=text,
    )


# ---------------------------------------------------------------------------- fixture lexicon
# A2's real resources (data/raw/lexicons/rule_families.toml, abusive_id.csv) are owned by
# another agent and may not exist yet. This lane ships its own small fixture lexicon so
# these tests pass independently of that lane's progress.

RULE_FAMILIES_TOML = """
[insult]
patterns = ["\\\\bgoblok\\\\b"]

[slur]
patterns = ["\\\\bkafirlu\\\\b"]

[sexual]
patterns = ["\\\\bmesumbanget\\\\b"]

[threat]
patterns = ["\\\\bkubunuhlu\\\\b"]

[doxx]
patterns = ["\\\\balamatlu\\\\b"]

[death_wish]
patterns = ["\\\\bmatisajalu\\\\b"]

[selfharm_encouragement]
patterns = ["\\\\bgantungdirisana\\\\b"]
"""

ABUSIVE_CSV = "word\ntololsekali\n"

#: one clean trigger string per family in A2_HARM_FAMILIES.
_FAMILY_TRIGGER = {
    "insult": "goblok",
    "slur": "kafirlu",
    "sexual": "mesumbanget",
    "threat": "kubunuhlu",
    "doxx": "alamatlu",
    "death_wish": "matisajalu",
    "selfharm_encouragement": "gantungdirisana",
}


@pytest.fixture
def lexicon_dir(tmp_path):
    d = tmp_path / "lexicons"
    d.mkdir()
    (d / "rule_families.toml").write_text(RULE_FAMILIES_TOML, encoding="utf-8")
    (d / "abusive_id.csv").write_text(ABUSIVE_CSV, encoding="utf-8")
    return d


@pytest.fixture
def a2(lexicon_dir):
    return A2Rules(lexicon_dir=lexicon_dir)


# ============================================================================
# TRAP 1 / D29 - the load-bearing guard for this whole lane
# ============================================================================

def test_a1_and_a2_can_never_emit_a_label_guards_d29_trap1(a2):
    """A1 and A2 emit HarmSignal, never Label. Re-mapping their silence into a positive
    vote for benign_other is exactly the D29 bug (AGENTS.md §3, TRAP 1)."""
    a1 = A1IndoBERTweet(backend="stub")
    comments = [
        _comment("c1", "goblok banget lu, gak bisa mikir ya"),
        _comment("c2", "makasih videonya kak, keren banget"),
    ]
    a1_votes = a1.annotate(comments)
    a2_votes = a2.annotate(comments)
    for v in (*a1_votes, *a2_votes):
        assert type(v.signal) is HarmSignal
        assert not isinstance(v.signal, Label)

    # The modules' own docstrings claim Label is never even imported. Pin that claim.
    assert "Label" not in vars(a1_mod)
    assert "Label" not in vars(a2_mod)


def test_no_harm_evidence_is_never_a_positive_vote_for_benign_other():
    """The exact mechanism of D29: 'no harm found' must stay an abstention, not turn
    into benign_other."""
    assert adapt_a1({"HS": 0, "Abusive": 0}) is HarmSignal.NO_HARM_EVIDENCE
    assert adapt_a2({"family": None}) is HarmSignal.NO_HARM_EVIDENCE


# ============================================================================
# R4 - every vote passes through its adapter, raw retained
# ============================================================================

def test_a1_vote_passes_through_adapt_a1_with_raw_retained():
    a1 = A1IndoBERTweet(backend="stub")
    v = a1.annotate([_comment("c1", "goblok banget lu")])[0]
    assert v.signal == adapt_a1(v.raw)
    assert isinstance(v.raw, dict)
    assert set(IBROHIM_BUDI_COLUMNS) <= set(v.raw)


def test_a2_vote_passes_through_adapt_a2_with_raw_retained(a2):
    v = a2.annotate([_comment("c1", "goblok lu")])[0]
    assert v.signal == adapt_a2(v.raw)
    assert isinstance(v.raw, dict) and "family" in v.raw


def test_a3_vote_passes_through_adapt_a3_with_raw_retained():
    a3 = A3LocalLLM(backend="stub", generate_fn=lambda text: "benign_other")
    v = a3.annotate([_comment("c1", "halo")])[0]
    assert v.label == adapt_a3(v.raw)
    assert v.raw == "benign_other"


def test_a2_raw_retains_every_family_and_term_that_fired(a2):
    """One matched term per family that fires, across ALL families that fired -
    not just the first one reported as `family`."""
    v = a2.annotate([_comment("c1", "goblok banget, kubunuhlu sekalian")])[0]
    assert vote_family_consistent(v)
    assert {"insult", "threat"} <= set(v.raw["families"])
    assert "goblok" in v.raw["matched"]
    assert "kubunuhlu" in v.raw["matched"]


def vote_family_consistent(v: HarmVote) -> bool:
    return v.raw["family"] in v.raw["families"] or v.raw["family"] is None


# ============================================================================
# A2: broad family coverage + clean-text abstention
# ============================================================================

@pytest.mark.parametrize("family", sorted(A2_HARM_FAMILIES))
def test_a2_fires_on_each_harm_family(a2, family):
    text = _FAMILY_TRIGGER[family]
    vote = a2.annotate([_comment("c", text)])[0]
    assert vote.signal is HarmSignal.HARM
    assert vote.raw["family"] == family


def test_a2_returns_no_harm_evidence_on_clean_text(a2):
    vote = a2.annotate([_comment("c", "makasih videonya kak, keren banget")])[0]
    assert vote.signal is HarmSignal.NO_HARM_EVIDENCE
    assert vote.raw["family"] is None


def test_family_priority_matches_a2_harm_families_both_directions():
    """A2's own docstring claims this: every FAMILY_PRIORITY entry is in
    A2_HARM_FAMILIES and vice versa, so a new family cannot be silently dropped."""
    assert set(FAMILY_PRIORITY) == set(A2_HARM_FAMILIES)


def test_a2_refuses_to_run_with_zero_patterns(tmp_path):
    empty = tmp_path / "empty_lexicons"
    empty.mkdir()
    (empty / "rule_families.toml").write_text("", encoding="utf-8")
    with pytest.raises(RuleLoadError):
        A2Rules(lexicon_dir=empty)


def test_a1_harm_keys_are_the_non_strength_ibrohim_budi_columns():
    non_strength = set(IBROHIM_BUDI_COLUMNS) - {"HS_Weak", "HS_Moderate", "HS_Strong"}
    assert set(A1_HARM_KEYS) == non_strength
    assert set(A1_KEYS_FROM_A1) == non_strength


# ============================================================================
# A3: off-schema counting, never coerced
# ============================================================================

def test_a3_off_schema_output_raises_rather_than_being_coerced():
    with pytest.raises(ValueError):
        adapt_a3("toxic")
    with pytest.raises(ValueError):
        adapt_a3("")


def test_a3_annotate_counts_off_schema_and_drops_the_vote_without_inventing_a_label():
    calls = iter(["legitimate_criticism", "toxic", "benign_other"])
    a3 = A3LocalLLM(backend="stub", generate_fn=lambda text: next(calls))
    comments = [_comment("c1", "x"), _comment("c2", "y"), _comment("c3", "z")]

    votes = a3.annotate(comments)

    assert [v.comment_id for v in votes] == ["c1", "c3"], "the off-schema comment must be dropped, not coerced"
    assert a3.n_total == 3
    assert a3.n_off_schema == 1
    assert a3.off_schema_examples == [("c2", "toxic")]

    summary = off_schema_summary(a3)
    assert summary == {
        "n_total": 3, "n_votes": 2, "n_off_schema": 1,
        "off_schema_examples": [("c2", "toxic")],
    }


def test_a3_off_schema_counters_reset_between_annotate_calls():
    a3 = A3LocalLLM(backend="stub", generate_fn=lambda text: "not_a_real_label")
    a3.annotate([_comment("c1", "x")])
    assert a3.n_off_schema == 1
    a3.annotate([_comment("c2", "y"), _comment("c3", "z")])
    assert a3.n_total == 2 and a3.n_off_schema == 2, "counts must be per-call, not cumulative"


# ============================================================================
# A3 resume checkpoint (run.py): a crashed 16k-comment run must not start over,
# and must never replay generations from a different model, prompt or corpus.
# ============================================================================

def _write_cache(path, guard, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"guard": guard}, sort_keys=True, ensure_ascii=False)]
    lines += [json.dumps({"comment_id": c, "raw": r}) for c, r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a3_replays_cached_generations_instead_of_regenerating():
    generated = []

    def fake_generate(text):
        generated.append(text)
        return "benign_other"

    a3 = A3LocalLLM(backend="stub", generate_fn=fake_generate)
    comments = [_comment("c1", "x"), _comment("c2", "y"), _comment("c3", "z")]

    votes = a3.annotate(comments, cache={"c1": "harassment_personal", "c2": "benign_other"})

    assert generated == ["z"], "only the uncached comment should hit the model"
    assert [v.comment_id for v in votes] == ["c1", "c2", "c3"]
    assert votes[0].label == Label.HARASSMENT_PERSONAL, "the cached generation must be used verbatim"


def test_a3_cached_generations_still_go_through_the_adapter():
    """R4: caching stores the RAW string, so a bad one is still counted, not trusted."""
    a3 = A3LocalLLM(backend="stub", generate_fn=lambda text: "benign_other")
    votes = a3.annotate([_comment("c1", "x")], cache={"c1": "toxic"})

    assert votes == [], "an off-schema cached generation must be dropped like a fresh one"
    assert a3.n_off_schema == 1, "a resumed run must report the same off-schema count"


def test_a3_on_generate_fires_only_for_newly_generated_comments():
    seen = []
    a3 = A3LocalLLM(backend="stub", generate_fn=lambda text: "benign_other")
    a3.annotate(
        [_comment("c1", "x"), _comment("c2", "y")],
        cache={"c1": "benign_other"},
        on_generate=lambda cid, raw: seen.append((cid, raw)),
    )
    assert seen == [("c2", "benign_other")], "replayed comments must not be re-appended"


def test_a3_progress_callback_reports_against_the_total():
    calls = []
    a3 = A3LocalLLM(backend="stub", generate_fn=lambda text: "benign_other")
    a3.annotate([_comment(f"c{i}", "x") for i in range(3)],
                progress=lambda done, total: calls.append((done, total)))
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_a3_cache_is_refused_when_the_model_identity_differs(tmp_path, capsys):
    path = tmp_path / "a3_generations.jsonl"
    good = run_mod.a3_cache_guard({"model_id": "real-model"}, "dev", "abc123")
    _write_cache(path, run_mod.a3_cache_guard({"model_id": "OTHER"}, "dev", "abc123"),
                 [("c1", "benign_other")])

    assert run_mod._load_a3_cache(path, good) == {}, "a different model must invalidate the cache"
    assert not path.exists(), "the stale checkpoint is moved aside"
    assert (tmp_path / "a3_generations.jsonl.stale").is_file(), "and kept, not deleted"
    assert "NOT reusing" in capsys.readouterr().err


def test_a3_cache_is_refused_when_the_corpus_differs(tmp_path):
    path = tmp_path / "a3_generations.jsonl"
    identity = {"model_id": "real-model"}
    _write_cache(path, run_mod.a3_cache_guard(identity, "dev", "OLD_CORPUS"),
                 [("c1", "benign_other")])

    fresh = run_mod.a3_cache_guard(identity, "dev", "NEW_CORPUS")
    assert run_mod._load_a3_cache(path, fresh) == {}, "a re-generated corpus must invalidate the cache"


def test_a3_cache_is_refused_across_splits(tmp_path):
    """The held-out run must never replay development generations (R3)."""
    path = tmp_path / "a3_generations.jsonl"
    identity = {"model_id": "real-model"}
    _write_cache(path, run_mod.a3_cache_guard(identity, "dev", "abc123"),
                 [("c1", "benign_other")])

    heldout = run_mod.a3_cache_guard(identity, "heldout", "abc123")
    assert run_mod._load_a3_cache(path, heldout) == {}


def test_a3_cache_is_reused_when_everything_matches(tmp_path):
    path = tmp_path / "a3_generations.jsonl"
    guard = run_mod.a3_cache_guard({"model_id": "real-model"}, "dev", "abc123")
    _write_cache(path, guard, [("c1", "benign_other"), ("c2", "harassment_personal")])

    assert run_mod._load_a3_cache(path, guard) == {
        "c1": "benign_other", "c2": "harassment_personal",
    }


def test_a3_cache_survives_a_truncated_final_line(tmp_path):
    """A crash mid-write leaves a partial line; everything before it is still good."""
    path = tmp_path / "a3_generations.jsonl"
    guard = run_mod.a3_cache_guard({"model_id": "real-model"}, "dev", "abc123")
    _write_cache(path, guard, [("c1", "benign_other")])
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"comment_id": "c2", "ra')

    assert run_mod._load_a3_cache(path, guard) == {"c1": "benign_other"}


def test_a3_cache_paths_are_namespaced_by_split(tmp_path):
    dev = run_mod.a3_cache_path("dev", tmp_path)
    heldout = run_mod.a3_cache_path("heldout", tmp_path)
    assert dev != heldout
    assert heldout.parent.name == "heldout"


def test_a3_generation_failure_drops_the_vote_without_crashing_the_run():
    """Regression for a real crash (2026-09-08): one unusually long comment overflowed
    the model's context window 24,500 comments into a 122k-comment run and took the
    whole process down. A per-comment generation failure must be dropped like any
    off-schema output, never propagate."""
    def flaky(text):
        if text == "boom":
            raise RuntimeError("Requested tokens (3263) exceed context window of 2048")
        return "benign_other"

    a3 = A3LocalLLM(backend="stub", generate_fn=flaky)
    comments = [_comment("c1", "ok"), _comment("c2", "boom"), _comment("c3", "ok")]

    votes = a3.annotate(comments)

    assert [v.comment_id for v in votes] == ["c1", "c3"], "the failing comment's vote must be dropped, not the run"
    assert a3.n_total == 3
    assert a3.n_off_schema == 1, "a generation failure is counted the same way an off-schema output is"


def test_a3_generation_failure_is_cached_so_a_resume_does_not_retry_it(capsys):
    from src.annotate.a3_llm import _GENERATION_ERROR_SENTINEL

    def flaky(text):
        raise RuntimeError("context window overflow")

    a3 = A3LocalLLM(backend="stub", generate_fn=flaky)
    seen = []
    votes = a3.annotate([_comment("c1", "boom")], on_generate=lambda cid, raw: seen.append((cid, raw)))

    assert votes == []
    assert seen == [("c1", _GENERATION_ERROR_SENTINEL)], (
        "the sentinel must be cached so a resumed run replays the failure instead of "
        "hitting the model again with the same doomed prompt"
    )
    assert "dropping this vote, not the run" in capsys.readouterr().err


def test_a3_replayed_generation_error_sentinel_is_dropped_like_any_off_schema_string():
    from src.annotate.a3_llm import _GENERATION_ERROR_SENTINEL

    a3 = A3LocalLLM(backend="stub", generate_fn=lambda text: "benign_other")
    votes = a3.annotate([_comment("c1", "x")], cache={"c1": _GENERATION_ERROR_SENTINEL})

    assert votes == []
    assert a3.n_off_schema == 1


def test_resume_is_off_by_default_and_writes_no_sidecar(tmp_path):
    """A plain run must touch no checkpoint file at all."""
    import pandas as pd

    root = tmp_path
    (root / "data/interim").mkdir(parents=True)
    pd.DataFrame([
        {"comment_id": "c1", "text_normalized": "halo", "text_original": "halo",
         "split": "dev", "platform": "youtube"},
    ]).to_parquet(root / "data/interim/comments_normalized.parquet", index=False)

    run_mod.run(config={}, root=root, annotators=["a3"], backend="stub", split="dev")

    assert not run_mod.a3_cache_path("dev", root).exists()


# ============================================================================
# A3 stub: deterministic, and loudly announces itself
# ============================================================================

def test_a3_stub_label_is_a_pure_deterministic_function():
    text = "videonya jelek banget kualitasnya"
    assert _stub_label(text) == _stub_label(text) == Label.LEGITIMATE_CRITICISM.value


def test_a3_stub_backend_is_deterministic_end_to_end():
    a3 = A3LocalLLM(backend="stub")
    text = "goblok banget lu, gak bisa mikir ya"
    c = _comment("c1", text)
    v1 = a3.annotate([c])[0]
    v2 = a3.annotate([c])[0]
    assert v1.label == v2.label == Label.HARASSMENT_PERSONAL


def test_a3_stub_prints_a_loud_not_real_banner(capsys):
    A3LocalLLM(backend="stub")
    captured = capsys.readouterr()
    for stream in (captured.out, captured.err):
        assert "S T U B" in stream
        assert "NOT" in stream


def test_a1_stub_prints_a_loud_not_real_banner(capsys):
    A1IndoBERTweet(backend="stub")
    captured = capsys.readouterr()
    for stream in (captured.out, captured.err):
        assert "S T U B" in stream
        assert "NOT" in stream


def test_a3_stub_model_identity_is_marked_not_real():
    a3 = A3LocalLLM(backend="stub")
    identity = a3.model_identity()
    assert identity["NOT_A_REAL_MODEL"] is True
    assert identity["backend"] == "stub"


def test_a3_real_backend_requires_a_model_path():
    with pytest.raises(ValueError):
        A3LocalLLM(backend="llamacpp")
    with pytest.raises(ValueError):
        A3LocalLLM(backend="transformers")


def test_a3_generates_at_most_24_new_tokens_even_if_config_asks_for_more():
    a3 = A3LocalLLM(backend="stub", max_new_tokens=999)
    assert a3.max_new_tokens == 24


# ---------------------------------------------------------------------------------
# llamacpp backend: n_gpu_layers/n_threads pass-through, chat-template generation.
#
# docs/07-a3-local-llm.md - on some 12th/13th-gen Intel CPUs the prebuilt CUDA wheel
# for llama-cpp-python crashes with an illegal-instruction fault REGARDLESS of GPU
# offload level, so n_gpu_layers must be a real constructor argument (a documented
# hardware workaround, not just a performance knob) and it must actually reach
# `Llama(...)`. A fake `llama_cpp` module is injected via sys.modules so this stays
# independent of whether the real package is installed, matching every other test in
# this file.
# ---------------------------------------------------------------------------------

class _FakeLlama:
    """Stands in for `llama_cpp.Llama` - records constructor kwargs, and its
    `create_chat_completion` returns a scripted answer so `_llamacpp_generate` can be
    exercised without a real model on disk."""

    last_kwargs: dict | None = None
    chat_reply: str = "benign_other"
    raise_on_chat: bool = False

    def __init__(self, **kwargs):
        _FakeLlama.last_kwargs = kwargs

    def create_chat_completion(self, messages, **kwargs):
        if _FakeLlama.raise_on_chat:
            raise RuntimeError("no chat template on this checkpoint")
        return {"choices": [{"message": {"content": _FakeLlama.chat_reply}}]}

    def __call__(self, prompt, **kwargs):
        return {"choices": [{"text": _FakeLlama.chat_reply}]}


@pytest.fixture
def fake_llama_cpp(monkeypatch):
    """Injects a fake `llama_cpp` module and disables the CUDA-DLL-directory fixup
    (which is a real, Windows-only side effect this test must not trigger)."""
    import sys
    import types

    _FakeLlama.last_kwargs = None
    _FakeLlama.chat_reply = "benign_other"
    _FakeLlama.raise_on_chat = False

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    monkeypatch.setattr(a3_mod, "_register_cuda_dll_dirs", lambda: None)
    yield _FakeLlama


def test_n_gpu_layers_defaults_to_full_offload():
    """-1 = offload everything. The documented default for a machine without the
    dispatch bug; changing it is an explicit, per-machine override."""
    a3 = A3LocalLLM(backend="stub")
    assert a3.n_gpu_layers == -1


def test_n_gpu_layers_and_n_threads_reach_the_real_llama_constructor(fake_llama_cpp):
    a3 = A3LocalLLM(
        model="fake.gguf", backend="llamacpp", n_gpu_layers=0, n_threads=24, n_ctx=512,
    )
    a3._load_llamacpp()
    assert fake_llama_cpp.last_kwargs["n_gpu_layers"] == 0
    assert fake_llama_cpp.last_kwargs["n_threads"] == 24
    assert fake_llama_cpp.last_kwargs["n_ctx"] == 512


def test_n_threads_omitted_when_not_set_leaves_llama_cpp_pick_its_own_default(fake_llama_cpp):
    a3 = A3LocalLLM(model="fake.gguf", backend="llamacpp")
    a3._load_llamacpp()
    assert "n_threads" not in fake_llama_cpp.last_kwargs


def test_model_identity_records_n_gpu_layers_for_the_llamacpp_backend(fake_llama_cpp):
    """Not cosmetic (docs/07-a3-local-llm.md): n_gpu_layers=0 on this backend usually
    means the CPU-dispatch-bug workaround, not a deliberate choice, and it changes
    throughput by roughly an order of magnitude - worth knowing which produced a run."""
    a3 = A3LocalLLM(model="fake.gguf", backend="llamacpp", n_gpu_layers=0, hash_model=False)
    assert a3.model_identity()["n_gpu_layers"] == 0


def test_llamacpp_generation_uses_chat_completion_with_the_frozen_prompt_as_one_turn(
    fake_llama_cpp,
):
    """The frozen prompt (R3) must reach the model unchanged, just wrapped as a single
    user turn rather than sent as raw completion text - see a3_llm.py::_llamacpp_generate."""
    fake_llama_cpp.chat_reply = "harassment_personal"
    a3 = A3LocalLLM(model="fake.gguf", backend="llamacpp")
    result = a3._generate_one("dasar goblok")
    assert result == "harassment_personal"


def test_llamacpp_falls_back_to_raw_completion_if_no_chat_template(fake_llama_cpp):
    """Some GGUF conversions omit a chat template. `create_chat_completion` raising is
    the signal to fall back, not a fatal error."""
    fake_llama_cpp.raise_on_chat = True
    fake_llama_cpp.chat_reply = "benign_other"
    a3 = A3LocalLLM(model="fake.gguf", backend="llamacpp")
    result = a3._generate_one("halo semua")
    assert result == "benign_other"


def test_register_cuda_dll_dirs_is_a_safe_noop_off_windows(monkeypatch):
    """Must never raise, on any platform, with or without torch installed - it is a
    best-effort fixup, not something annotate() can afford to depend on succeeding."""
    monkeypatch.setattr(a3_mod.sys, "platform", "linux")
    a3_mod._cuda_dll_dirs_registered = False
    a3_mod._register_cuda_dll_dirs()  # must not raise


def test_build_a3_passes_n_gpu_layers_and_n_threads_from_config():
    """config/local.yaml documents n_gpu_layers=0 as a per-machine hardware workaround
    (docs/07-a3-local-llm.md), not a default - build_a3 must only override when the
    config actually sets it, so a machine without the dispatch bug keeps -1."""
    cfg = {"annotators": {"a3_llm": {
        "model": "fake.gguf", "n_gpu_layers": 0, "n_threads": 24, "n_ctx": 512,
    }}}
    a3 = run_mod.build_a3(cfg, backend="llamacpp")
    assert a3.n_gpu_layers == 0
    assert a3.n_threads == 24
    assert a3.n_ctx == 512


def test_build_a3_defaults_n_gpu_layers_when_config_omits_it():
    cfg = {"annotators": {"a3_llm": {"model": "fake.gguf"}}}
    a3 = run_mod.build_a3(cfg, backend="llamacpp")
    assert a3.n_gpu_layers == -1
    assert a3.n_threads is None


# ============================================================================
# run.py: CLI plumbing, vote parquet schema, models.lock.json, platform carry-through
# ============================================================================

def _write_normalized_parquet(path, rows) -> None:
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def test_run_writes_vote_parquets_matching_the_contract_schema(tmp_path, lexicon_dir):
    input_path = tmp_path / "comments_normalized.parquet"
    rows = [
        {"comment_id": "c1", "platform": "youtube", "author_hash": "h1",
         "published_at": "2026-01-01T00:00:00Z", "text_original": "goblok lu",
         "text_normalized": "goblok lu", "split": "dev"},
        {"comment_id": "c2", "platform": "instagram", "author_hash": "h2",
         "published_at": "2026-01-01T00:00:00Z",
         "text_original": "videonya jelek banget kualitasnya",
         "text_normalized": "videonya jelek banget kualitasnya", "split": "dev"},
        {"comment_id": "c3", "platform": "x", "author_hash": "h3",
         "published_at": "2026-01-01T00:00:00Z", "text_original": "makasih ya kak",
         "text_normalized": "makasih ya kak", "split": "dev"},
        {"comment_id": "held1", "platform": "youtube", "author_hash": "h4",
         "published_at": "2026-01-01T00:00:00Z", "text_original": "goblok banget",
         "text_normalized": "goblok banget", "split": "heldout"},
    ]
    _write_normalized_parquet(input_path, rows)

    config = {"annotators": {"a2_rules": {"lexicon_dir": str(lexicon_dir)}}}
    summary = run_mod.run(
        config=config, input_path=input_path, root=tmp_path,
        annotators=("a1", "a2", "a3"), backend="stub", split="dev",
    )

    assert summary["n_comments"] == 3, "the heldout row must be excluded from a dev run"

    a1_df = pd.read_parquet(tmp_path / "data/interim/votes_a1.parquet")
    a2_df = pd.read_parquet(tmp_path / "data/interim/votes_a2.parquet")
    a3_df = pd.read_parquet(tmp_path / "data/interim/votes_a3.parquet")

    assert list(a1_df.columns) == ["comment_id", "annotator_id", "signal", "raw_json", "confidence", "platform"]
    assert list(a2_df.columns) == ["comment_id", "annotator_id", "signal", "raw_json", "confidence", "platform"]
    assert list(a3_df.columns) == ["comment_id", "annotator_id", "label", "raw_json", "confidence", "platform"]

    assert set(a1_df["comment_id"]) == {"c1", "c2", "c3"}
    assert "held1" not in set(a1_df["comment_id"])
    for cid in a1_df["raw_json"]:
        json.loads(cid)  # must be valid, parseable JSON

    lock = json.loads((tmp_path / "models.lock.json").read_text(encoding="utf-8"))
    assert set(lock) == {"a1", "a2", "a3"}
    assert lock["a2"]["rules_sha256"]
    assert lock["a3"]["NOT_A_REAL_MODEL"] is True


def test_run_carries_platform_through_to_vote_parquets(tmp_path, lexicon_dir):
    """R9/D35: annotators never branch on platform, but run.py must still carry the
    column through so per-platform reporting works downstream."""
    input_path = tmp_path / "comments_normalized.parquet"
    rows = [
        {"comment_id": "c1", "platform": "instagram", "text_original": "goblok lu",
         "text_normalized": "goblok lu", "split": "dev"},
        {"comment_id": "c2", "platform": "x", "text_original": "makasih ya",
         "text_normalized": "makasih ya", "split": "dev"},
    ]
    _write_normalized_parquet(input_path, rows)
    config = {"annotators": {"a2_rules": {"lexicon_dir": str(lexicon_dir)}}}

    run_mod.run(config=config, input_path=input_path, root=tmp_path,
                annotators=("a2",), backend="stub", split="dev")

    a2_df = pd.read_parquet(tmp_path / "data/interim/votes_a2.parquet")
    got = dict(zip(a2_df["comment_id"], a2_df["platform"]))
    assert got == {"c1": "instagram", "c2": "x"}


def test_run_falls_back_gracefully_when_platform_column_is_absent(tmp_path, lexicon_dir):
    """Today's fixture output (pre-D35) has no `platform` column. run.py must not crash;
    it writes an empty platform rather than inventing one."""
    input_path = tmp_path / "comments_normalized.parquet"
    rows = [
        {"comment_id": "c1", "channel_id": "UC1", "video_id": "v1",
         "text_original": "goblok lu", "text_normalized": "goblok lu", "split": "dev"},
    ]
    _write_normalized_parquet(input_path, rows)
    config = {"annotators": {"a2_rules": {"lexicon_dir": str(lexicon_dir)}}}

    run_mod.run(config=config, input_path=input_path, root=tmp_path,
                annotators=("a2",), backend="stub", split="dev")

    a2_df = pd.read_parquet(tmp_path / "data/interim/votes_a2.parquet")
    assert a2_df.loc[0, "platform"] == ""


def test_run_respects_split_flag_for_heldout(tmp_path, lexicon_dir):
    input_path = tmp_path / "comments_normalized.parquet"
    rows = [
        {"comment_id": "dev1", "text_original": "goblok", "text_normalized": "goblok", "split": "dev"},
        {"comment_id": "held1", "text_original": "goblok", "text_normalized": "goblok", "split": "heldout"},
    ]
    _write_normalized_parquet(input_path, rows)
    config = {"annotators": {"a2_rules": {"lexicon_dir": str(lexicon_dir)}}}

    summary = run_mod.run(config=config, input_path=input_path, root=tmp_path,
                           annotators=("a2",), backend="stub", split="heldout")
    assert summary["n_comments"] == 1
    a2_df = pd.read_parquet(tmp_path / "data/interim/heldout/votes_a2.parquet")
    assert list(a2_df["comment_id"]) == ["held1"]


def test_r3_the_heldout_run_writes_beside_the_dev_votes_never_over_them(tmp_path, lexicon_dir):
    """Both splits are annotated by the same frozen system, so one set of output paths
    looks natural and is wrong. `--split heldout` over the dev paths would make
    `python -m src.consensus` rebuild consensus.parquet from Channel E alone, turning
    every dev-side number - phrase families, the baseline arms, the dev column of the
    transfer table - into a held-out number. No frozen file changes, so the freeze
    cannot detect it.
    """
    input_path = tmp_path / "comments_normalized.parquet"
    _write_normalized_parquet(input_path, [
        {"comment_id": "dev1", "text_original": "goblok", "text_normalized": "goblok",
         "split": "dev"},
        {"comment_id": "held1", "text_original": "goblok", "text_normalized": "goblok",
         "split": "heldout"},
    ])
    config = {"annotators": {"a2_rules": {"lexicon_dir": str(lexicon_dir)}}}
    kwargs = dict(config=config, input_path=input_path, root=tmp_path,
                  annotators=("a2",), backend="stub")

    run_mod.run(**kwargs, split="dev")
    run_mod.run(**kwargs, split="heldout")          # must not clobber the dev run

    dev = pd.read_parquet(tmp_path / "data/interim/votes_a2.parquet")
    held = pd.read_parquet(tmp_path / "data/interim/heldout/votes_a2.parquet")
    assert list(dev["comment_id"]) == ["dev1"], "the development votes survived"
    assert list(held["comment_id"]) == ["held1"]


def test_votes_path_keeps_the_documented_dev_paths_unchanged():
    """Build contract §3 names the dev paths; only the held-out run is namespaced."""
    assert run_mod.votes_path("a1", "dev") == run_mod.VOTES_A1_PATH
    assert run_mod.votes_path("a3", "heldout").as_posix() == \
        "data/interim/heldout/votes_a3.parquet"
    with pytest.raises(ValueError):
        run_mod.votes_path("a1", "nonsense")


def test_run_respects_limit_flag(tmp_path, lexicon_dir):
    input_path = tmp_path / "comments_normalized.parquet"
    rows = [
        {"comment_id": f"c{i}", "text_original": "makasih", "text_normalized": "makasih", "split": "dev"}
        for i in range(10)
    ]
    _write_normalized_parquet(input_path, rows)
    config = {"annotators": {"a2_rules": {"lexicon_dir": str(lexicon_dir)}}}

    summary = run_mod.run(config=config, input_path=input_path, root=tmp_path,
                           annotators=("a2",), backend="stub", split="dev", limit=3)
    assert summary["n_comments"] == 3


def test_run_rejects_unknown_annotator_name(tmp_path, lexicon_dir):
    input_path = tmp_path / "comments_normalized.parquet"
    _write_normalized_parquet(input_path, [
        {"comment_id": "c1", "text_original": "hi", "text_normalized": "hi", "split": "dev"},
    ])
    config = {"annotators": {"a2_rules": {"lexicon_dir": str(lexicon_dir)}}}
    with pytest.raises(ValueError):
        run_mod.run(config=config, input_path=input_path, root=tmp_path,
                     annotators=("a9",), backend="stub")


def test_run_preserves_other_annotators_identity_across_partial_runs(tmp_path, lexicon_dir):
    """`--annotators a2` after a full run must not erase a1's/a3's recorded identity -
    models.lock.json is inside the frozen surface (R3) and is meant to accumulate."""
    input_path = tmp_path / "comments_normalized.parquet"
    _write_normalized_parquet(input_path, [
        {"comment_id": "c1", "text_original": "goblok", "text_normalized": "goblok", "split": "dev"},
    ])
    config = {"annotators": {"a2_rules": {"lexicon_dir": str(lexicon_dir)}}}

    run_mod.run(config=config, input_path=input_path, root=tmp_path,
                annotators=("a1", "a2", "a3"), backend="stub", split="dev")
    run_mod.run(config=config, input_path=input_path, root=tmp_path,
                annotators=("a2",), backend="stub", split="dev")

    lock = json.loads((tmp_path / "models.lock.json").read_text(encoding="utf-8"))
    assert set(lock) == {"a1", "a2", "a3"}


def test_models_lock_is_stable_across_dev_and_heldout_runs(tmp_path, lexicon_dir):
    """R3: the dev run writes models.lock.json, the operator locks it, then the
    heldout run reproduces it byte-for-byte so freeze.verify() sees no drift. Model
    identity must not depend on which comments were annotated."""
    input_path = tmp_path / "comments_normalized.parquet"
    _write_normalized_parquet(input_path, [
        {"comment_id": "dev1", "text_original": "goblok", "text_normalized": "goblok", "split": "dev"},
        {"comment_id": "held1", "text_original": "tolol beda konteks", "text_normalized": "tolol beda konteks", "split": "heldout"},
    ])
    config = {"annotators": {"a2_rules": {"lexicon_dir": str(lexicon_dir)}}}

    run_mod.run(config=config, input_path=input_path, root=tmp_path,
                annotators=("a1", "a2", "a3"), backend="stub", split="dev")
    dev_lock = (tmp_path / "models.lock.json").read_text(encoding="utf-8")

    run_mod.run(config=config, input_path=input_path, root=tmp_path,
                annotators=("a1", "a2", "a3"), backend="stub", split="heldout")
    heldout_lock = (tmp_path / "models.lock.json").read_text(encoding="utf-8")

    assert dev_lock == heldout_lock


def test_a3_off_schema_run_is_counted_in_the_run_summary(tmp_path, lexicon_dir, monkeypatch):
    input_path = tmp_path / "comments_normalized.parquet"
    _write_normalized_parquet(input_path, [
        {"comment_id": "c1", "text_original": "apapun", "text_normalized": "apapun", "split": "dev"},
    ])
    config = {"annotators": {"a2_rules": {"lexicon_dir": str(lexicon_dir)}}}

    def fake_build_a3(cfg, backend):
        return A3LocalLLM(backend="stub", generate_fn=lambda text: "off_schema_junk")

    monkeypatch.setattr(run_mod, "build_a3", fake_build_a3)
    summary = run_mod.run(config=config, input_path=input_path, root=tmp_path,
                           annotators=("a3",), backend="stub", split="dev")
    assert summary["a3"]["n_off_schema"] == 1
    assert summary["a3"]["n_votes"] == 0


# ============================================================================
# import hygiene - heavy backends must stay lazy
# ============================================================================

def test_annotate_modules_import_without_torch_or_transformers_installed():
    """These modules must be importable in an environment with neither torch nor
    transformers on sys.path (this venv). A module-level import of either would fail
    this test with an ImportError before any assertion runs."""
    for name in (
        "src.annotate.a1_indobertweet",
        "src.annotate.a2_rules",
        "src.annotate.a3_llm",
        "src.annotate.run",
    ):
        importlib.reload(importlib.import_module(name))
