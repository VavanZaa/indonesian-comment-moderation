"""Candidate generation (build contract §5): whole-token n-grams over
`text_normalized`, matched back to comments through an inverted index, scored
through the frozen `score_phrase`. `match_ids` is what family grouping needs."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.phrases.candidates import (
    Candidate,
    candidates_payload,
    generate_candidates,
    load_platform_map,
    phrase_matches,
    score_candidates,
    segments,
    tokenize,
)
from src.schemas import ConsensusRecord, Label, Tier

_WEIGHT = {Tier.HIGH: 1.0, Tier.MEDIUM: 0.5, Tier.AMBIGUOUS: 0.0}


def _rec(cid: str, label: Label | None, tier: Tier = Tier.HIGH) -> ConsensusRecord:
    return ConsensusRecord(cid, label, tier, _WEIGHT[tier])


# --- whole-token matching (build contract §5, mine.py's own doctest) ---------

def test_phrase_matches_is_whole_token_not_a_substring():
    assert phrase_matches("jelek", "videonya jelek banget")
    assert not phrase_matches("jelek", "jeleknya keliatan parah")


def test_tokenize_keeps_a_suffixed_word_as_one_token():
    assert tokenize("jeleknya keliatan") == ["jeleknya", "keliatan"]


def test_generate_candidates_never_matches_a_word_that_only_shares_a_prefix():
    docs = [(f"c{i}", "jeleknya banget sih") for i in range(30)]
    cands = generate_candidates(docs, ngram_range=(1, 1), min_matches=25)
    assert "jelek" not in {c.phrase for c in cands}
    assert "jeleknya" in {c.phrase for c in cands}


# --- n-grams do not cross punctuation / emoji boundaries ----------------------

def test_ngrams_do_not_cross_a_punctuation_boundary():
    docs = [(f"c{i}", "goblok. mending mati") for i in range(30)]
    cands = generate_candidates(docs, ngram_range=(2, 2), min_matches=25)
    phrases = {c.phrase for c in cands}
    assert "mending mati" in phrases
    assert "goblok mending" not in phrases


def test_segments_splits_at_punctuation_and_emoji():
    assert segments("goblok. mending mati \U0001F602 anjir") == [
        ["goblok"], ["mending", "mati"], ["anjir"],
    ]


# --- inverted-index generation, min_matches, dedup within a comment ----------

def test_min_matches_filters_out_low_support_ngrams():
    docs = [(f"c{i}", "goblok banget") for i in range(24)]
    docs.append(("c_other", "something else entirely"))
    cands = generate_candidates(docs, ngram_range=(1, 1), min_matches=25)
    assert "goblok" not in {c.phrase for c in cands}

    docs.append(("c25", "goblok"))
    cands = generate_candidates(docs, ngram_range=(1, 1), min_matches=25)
    assert "goblok" in {c.phrase for c in cands}


def test_a_phrase_repeated_within_one_comment_counts_that_comment_once():
    docs = [("c1", "goblok banget goblok")] + [(f"c{i}", "goblok") for i in range(2, 26)]
    cands = generate_candidates(docs, ngram_range=(1, 1), min_matches=25)
    goblok = next(c for c in cands if c.phrase == "goblok")
    assert goblok.match_ids.count("c1") == 1
    assert goblok.n_all == 25


def test_downward_closure_stops_the_scan_once_a_level_has_no_survivors():
    """"common" repeats in every comment and survives at n=1; the unique suffix
    means no bigram ever repeats, so the loop must not keep scanning n=2..4."""
    docs = [(f"c{i}", f"common word{i}") for i in range(30)]
    cands = generate_candidates(docs, ngram_range=(1, 4), min_matches=25)
    assert {c.phrase for c in cands} == {"common"}
    assert all(c.n == 1 for c in cands)


# --- scoring: score_phrase is the only place counts are produced -------------

def test_score_candidates_ignores_the_training_weight_tier():
    """R2: phrase mining treats HIGH and MEDIUM as equally resolved."""
    high_records = {f"h{i}": _rec(f"h{i}", Label.HARASSMENT_PERSONAL, Tier.HIGH)
                    for i in range(10)}
    medium_records = {f"m{i}": _rec(f"m{i}", Label.HARASSMENT_PERSONAL, Tier.MEDIUM)
                       for i in range(10)}
    cand_high = Candidate(phrase="goblok", n=1, match_ids=sorted(high_records))
    cand_medium = Candidate(phrase="goblok", n=1, match_ids=sorted(medium_records))

    [(_, score_high)] = score_candidates([cand_high], high_records)
    [(_, score_medium)] = score_candidates([cand_medium], medium_records)

    assert score_high.n_harmful == score_medium.n_harmful == 10
    assert score_high.p_harmful_lower() == score_medium.p_harmful_lower() == 1.0


def test_score_candidates_drops_match_ids_with_no_consensus_record():
    records = {f"c{i}": _rec(f"c{i}", Label.HARASSMENT_PERSONAL) for i in range(5)}
    cand = Candidate(phrase="x", n=1, match_ids=[*records, "no_such_record"])
    [(_, score)] = score_candidates([cand], records)
    assert score.n_all == 5


# --- JSON payload: match_ids required for §6's union rule ---------------------

def test_candidates_payload_carries_match_ids_and_round_trips_json():
    records = {f"c{i}": _rec(f"c{i}", Label.HARASSMENT_PERSONAL) for i in range(25)}
    cand = Candidate(phrase="goblok", n=1, match_ids=sorted(records))
    scored = score_candidates([cand], records)
    payload = candidates_payload(scored, ngram_range=(1, 4), min_matches=25)

    assert payload["candidates"][0]["match_ids"] == sorted(records)
    assert payload["candidates"][0]["n_harmful"] == 25
    reloaded = json.loads(json.dumps(payload, ensure_ascii=False))
    assert reloaded == payload


# --- platform map (R9/D35) ---------------------------------------------------

def test_load_platform_map_reads_the_platform_column(tmp_path):
    frame = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3"],
        "label": ["harassment_personal", None, "benign_other"],
        "tier": ["high", "ambiguous", "high"],
        "weight": [1.0, 0.0, 1.0],
        "votes_json": ["{}", "{}", "{}"],
        "corroboration": [2, 0, 0],
        "platform": ["youtube", "instagram", None],
    })
    path = tmp_path / "consensus.parquet"
    frame.to_parquet(path, index=False)

    mapping = load_platform_map(path)
    assert mapping == {"c1": "youtube", "c2": "instagram", "c3": "unknown"}


def test_load_platform_map_missing_column_degrades_to_empty(tmp_path):
    frame = pd.DataFrame({"comment_id": ["c1"], "label": ["benign_other"]})
    path = tmp_path / "consensus.parquet"
    frame.to_parquet(path, index=False)
    assert load_platform_map(path) == {}


def test_load_platform_map_missing_file_is_a_loud_operator_error(tmp_path):
    with pytest.raises(SystemExit):
        load_platform_map(tmp_path / "nope.parquet")
