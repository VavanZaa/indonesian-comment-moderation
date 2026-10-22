"""Tests for the FROZEN SURFACE normalizer (AGENTS.md R3, build contract §2).

Every behaviour is exercised with real Indonesian examples, not invented tokens, because
this module is hashed by `src/freeze.py` and read by four other lanes.
"""
from __future__ import annotations

import sys

import pandas as pd
import pytest

from src.annotate.adapters import A2_HARM_FAMILIES
from src.normalize.text import (
    Normalizer,
    canonicalize_variant,
    load_abusive_terms,
    load_kamus_alay,
    load_rule_families,
    load_static_profanity,
    main,
    normalize,
)

# ------------------------------------------------------------------------------------
# public API shape - four other lanes import these exact names (build contract §2)
# ------------------------------------------------------------------------------------


def test_public_api_names_exist_with_the_contracted_signatures():
    assert callable(normalize)
    assert callable(canonicalize_variant)
    n = Normalizer()  # default lexicon_dir must work with no arguments
    assert callable(n)
    assert n("goblok") == normalize("goblok")


# ------------------------------------------------------------------------------------
# character-level cleaning
# ------------------------------------------------------------------------------------


def test_casefold_and_nfkc_fold_fullwidth_forms():
    assert normalize("GOBLOK") == "goblok"
    assert normalize("ＧＯＢＬＯＫ") == "goblok"  # fullwidth GOBLOK


def test_zero_width_characters_are_stripped():
    assert normalize("go​blok") == "goblok"  # zero width space mid-word


def test_combining_marks_are_stripped():
    assert normalize("góblok") == "goblok"  # combining acute accent


def test_whitespace_is_collapsed():
    assert normalize("goblok    banget   sih") == "goblok banget sih"
    assert normalize("  goblok  ") == "goblok"


def test_emoji_survive_normalization():
    """TRAP 1: emoji carry the tone that separates joking from harassment."""
    out = normalize("wkwkwk goblok lu \U0001F62D")
    assert "\U0001F62D" in out
    out2 = normalize("keren❤️ banget")  # heart + variation selector glue
    assert "❤️" in out2


# ------------------------------------------------------------------------------------
# leetspeak substitution (fixed map, unconditional)
# ------------------------------------------------------------------------------------


def test_leetspeak_substitution_uses_the_pinned_map():
    assert normalize("g0blok") == "goblok"
    assert normalize("an1r") == "anir"          # 1 -> i, unconditional
    assert normalize("s4ya") == "saya"           # 4 -> a
    assert normalize("b8m") == "bbm"             # 8 -> b (BBM = an IM app, not a slur)


def test_leet_guard_short_cores_are_left_alone():
    """`s1` (Indonesian school grade level) and similar short cores must not fold."""
    assert normalize("s1") == "s1"
    assert normalize("d3") == "d3"


# ------------------------------------------------------------------------------------
# elongation: 3+ same-letter repeats collapse to 1; doubled letters are NOT touched
# ------------------------------------------------------------------------------------


def test_elongation_of_three_or_more_collapses_to_one():
    assert normalize("goblokkk") == "goblok"
    assert normalize("anjirrrr") == "anjir"


def test_two_repeats_are_left_alone_real_indonesian_doubled_letters():
    """`maaf`, `saat` are ordinary Indonesian words with a genuine doubled vowel.
    Collapsing 2-repeats would corrupt them into `maf`/`sat` and silently merge
    unrelated phrase families - see the comment on `_ELONGATION` in text.py."""
    assert normalize("maaf") == "maaf"
    assert normalize("saat") == "saat"
    assert normalize("massa") == "massa"


def test_laughter_is_canonicalized_by_syllable_not_deleted():
    """`wkwkwkwkwk` is a tone signal (demo scene 5), not noise - kept, not stripped."""
    out = normalize("wkwkwkwkwk parah")
    assert "wk" in out or "haha" in out  # canonicalised via _LAUGH or kamus_alay
    assert out != " parah"


# ------------------------------------------------------------------------------------
# separator-obfuscation joining
# ------------------------------------------------------------------------------------


def test_dot_separated_obfuscation_is_joined():
    assert normalize("g.o.b.l.o.k") == "goblok"


def test_dash_separated_obfuscation_is_joined():
    assert normalize("g-o-b-l-o-k") == "goblok"


def test_space_separated_single_letters_are_joined():
    assert normalize("g o b l o k") == "goblok"


def test_ordinary_abbreviations_with_dots_survive():
    """`a.n` and similar short abbreviations must not be mangled - the separator
    pattern requires 3+ fragments, which real two-letter abbreviations never have."""
    assert normalize("a.n budi") == "a.n budi"


# ------------------------------------------------------------------------------------
# censor unmasking, gated on a known lexicon term
# ------------------------------------------------------------------------------------


def test_censor_mask_unmasks_to_a_known_term():
    assert normalize("g*blok") == "goblok"


def test_censor_mask_stays_masked_when_ambiguous_or_unknown():
    """No known 6-letter word matches `x*xxxx`, so the mask must not be guessed away."""
    out = normalize("q*zxwv")
    assert "*" in out


# ------------------------------------------------------------------------------------
# kamus-alay slang expansion
# ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slang,formal",
    [
        ("gk", "tidak"),
        ("bgt", "banget"),
        ("yg", "yang"),
        ("gua", "saya"),
        ("gue", "saya"),
        ("lu", "kamu"),
        ("lo", "kamu"),
    ],
)
def test_kamus_alay_expands_required_examples(slang, formal):
    assert normalize(slang) == formal


def test_njir_and_wkwk_are_in_the_lexicon():
    m = load_kamus_alay()
    assert "njir" in m
    assert "wkwk" in m


def test_kamus_alay_has_at_least_150_real_pairs():
    m = load_kamus_alay()
    assert len(m) >= 150
    # spot-check a few for genuineness, not invented tokens
    for slang in ("gk", "bgt", "yg", "krn", "dgn", "udh", "kalo"):
        assert slang in m


# ------------------------------------------------------------------------------------
# the demo family: one canonicalize_variant key for every spelling
# ------------------------------------------------------------------------------------


def test_demo_family_shares_one_canonicalize_variant_key():
    variants = ["goblok", "g0blok", "goblokkk", "g*blok", "g o b l o k"]
    keys = {canonicalize_variant(v) for v in variants}
    assert keys == {"goblok"}


def test_canonicalize_variant_contract_example():
    assert canonicalize_variant("g*b l 0 kkk") == "goblok"


def test_canonicalize_variant_is_never_used_by_normalize():
    """normalize() must stay conservative - it must not strip spaces/punctuation the
    way canonicalize_variant does."""
    assert normalize("goblok banget") == "goblok banget"
    assert canonicalize_variant("goblok banget") == "goblokbanget"


# ------------------------------------------------------------------------------------
# jelek survives intact (R6 / demo scene 2 depends on this)
# ------------------------------------------------------------------------------------


def test_jelek_survives_normalization_intact():
    assert normalize("jelek") == "jelek"
    assert normalize("videonya jelek banget menurutku") == "videonya jelek banget menurutku"


def test_jelek_is_not_in_the_abusive_or_rule_lexicons():
    """`jelek` must never fire A2's harm signal - see the exclusion note in
    rule_families.toml. If this test fails, someone added it back."""
    assert "jelek" not in load_abusive_terms()
    families = load_rule_families()
    for patterns in families.values():
        assert not any("jelek" in p for p in patterns)


# ------------------------------------------------------------------------------------
# R9/D35 platform furniture: mentions, hashtags, RT prefixes, short URLs
# ------------------------------------------------------------------------------------


def test_leading_mention_is_stripped():
    assert normalize("@someuser keren banget videonya") == "keren banget videonya"


def test_reply_chain_of_four_handles_is_not_mined_as_a_4gram():
    out = normalize("@user1 @user2 @user3 @user4 setuju banget sama ini")
    assert "@" not in out
    assert out == "setuju banget sama ini"


def test_midsentence_mention_is_kept_not_leading():
    """Only LEADING mentions are furniture; a mention mid-sentence is content."""
    out = normalize("makasih ya @creator sudah bikin video")
    assert "@creator" in out


def test_hashtag_marker_is_dropped_word_is_kept():
    assert normalize("#goblok banget orangnya") == "goblok banget orangnya"
    assert normalize("video ini #keren banget") == "video ini keren banget"


def test_rt_prefix_is_stripped():
    assert normalize("RT @someuser: dia emang goblok banget") == "dia memang goblok banget"


def test_tco_and_bitly_urls_are_stripped():
    out = normalize("cek https://t.co/AbC123XyZ ya guys")
    assert "t.co" not in out
    assert out == "cek ya guys"
    out2 = normalize("info lengkap di bit.ly/xyz123 ya")
    assert "bit.ly" not in out2
    assert out2 == "info lengkap di ya"


def test_mask_char_hash_still_used_for_midword_censoring_not_hashtags():
    """`#` at a token boundary is a hashtag marker; `#` glued mid-word is still a
    censor mask (MASK_CHARS) - the two must not be confused."""
    assert normalize("g#blok") == "goblok"       # mid-word censor mask, unmasked
    assert normalize("#goblok") == "goblok"      # leading hashtag marker, dropped


# ------------------------------------------------------------------------------------
# idempotence
# ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "g0blokkk bgt sih \U0001F602",
        "wkwkwk goblok lu \U0001F62D",
        "@user1 @user2 @user3 @user4 keren banget",
        "RT @someuser: #goblok banget parah",
        "maaf ya, saat ini belum bisa",
        "g.o.b.l.o.k",
        "g*blok",
    ],
)
def test_normalize_is_idempotent(text):
    once = normalize(text)
    twice = normalize(once)
    assert once == twice


# ------------------------------------------------------------------------------------
# R7: text_original is never overwritten
# ------------------------------------------------------------------------------------


def test_run_never_mutates_text_original(tmp_path):
    from src.normalize.text import run

    raw = pd.DataFrame(
        {
            "comment_id": ["youtube:c1", "youtube:c2"],
            "platform": ["youtube", "youtube"],
            "figure_id": ["figure_a", "figure_a"],
            "source_account": ["UC_x", "UC_x"],
            "post_id": ["v1", "v1"],
            "post_url": ["https://youtu.be/v1", "https://youtu.be/v1"],
            "parent_id": [None, None],
            "author_hash": ["h1", "h2"],
            "published_at": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "text_original": ["G0BLOK BGT!!!", "@a @b @c @d jelek banget wkwkwk"],
            "split": ["dev", "dev"],
            "genre": ["entertainment", "entertainment"],
            "collection_mode": ["fixture", "fixture"],
        }
    )
    in_path = tmp_path / "comments_raw.parquet"
    out_path = tmp_path / "comments_normalized.parquet"
    raw.to_parquet(in_path, engine="pyarrow", index=False)

    rc = run(in_path, out_path)
    assert rc == 0

    out = pd.read_parquet(out_path)
    assert out["text_original"].tolist() == raw["text_original"].tolist()
    assert "text_normalized" in out.columns
    # every input column preserved
    for col in raw.columns:
        assert col in out.columns
    assert out.loc[0, "text_normalized"] == "goblok banget!!!"
    assert out.loc[1, "text_normalized"] == "jelek banget haha"


def test_cli_reads_and_writes_the_contracted_paths(tmp_path, monkeypatch):
    """Exercises the actual CLI entry point (`python -m src.normalize.text`), not just
    the `run()` helper, with --input/--output/--lexicons overrides in a tmp_path."""
    raw = pd.DataFrame(
        {
            "comment_id": ["x:c1"],
            "text_original": ["ini G0BLOK banget!!"],
        }
    )
    in_path = tmp_path / "raw.parquet"
    out_path = tmp_path / "out" / "normalized.parquet"
    raw.to_parquet(in_path, engine="pyarrow", index=False)

    rc = main([
        "--input", str(in_path),
        "--output", str(out_path),
    ])
    assert rc == 0
    assert out_path.is_file()
    out = pd.read_parquet(out_path)
    assert out.loc[0, "text_original"] == "ini G0BLOK banget!!"
    assert out.loc[0, "text_normalized"] == "ini goblok banget!!"


def test_cli_fails_clearly_when_input_is_missing(tmp_path, capsys):
    rc = main(["--input", str(tmp_path / "does_not_exist.parquet"),
               "--output", str(tmp_path / "out.parquet")])
    assert rc != 0
    captured = capsys.readouterr()
    assert "ERROR" in captured.err


def test_cli_errors_when_text_original_column_missing(tmp_path):
    bad = pd.DataFrame({"not_text_original": ["hello"]})
    in_path = tmp_path / "raw.parquet"
    bad.to_parquet(in_path, engine="pyarrow", index=False)
    rc = main(["--input", str(in_path), "--output", str(tmp_path / "out.parquet")])
    assert rc != 0


# ------------------------------------------------------------------------------------
# resources: sizes and shapes pinned by the build contract
# ------------------------------------------------------------------------------------


def test_abusive_lexicon_has_at_least_120_terms_with_variants():
    terms = load_abusive_terms()
    assert len(terms) >= 120
    # spelling variants, not just base forms
    assert "goblok" in terms
    assert any(v in terms for v in ("gblok", "goblog", "gblk"))
    assert "anjing" in terms
    assert any(v in terms for v in ("anjg", "anjir", "njir", "anjay"))


def test_rule_families_keys_match_a2_harm_families_exactly():
    families = load_rule_families()
    assert set(families.keys()) == set(A2_HARM_FAMILIES)
    for family, patterns in families.items():
        assert patterns, f"[{family}] must not be empty - A2 coverage is load-bearing"


def test_static_profanity_contains_jelek_and_plain_goblok_only():
    words = load_static_profanity()
    assert "jelek" in words
    assert "goblok" in words
    variants = ("g0blok", "goblokkk", "gblok", "g*blok", "goblog", "g0b1ok")
    for v in variants:
        assert v not in words, f"static list must not contain goblok variant {v!r}"


def test_static_profanity_naively_blocks_more_than_the_learned_lexicon():
    """Demo scene 2 setup check: the naive list includes ordinary evaluative words
    ('jelek') that the curated abusive lexicon deliberately excludes."""
    static = set(load_static_profanity())
    abusive = set(load_abusive_terms())
    assert "jelek" in static
    assert "jelek" not in abusive


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
