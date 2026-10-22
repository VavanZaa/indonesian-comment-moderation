"""Indonesian text normalization.  **FROZEN SURFACE** (AGENTS.md R3).

Everything in this module and in `data/raw/lexicons/` is hashed by `src/freeze.py`.
That means two things:

1. It must be **deterministic**. No set iteration order, no dict-insertion-order
   dependence, no locale-dependent casing. Same input, same bytes, forever.
2. Once the freeze is locked you do not "just fix" a slang form you noticed while
   glancing at Channel E. That is the exact failure R3 exists to catch.

Two folds, deliberately different:

    normalize(text)             pipeline text.  Conservative.  Emoji KEPT.
    canonicalize_variant(text)  family-grouping key only.  Aggressive, lossy.

`normalize` output is what annotators read, what phrase mining n-grams over, and what
`baseline_1` matches against (R6).  It has to stay readable Indonesian: if it mangled
`maaf` into `maf` or ate the emoji off `wkwkwk goblok lu 😭`, the criticism-vs-harassment
boundary that the whole demo rests on (AGENTS.md TRAP 1) would be destroyed before any
annotator saw the text.

`canonicalize_variant` is only ever a dictionary key in `src/phrases/grouping.py`.  It is
allowed to be lossy because nothing is ever displayed from it.

R7: `text_original` is never overwritten.  The CLI adds a column, it does not replace one.

R9/D35: comments now arrive from YouTube, Instagram and X, so `normalize` also strips the
platform furniture that would otherwise mine as a phrase n-gram - leading `@mentions` and
reply chains of them, `#hashtags` (word kept, marker dropped), `RT @x:` prefixes, and
t.co/bit.ly links. See `_strip_platform_furniture` below.
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

__all__ = [
    "Normalizer",
    "normalize",
    "canonicalize_variant",
    "load_kamus_alay",
    "load_abusive_terms",
    "load_rule_families",
    "load_static_profanity",
    "main",
]

# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------

#: repo root, derived from this file, so the default lexicons resolve regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LEXICON_DIR = Path("data/raw/lexicons")
DEFAULT_CONFIG = Path("config/local.yaml")
DEFAULT_INPUT = Path("data/raw/comments_raw.parquet")
DEFAULT_OUTPUT = Path("data/interim/comments_normalized.parquet")
STATIC_PROFANITY_PATH = Path("data/raw/static_profanity_id.txt")


def _resolve(p: Path | str) -> Path:
    """Accept a repo-relative path from any working directory."""
    p = Path(p)
    if p.exists():
        return p
    candidate = _REPO_ROOT / p
    return candidate if candidate.exists() else p


# --------------------------------------------------------------------------------------
# character-level tables
# --------------------------------------------------------------------------------------

#: Pinned by docs/05-build-contract.md §2. Do not extend without re-locking the freeze.
LEET_MAP: dict[str, str] = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
    "@": "a", "$": "s",
}

#: Censor markers. `@` and `$` are NOT here - the contract makes them leet substitutions.
MASK_CHARS = frozenset("*#%")

#: Dropped outright. Invisible characters are pure obfuscation carriers.
_ZERO_WIDTH = frozenset({
    "​",  # zero width space
    "‌",  # zero width non-joiner
    "⁠",  # word joiner
    "﻿",  # zero width no-break space / BOM
    "­",  # soft hyphen
    "᠎",  # mongolian vowel separator
    "⁡", "⁢", "⁣", "⁤",  # invisible math operators
})

#: Kept ONLY when they are gluing an emoji together. `❤️` must survive as `❤️`, but
#: `g‍o‍b‍l‍o‍k` must not survive as six letters.
_EMOJI_GLUE = frozenset({"‍", "️", "︎", "⃣"})

#: Separator-obfuscation characters: `g.o.b.l.o.k`, `g-o-b-l-o-k`, `g_o_b_l_o_k`.
_SEP_CLASS = r"[.\-_·•~^|/\\+,:;]"

#: `g.o.b.l.o.k` needs at least three separators, so ordinary abbreviations (`a.n`,
#: `t.co`) and decimals are untouched.
_SEP_OBFUSCATED = re.compile(
    rf"(?<![0-9a-z])([0-9a-z](?:{_SEP_CLASS}[0-9a-z]){{2,}})(?![0-9a-z])"
)
_SEP_STRIP = re.compile(_SEP_CLASS)

#: A "word core" for token-level rules. Mask chars stay inside it so `g*blok` is one core.
_CORE = re.compile(r"[0-9a-z@$*#%]+")

#: Single-character token that a human split a word into. Emoji are excluded by
#: construction - they are not in this class - so `😂 😭 🤣` is never glued into one token.
_SINGLE = re.compile(r"^[0-9a-z@$*#%]$")

#: AGENTS.md contract: 3+ repeats collapse to ONE, and only for ASCII letters.
#:
#: Two repeats are deliberately left alone. Indonesian is full of legitimate doubled
#: letters - `maaf`, `saat`, `taat`, `massa`, `diikuti`, `keesokan` - and collapsing them
#: would corrupt ordinary words and, worse, silently merge unrelated phrase families.
#: The cost is that an elongated `maaaaf` lands on `maf`; that is repaired in
#: `kamus_alay.csv` (`maf -> maaf`), which is a real alay spelling anyway.
#:
#: Emoji and punctuation runs are NOT collapsed. `😂😂😂` and `!!!` are tone, and tone is
#: what separates joking from harassment (TRAP 1).
_ELONGATION = re.compile(r"([a-z])\1{2,}")

#: Laughter is elongated by SYLLABLE, not by character, so `wkwkwkwkwk` survives
#: `_ELONGATION` untouched. Indonesian comment laughter is a tone signal worth keeping as
#: a token (demo scene 5: `wkwkwk goblok lu 😭`), so it is canonicalised, not deleted.
_LAUGH = re.compile(r"^(wk|ha|he|hi|hu|xi)\1+[a-z]?$")


# --------------------------------------------------------------------------------------
# platform furniture (R9/D35) - YouTube, Instagram and X arrive with different junk
# --------------------------------------------------------------------------------------
#
# Comments now come from three apps (build contract §2). Without this step, an X reply
# opening with four `@handles` mines those handles as a 4-gram phrase candidate, and a
# `RT @user:` prefix or a t.co/bit.ly link does the same. All four patterns only ever
# DELETE (or, for hashtags, delete one character) - they never invent or rewrite content,
# so this step commutes cleanly with everything that follows it and runs once, first, on
# the raw string.

#: Twitter/X retweet marker plus the handle it quotes: `"RT @user: text"` -> `"text"`.
#: Tried BEFORE `_LEADING_MENTIONS` - that pattern alone would eat `@user` and leave a
#: dangling `"RT :"` behind.
_RT_PREFIX = re.compile(r"^\s*RT\s+@[A-Za-z0-9_.]{1,30}\s*:\s*", re.IGNORECASE)

#: A leading run of `@mentions` - an X reply chain (`@a @b @c @d dia goblok banget`).
#: LEADING only: an `@mention` in the middle of a sentence is content (someone being
#: talked about or at), not furniture, and is left alone. R8 already hashes the
#: *commenter's* handle at ingest; this is about not mining someone else's handle as a
#: phrase n-gram, not a second privacy pass.
_LEADING_MENTIONS = re.compile(r"^(?:@[A-Za-z0-9_.]{1,30}[,:]?\s*)+")

#: `#goblok` -> `goblok`: keep the word, drop the marker, anywhere in the comment (not
#: just leading - a hashtag can land mid-sentence). The lookbehind requires the `#` to
#: sit at a token boundary, so a mid-word censor mask (`g#blok`, which MASK_CHARS still
#: owns) is never touched by this.
_HASHTAG = re.compile(r"(?<!\S)#(\w+)")

#: Shortened-URL furniture: t.co is X's own shortener, bit.ly is the other one people
#: paste into comments. The whole URL goes (scheme optional), so "cek bit.ly/xyz ya"
#: loses the link but keeps the sentence around it intact.
_SHORT_URL = re.compile(r"\b(?:https?://)?(?:www\.)?(?:t\.co|bit\.ly)/\S+", re.IGNORECASE)


def _strip_platform_furniture(text: str) -> str:
    """R9/D35. Order matters: the RT-prefix pass must run before the leading-mentions
    pass, or the mention half of `"RT @user:"` is already gone and a bare `"RT :"`
    survives to be mined as junk.
    """
    s = _RT_PREFIX.sub("", text)
    s = _LEADING_MENTIONS.sub("", s)
    s = _SHORT_URL.sub("", s)
    s = _HASHTAG.sub(r"\1", s)
    return s


# --------------------------------------------------------------------------------------
# lexicon loading
# --------------------------------------------------------------------------------------

def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def load_kamus_alay(path: Path | str = DEFAULT_LEXICON_DIR) -> dict[str, str]:
    """`slang,formal` CSV -> mapping. Directory or file path both accepted."""
    path = _resolve(path)
    if path.is_dir():
        path = path / "kamus_alay.csv"
    rows = _read_lines(path)
    mapping: dict[str, str] = {}
    for row in rows:
        if "," not in row:
            continue
        slang, _, formal = row.partition(",")
        slang, formal = slang.strip().casefold(), formal.strip().casefold()
        if not slang or not formal or slang == "slang":
            continue
        mapping[slang] = formal
    return mapping


def load_abusive_terms(path: Path | str = DEFAULT_LEXICON_DIR) -> list[str]:
    """`word` CSV -> sorted unique terms. Used by A2 and by censor unmasking."""
    path = _resolve(path)
    if path.is_dir():
        path = path / "abusive_id.csv"
    rows = [r.strip().casefold() for r in _read_lines(path)]
    return sorted({r for r in rows if r and r != "word"})


def load_rule_families(path: Path | str = DEFAULT_LEXICON_DIR) -> dict[str, list[str]]:
    """`rule_families.toml` -> {family: [regex, ...]}.

    A2's resource, not the normalizer's. Exposed here because this module owns the file
    and A2 should not have to know its on-disk shape.
    """
    import tomllib

    path = _resolve(path)
    if path.is_dir():
        path = path / "rule_families.toml"
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {
        family: list(block.get("patterns", []))
        for family, block in data.items()
        if isinstance(block, dict)
    }


def load_static_profanity(path: Path | str = STATIC_PROFANITY_PATH) -> list[str]:
    """R6 baseline arm: the naive creator's blocked-word list, `#` comments stripped."""
    return sorted({t.casefold() for t in _read_lines(_resolve(path))})


# --------------------------------------------------------------------------------------
# character helpers
# --------------------------------------------------------------------------------------

def _is_pictographic(ch: str) -> bool:
    """Good enough emoji test: emoji are Symbol-other, plus the modifier/flag blocks."""
    cp = ord(ch)
    if 0x1F1E6 <= cp <= 0x1F1FF or 0x1F3FB <= cp <= 0x1F3FF:
        return True
    return unicodedata.category(ch) in ("So", "Sk")


def _clean_characters(text: str) -> str:
    """NFKC + casefold + strip zero-width and combining marks, emoji-aware.

    NFKC folds fullwidth `ｇｏｂｌｏｋ` onto ASCII. NFD then exposes the combining marks
    behind `góblók` so stripping category Mn removes diacritic obfuscation and Zalgo.
    Emoji glue (ZWJ, VS16) is preserved when it follows a pictographic character, so
    `👨‍👩‍👧` and `❤️` come out intact while ZWJ word-splitting does not.
    """
    s = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text))
    out: list[str] = []
    for ch in s:
        if ch in _ZERO_WIDTH:
            continue
        if ch in _EMOJI_GLUE:
            if out and _is_pictographic(out[-1]):
                out.append(ch)
            continue
        if unicodedata.category(ch) == "Mn":
            continue
        out.append(ch)
    return "".join(out).casefold()


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _join_separated(text: str) -> str:
    """`g.o.b.l.o.k` and `g o b l o k` -> `goblok`.

    Both forms need at least three fragments, which is why ordinary Indonesian text
    survives: single-letter words are essentially nonexistent, and pure-digit runs are
    excluded by the two-letter minimum so `1 2 3` stays a list.
    """
    text = _SEP_OBFUSCATED.sub(lambda m: _SEP_STRIP.sub("", m.group(1)), text)

    tokens = text.split(" ")
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 3 and sum(t.isalpha() for t in run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run.clear()

    for tok in tokens:
        if _SINGLE.match(tok):
            run.append(tok)
        else:
            flush()
            out.append(tok)
    flush()
    return " ".join(t for t in out if t)


def _leet_core(core: str) -> str:
    """Leet-substitute one word core.

    Guards, both of which exist to protect ordinary text rather than to weaken the rule:
    - a core with no ASCII letter is left alone, so `2024` and `100` stay numbers;
    - cores of length <= 2 are left alone, so `s1`/`d3` (Indonesian degree levels) survive;
    - a leading `@`/`$` is left alone, so an `@mention` is not turned into a word.
    """
    if len(core) <= 2 or not any("a" <= c <= "z" for c in core):
        return core
    chars = list(core)
    for i, c in enumerate(chars):
        if i == 0 and c in "@$":
            continue
        chars[i] = LEET_MAP.get(c, c)
    return "".join(chars)


def _leet_all(text: str) -> str:
    """Unguarded leet fold. Only `canonicalize_variant` uses this."""
    return "".join(LEET_MAP.get(c, c) for c in text)


def _collapse_any_repeat(text: str) -> str:
    return re.sub(r"(.)\1+", r"\1", text)


# --------------------------------------------------------------------------------------
# the normalizer
# --------------------------------------------------------------------------------------

class Normalizer:
    """Loads the lexicons once and applies them. Callable, per the build contract.

    >>> n = Normalizer()
    >>> n("g0blokkk bgt sih 😂")
    'goblok banget sih 😂'
    """

    def __init__(self, lexicon_dir: Path = DEFAULT_LEXICON_DIR) -> None:
        self.lexicon_dir = Path(lexicon_dir)
        self.kamus_alay: dict[str, str] = load_kamus_alay(lexicon_dir)
        self.abusive: tuple[str, ...] = tuple(load_abusive_terms(lexicon_dir))

        # Vocabulary the censor-unmasking gate is allowed to reveal. `g*blok` only
        # becomes `goblok` because `goblok` is a known term; a masked token that folds
        # onto nothing recognisable is left masked rather than invented.
        known: set[str] = set(self.abusive)
        for slang, formal in self.kamus_alay.items():
            known.add(slang)
            known.update(re.findall(r"[0-9a-z]+", formal))
        known = {w for w in known if w and len(w) >= 3}

        self._known_by_len = self._index(known)
        self._folded_by_len = self._index(
            {_collapse_any_repeat(_leet_all(w)) for w in known}
        )

    @staticmethod
    def _index(words: set[str]) -> dict[int, tuple[str, ...]]:
        by_len: dict[int, list[str]] = {}
        for w in words:
            by_len.setdefault(len(w), []).append(w)
        # sorted: the unmask candidate search must not depend on set iteration order.
        return {k: tuple(sorted(v)) for k, v in by_len.items()}

    # -- individual steps ---------------------------------------------------------

    def _unmask(self, core: str, index: dict[int, tuple[str, ...]]) -> str:
        """`g*blok` -> `goblok`, but only when exactly one known term fits the mask.

        Ambiguity is left masked on purpose. Picking a winner among several candidates
        would make the normalizer invent evidence, and the invented word would then be
        counted in a phrase family's `n_all`.
        """
        if len(core) < 3 or not any(c in MASK_CHARS for c in core):
            return core
        pattern = re.compile(
            "".join("." if c in MASK_CHARS else re.escape(c) for c in core) + r"\Z"
        )
        candidates = [w for w in index.get(len(core), ()) if pattern.match(w)]
        return candidates[0] if len(candidates) == 1 else core

    def _rewrite_core(self, core: str) -> str:
        core = self._unmask(core, self._known_by_len)
        core = _LAUGH.sub(lambda m: m.group(1) * 2, core)
        return self.kamus_alay.get(core, core)

    # -- public API ---------------------------------------------------------------

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        """The pipeline fold. Conservative, readable, emoji preserved."""
        if not text:
            return ""
        s = _strip_platform_furniture(text)
        s = _clean_characters(s)
        s = _collapse_whitespace(s)
        s = _join_separated(s)
        s = _CORE.sub(lambda m: _leet_core(m.group(0)), s)
        s = _ELONGATION.sub(r"\1", s)
        s = _CORE.sub(lambda m: self._rewrite_core(m.group(0)), s)
        return _collapse_whitespace(s)

    def canonicalize_variant(self, text: str) -> str:
        """The family-grouping key. Aggressive and lossy - never use on pipeline text.

        >>> Normalizer().canonicalize_variant("g*b l 0 kkk")
        'goblok'
        """
        s = self.normalize(text)
        # everything that is not a letter or digit goes, including spaces and emoji
        s = "".join(c for c in s if c.isalnum() or c in MASK_CHARS)
        s = _leet_all(s)
        s = _collapse_any_repeat(s)
        s = self._unmask(s, self._folded_by_len)
        return "".join(c for c in s if c not in MASK_CHARS)


@lru_cache(maxsize=1)
def _default() -> Normalizer:
    return Normalizer()


def normalize(text: str) -> str:
    """Pipeline normalization. See `Normalizer.normalize`."""
    return _default().normalize(text)


def canonicalize_variant(text: str) -> str:
    """Family-grouping key. See `Normalizer.canonicalize_variant`."""
    return _default().canonicalize_variant(text)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _load_config(path: Path) -> dict:
    path = _resolve(path)
    if not path.is_file():
        print(
            f"[normalize] warning: {path} not found - using contract default paths. "
            f"Copy config/poc.yaml to config/local.yaml before locking the freeze (R3).",
            file=sys.stderr,
        )
        return {}
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def run(
    input_path: Path,
    output_path: Path,
    lexicon_dir: Path = DEFAULT_LEXICON_DIR,
) -> int:
    """Read the raw corpus, add `text_normalized`, write it back out.

    Both splits are normalized. `split == "heldout"` rows are NOT filtered here: they
    still need `text_normalized` for `src/report.py heldout` and for baseline_1. It is
    every stage *downstream* of this one that filters to dev (build contract §1).
    """
    import pandas as pd

    input_path = _resolve(input_path)
    if not input_path.is_file():
        print(
            f"[normalize] ERROR: input corpus not found: {input_path}\n"
            f"  Run the ingest stage first:\n"
            f"    python -m src.ingest.fixture --config config/local.yaml\n"
            f"    python -m src.ingest.youtube --config config/local.yaml",
            file=sys.stderr,
        )
        return 2

    df = pd.read_parquet(input_path)
    if "text_original" not in df.columns:
        print(
            f"[normalize] ERROR: {input_path} has no `text_original` column "
            f"(found: {list(df.columns)}). R7 requires it.",
            file=sys.stderr,
        )
        return 2

    normalizer = Normalizer(lexicon_dir)
    originals = df["text_original"].tolist()
    # R7: append a column, never assign back into text_original.
    df["text_normalized"] = [
        normalizer("" if t is None or t != t else str(t)) for t in originals
    ]
    assert df["text_original"].tolist() == originals, "R7 violated: text_original changed"

    output_path = Path(output_path)
    if not output_path.is_absolute() and not output_path.parent.exists():
        output_path = _REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)

    changed = sum(o != n for o, n in zip(originals, df["text_normalized"]))
    print(f"[normalize] {len(df)} comments -> {output_path}")
    print(f"[normalize] {changed} rewritten, {len(df) - changed} unchanged")
    print(f"[normalize] lexicon: {len(normalizer.kamus_alay)} alay pairs, "
          f"{len(normalizer.abusive)} abusive terms")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.normalize.text",
        description="Indonesian text normalization (FROZEN SURFACE, AGENTS.md R3).",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", default=None, help="override the raw corpus path")
    parser.add_argument("--output", default=None, help="override the output path")
    parser.add_argument("--lexicons", default=None, help="override the lexicon directory")
    args = parser.parse_args(argv)

    cfg = _load_config(Path(args.config))
    paths = cfg.get("paths") or {}
    return run(
        input_path=Path(args.input or paths.get("comments_raw", DEFAULT_INPUT)),
        output_path=Path(args.output or paths.get("comments_normalized", DEFAULT_OUTPUT)),
        lexicon_dir=Path(args.lexicons or paths.get("lexicon_dir", DEFAULT_LEXICON_DIR)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
