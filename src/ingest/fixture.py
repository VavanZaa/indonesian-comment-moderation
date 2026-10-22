"""Seeded offline Indonesian comment corpus - the pipeline's no-API-key path.

    python -m src.ingest.fixture --config config/local.yaml [--seed 0]

Writes the same artefact as every live/import connector (build contract
§1), so every stage downstream is runnable and testable today. That makes this module
load-bearing rather than a toy: if the fixture cannot reproduce the six demo scenes
across three representative platforms, nothing downstream can be verified before the real collection
runs.

MULTI-PLATFORM (AGENTS.md R9 / D35). The unit of collection is a public FIGURE whose
synthetic comment section spans three apps. `split` is assigned per figure, never per
platform or account: every source of a held-out figure is held out with it, and
`_figures_from_config` raises if a config tries to disagree. Each platform also gets its
own textual character - X replies open with `@handles` and stay short and clipped,
Instagram is emoji-heavy with `#hashtags`, YouTube keeps the longer prose the templates
already produce - and the `goblok` family is deliberately given a different harmful rate
on X than on YouTube/Instagram, so the per-platform `platform_support` reporting has
something real to show (see `PLATFORM_GROUP_MIX_OVERRIDE`).

WHAT THE CORPUS HAS TO CONTAIN (docs/00-frozen-scope.md, "Demo scenes")

  1  `mending mati`      death-wish family, high harmful rate, ZERO criticism -> SUGGESTED
  2  `jelek`             mostly legitimate criticism of videos and products -> REJECTED.
                         This scene carries the project: the static list blocks `jelek`,
                         this system rejects it. So the fixture must make `jelek` genuinely
                         mostly criticism, not merely rare.
  3  `goblok` family     real spelling variants: goblok / g0blok / goblokkk / g*blok /
                         `g o b l o k`, so `canonicalize_variant` has something to group
                         and `baseline_1 - baseline_0` has something to measure.
  4  held-out figure     the same harmful FAMILIES in different surface wording, on a
                         figure never used for discovery, so transfer is a real test.
  5  ambiguous reel      `wkwkwk goblok lu`, harsh-but-fair criticism, in-group joking.
  6  static vs learned   handled by 2 + 3 together.

Plus the worked example from `src/phrases/mine.py`: `anjir` at intent proportions
40% harmful / 10% criticism / 50% ambiguous - the exact GROUP_MIX ratios below - which is
the case that looks recommendable at 0.80 on resolved labels and is correctly rejected at
0.40 on the lower bound. Quotas are exact fractions (largest remainder) at every level, so
those ratios hold regardless of how many figures/platforms the totals get split across.

Plus finding F2 in `docs/04-architecture-mapping.md`: comments that CONDEMN a harmful
phrase while containing it -

    jangan bilang "mending mati" dong, jahat banget

The PoC has no stance detection, so a blocked-words list hides these. They must be
findable in the screen check, because showing one on camera is the deliberate motivation
for stance detection in V1.

INTENTS ARE NOT LABELS. Each generated comment carries the intent it was written to
express. That intent stays in memory: it drives the composition, the statistics printed
here and the tests, and it is deliberately NOT written to disk. Downstream labels must
come from the annotators (R5) - a gold column in `comments_raw.parquet` would be an
invitation to score the pipeline against the fixture's own assumptions.

DETERMINISM. Same seed, byte-identical parquet. Every random draw comes from one
`random.Random(seed)` consumed in a fixed order, and all proportions are exact quotas
(largest remainder), not samples.

DEPENDENCY ON `src/ingest/base.py` (owned by a different lane). `generate_records` below
is entirely self-contained - no import of `.base` anywhere in it or in anything it calls -
so the corpus's composition is testable independently of whether that module has landed
yet. Only `to_rows` (schema projection: hashing, comment ids, provenance strings) and
`build` (writing parquet / the dev manifest) import it, and they do so lazily, inside the
function body, exactly where they are needed.
"""
from __future__ import annotations

import argparse
import random
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..schemas import Label

# --------------------------------------------------------------------------- constants
DEFAULT_CONFIG = "config/local.yaml"
SPLIT_DEV = "dev"
SPLIT_HELDOUT = "heldout"
SPLITS: frozenset[str] = frozenset({SPLIT_DEV, SPLIT_HELDOUT})

PLATFORMS: tuple[str, ...] = ("youtube", "instagram", "x")
#: Reach is uneven across apps (R9) - overridable via fixture.platform_weights in config.
DEFAULT_PLATFORM_WEIGHTS: dict[str, float] = {"youtube": 0.55, "instagram": 0.30, "x": 0.15}

# --------------------------------------------------------------------------- intents
HARASS = Label.HARASSMENT_PERSONAL.value
SEXUAL = Label.SEXUAL_HARM.value
THREAT = Label.THREAT_PRIVACY.value
CRITICISM = Label.LEGITIMATE_CRITICISM.value
BENIGN = Label.BENIGN_OTHER.value
#: not a Label - consensus emits no label for these (Tier.AMBIGUOUS). R2b, scene 5.
AMBIG = "ambiguous"

INTENT_ORDER = (HARASS, SEXUAL, THREAT, CRITICISM, BENIGN, AMBIG)
HARMFUL_INTENTS = frozenset({HARASS, SEXUAL, THREAT})


# --------------------------------------------------------------------------- composition
#: Share of a figure-platform's comments per group, before genre adjustment.
#: Tuned so the overall harmful rate is ~18%, not the 50% a naive generator produces.
#: A corpus that is half harmful would make every phrase look recommendable and the
#: demo would prove nothing.
BASE_SHARES: dict[str, float] = {
    "filler": 0.615,       # ordinary neutral/positive comments - the bulk of any comment section
    "criticism": 0.060,    # criticism that does not use `jelek`
    "jelek": 0.075,        # scene 2
    "goblok": 0.060,       # scene 3
    "death_wish": 0.035,   # scene 1
    "anjir": 0.050,        # the mine.py worked example
    "insult": 0.050,       # tolol / bego / bangsat - other harassment, for realism
    "threat": 0.020,
    "sexual": 0.015,
    "ingroup": 0.020,      # scene 5
}

#: Intent quota within each group. These are what actually produce the scenes.
GROUP_MIX: dict[str, dict[str, float]] = {
    "filler": {BENIGN: 1.0},
    # harsh-but-fair criticism shades into ambiguity - that is scene 5, not noise
    "criticism": {CRITICISM: 0.86, AMBIG: 0.14},
    # SCENE 2. C(jelek) ~= 0.62, far above max_criticism_share 0.10 -> REJECTED,
    # while a static blocked-word list hides all of it.
    "jelek": {CRITICISM: 0.62, HARASS: 0.15, AMBIG: 0.17, BENIGN: 0.06},
    # SCENE 3. Default (YouTube) mix; see PLATFORM_GROUP_MIX_OVERRIDE for X/Instagram,
    # which deliberately differ (R9: platform_support needs a real difference to show).
    "goblok": {HARASS: 0.75, AMBIG: 0.17, BENIGN: 0.04, CRITICISM: 0.04},
    # SCENE 1. L_H ~= 0.93, C = 0.00 exactly -> SUGGESTED. The BENIGN slice is the
    # F2 condemnation case, which is why it is not zero.
    "death_wish": {HARASS: 0.93, AMBIG: 0.03, BENIGN: 0.04},
    # mine.py's worked example, exactly: lower 0.400, resolved 0.800, upper 0.900.
    "anjir": {HARASS: 0.40, CRITICISM: 0.10, AMBIG: 0.50},
    "insult": {HARASS: 0.80, AMBIG: 0.16, BENIGN: 0.04},
    "threat": {THREAT: 0.90, AMBIG: 0.10},
    "sexual": {SEXUAL: 0.88, AMBIG: 0.12},
    "ingroup": {AMBIG: 1.0},
}

#: R9 / D35: at least one family must have a genuinely different harmful rate on one
#: platform. X's clipped, drive-by replies run harsher for `goblok`; Instagram (more
#: in-group joking, wkwk-heavy) runs softer. YouTube keeps the GROUP_MIX default above.
#: This is a real generation-time difference, not decoration - `platform_support` in
#: `recommendations.json` should show materially different lower bounds per platform.
PLATFORM_GROUP_MIX_OVERRIDE: dict[str, dict[str, dict[str, float]]] = {
    "x": {
        "goblok": {HARASS: 0.85, AMBIG: 0.11, BENIGN: 0.02, CRITICISM: 0.02},
    },
    "instagram": {
        "goblok": {HARASS: 0.60, AMBIG: 0.28, BENIGN: 0.08, CRITICISM: 0.04},
    },
}

#: Genre shifts the mix: a news figure argues, a gossip figure attacks. Multipliers on
#: BASE_SHARES, renormalised. Without this every figure has an identical profile and the
#: four dev figures add nothing over one figure of 4x the size.
GENRE_WEIGHTS: dict[str, dict[str, float]] = {
    "news": {"criticism": 1.9, "jelek": 1.2, "goblok": 0.7, "death_wish": 0.6,
             "sexual": 0.3, "ingroup": 0.5, "anjir": 0.7},
    "gossip": {"goblok": 1.4, "death_wish": 1.7, "sexual": 1.7, "threat": 1.5,
               "criticism": 0.7, "jelek": 0.8},
    "entertainment": {"anjir": 1.3, "ingroup": 1.4, "goblok": 1.1},
    "creator_vlog": {"jelek": 1.4, "criticism": 1.2, "ingroup": 1.2, "threat": 0.6,
                     "sexual": 0.5},
}


# --------------------------------------------------------------------------- vocabulary
DEV_SLOTS: dict[str, list[str]] = {
    "bgt": ["banget", "bgt", "bngt", "parah", "poll", "pisan"],
    "eh": ["sih", "dong", "nih", "deh", "woy", "cuy", "ya"],
    "konten": ["videonya", "kontennya", "editannya", "thumbnailnya", "lagunya",
               "filmnya", "podcastnya", "vlognya"],
    "produk": ["produknya", "barangnya", "paketnya", "hpnya", "sepatunya", "casingnya"],
    "kota": ["Medan", "Surabaya", "Makassar", "Bandung", "Palembang", "Malang",
             "Semarang", "Balikpapan", "Pontianak"],
    "topik": ["skandal kemarin", "drama di twitter", "review hp baru", "kasus itu",
              "isu yang lagi rame", "part 2 nya", "behind the scene nya"],
    "sapa": ["bang", "kak", "min", "bro", "mas", "bang admin"],
    "waktu": ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli"],
}

#: SCENE 4. The held-out figure says the same things a different way: different address
#: terms, different cities, different sentence frames. Same families, new surface.
HELDOUT_SLOTS: dict[str, list[str]] = {
    "bgt": ["banget", "bgt", "kebangetan", "sumpah", "beneran"],
    "eh": ["sih", "loh", "tuh", "kok", "ya"],
    "konten": ["siarannya", "uploadannya", "rekamannya", "coveran nya", "streamingnya",
               "klipnya", "reels nya"],
    "produk": ["produknya", "kirimannya", "tokonya", "merknya", "jamnya"],
    "kota": ["Jayapura", "Kupang", "Ambon", "Manado", "Banda Aceh", "Mataram"],
    "topik": ["kolaborasi kemarin", "episode lama", "unboxing nya", "berita itu",
              "konser tahun lalu"],
    "sapa": ["kak", "om", "bang", "bund", "suhu"],
    "waktu": ["Agustus", "September", "Oktober", "November", "Desember"],
}

#: SCENE 3 - the character families. Weights are an exact quota, so every variant listed
#: here is guaranteed to appear, including the low-frequency separator form.
DEV_GOBLOK_VARIANTS: dict[str, float] = {
    "goblok": 0.40,
    "goblokkk": 0.18,
    "g0blok": 0.16,
    "g*blok": 0.12,
    "g o b l o k": 0.08,
    "gblok": 0.06,
}
#: SCENE 4 - the same family reached by different obfuscations on the unseen figure.
HELDOUT_GOBLOK_VARIANTS: dict[str, float] = {
    "goblok": 0.34,
    "g0bl0k": 0.20,
    "goblokk": 0.20,
    "g#blok": 0.14,
    "g.o.b.l.o.k": 0.12,
}

EMOJI = ["\U0001F602", "\U0001F62D", "\U0001F525", "\U0001F480", "\U0001F64F",
         "\U0001F44F", "\U0001F923", "\U0001F605", "\U0001F60D", "\U0001F44D",
         "\U0001F926", "\U0001F631"]
PREFIXES = ["wkwk ", "hadeh ", "astaga ", "eh ", "ya ampun ", "buset ", "duh "]
SUFFIXES = [" wkwk", " wkwkwk", " ya", " lah", " sumpah", " serius"]
LEET = {"a": "4", "i": "1", "e": "3", "o": "0", "s": "5"}

#: Instagram furniture: generic, non-identifying hashtags/handles - never a real account
#: (R8/UU PDP concern is about commenter identity, but there is no reason to use anything
#: but obviously-synthetic handles here either).
IG_HASHTAGS = ["viral", "fyp", "konten", "explore", "trending", "beranda", "gosip",
               "reviewjujur", "kontenkreator", "fypindo", "kdrama", "ootd"]
X_HANDLES = ["warganet88", "user4829", "anon_id92", "netizen_id", "akun_random",
             "infoterkini", "gosiphariini", "user_biasa", "ceritanetizen", "rapijaya21"]

#: Groups whose exact surface form is what a demo scene is about. Decoration may add
#: emoji or an interjection around these, but must not elongate or leet-mangle the token
#: the scene is named after - the goblok family already carries its variation explicitly.
SCENE_CRITICAL = frozenset({"death_wish", "jelek", "goblok", "anjir"})


# --------------------------------------------------------------------------- templates
DEV_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "filler": {
        BENIGN: [
            "makasih {bgt} kontennya {sapa}",
            "pertamax",
            "salam dari {kota}",
            "{konten} keren {bgt}, ditunggu part 2 nya",
            "akhirnya upload juga",
            "yang nonton {waktu} 2026 mana suaranya",
            "request bahas {topik} {eh} {sapa}",
            "suaranya enak didenger, betah nontonnya",
            "gue langsung subscribe {eh}",
            "ini yang gue tunggu tunggu dari kemarin",
            "auto nonton sampe abis",
            "{sapa} tolong pin komen gue",
            "kok gue baru tau channel ini ya",
            "jam segini masih ada yang nonton ga",
            "ngakak parah bagian akhir",
            "informatif {bgt}, makasih {sapa}",
            "semangat terus {sapa} berkarya",
            "lagunya masuk playlist gue",
            "notif on terus buat channel ini",
            "bagus {bgt} penjelasannya, gampang dipahamin",
            "gue nonton sambil makan, mantap",
            "{konten} bikin mood gue naik",
            "izin share ke grup ya {sapa}",
            "part berikutnya kapan {eh}",
            "gue setuju sama poin nomor 3",
            "baru tau kalo gitu, makasih ilmunya",
            "kualitas audionya udah jauh lebih bagus dari yang dulu",
            "kolab sama channel sebelah dong",
            "gue nonton ulang buat yang ketiga kali",
            "yang setuju like dong",
            "lokasi syutingnya dimana {eh}",
            "murah meriah, worth it {bgt}",
            "udah lama nunggu konten kayak gini",
            "{sapa} sehat sehat terus ya",
            "gue ketawa sendiri di kamar",
            "algoritma bawa gue kesini",
            "sumpah relate {bgt} sama hidup gue",
            "{produk} nya bagus, gue udah pake sebulan aman",
            "makasih udah dibikinin tutorialnya",
            "berapa lama ngeditnya {eh}",
        ],
    },
    "criticism": {
        CRITICISM: [
            "audionya kekecilan {sapa}, tolong dinaikin dikit",
            "menurut gue argumennya kurang kuat di menit 5",
            "risetnya kurang dalam {eh}, datanya udah lama",
            "editingnya kebanyakan jumpcut jadi pusing nontonnya",
            "judulnya clickbait, isinya gak sesuai",
            "harusnya sumbernya dicantumin biar kredibel",
            "kualitas {konten} turun dibanding tahun lalu",
            "penjelasannya muter muter, intinya apa {eh}",
            "harganya gak sebanding sama kualitas {produk}",
            "durasinya kepanjangan, banyak yang bisa dipotong",
            "musik latarnya terlalu kenceng, suara narasi ketutup",
            "datanya salah {sapa}, coba dicek lagi sumbernya",
            "gue kurang setuju sama kesimpulannya, terlalu buru buru",
            "sayang pencahayaannya kurang, jadi gelap di bagian dalem",
            "kurang riset {eh}, banyak yang kelewat",
            "{produk} nya cepet rusak, gue kecewa sama after salesnya",
            "kameranya goyang terus, bikin mual",
            "bahasannya terlalu dangkal buat topik seberat ini",
            "gue berharap lebih {bgt} dari episode ini",
            "thumbnailnya gak nyambung sama isinya",
            "iklannya kebanyakan, tiap 2 menit muncul",
            "harusnya ada disclaimer di awal biar gak salah paham",
            "subtitle nya banyak typo {sapa}",
            "opininya bias {bgt} ke satu pihak",
        ],
        AMBIG: [
            "jujur episode ini gagal total, tapi gue tetep nunggu yang berikutnya",
            "kecewa {bgt} sama {konten} ini, gak sesuai ekspektasi sama sekali",
            "ini konten paling ngasal yang pernah gue tonton dari channel ini",
            "males banget sih kalo tiap minggu isinya begini terus",
            "buang buang waktu 20 menit gue",
            "gue unsubscribe kalo kayak gini terus {eh}",
        ],
    },
    # ---------------- SCENE 2 -------------------------------------------------
    "jelek": {
        CRITICISM: [
            "{konten} jelek {bgt} kualitasnya, pecah pas di zoom",
            "editingnya jelek {eh}, transisinya kasar",
            "audionya jelek, banyak noise di background",
            "{produk} jelek, baru seminggu udah rusak",
            "hasil kameranya jelek kalo malem, noise semua",
            "jujur aja {konten} kali ini jelek, gak kayak biasanya",
            "packagingnya jelek, sampe penyok pas nyampe",
            "sinyalnya jelek jadi buffering terus dari tadi",
            "warnanya jelek {eh}, grading nya kurang",
            "mixing lagunya jelek, vokalnya ketutup drum",
            "bahannya jelek buat harga segitu",
            "jelek {bgt} kualitas cetaknya, buram",
            "menurut gue naskahnya jelek, alurnya loncat loncat",
            "layoutnya jelek, tulisannya kekecilan",
            "kualitas jahitannya jelek, benangnya keluar semua",
            "jelek sih hasil akhirnya, padahal prosesnya keliatan niat",
            "{konten} bagus tapi audionya jelek {bgt}, sayang",
            "efek suaranya jelek, kayak dari hp",
            "jelek {eh} desainnya, gak enak diliat",
            "batterynya jelek, sehari 3 kali charge",
        ],
        HARASS: [
            "mukanya jelek {bgt} masih sok cantik",
            "orang jelek kayak lu gausah banyak gaya",
            "jelek {bgt} lu, mending gausah muncul di kamera",
            "muka jelek kelakuan jelek, lengkap",
        ],
        AMBIG: [
            "jelek jelek gini gue suka {eh}",
            "jelek sih tapi lucu, gue tetep nonton",
            "jelek {bgt} tapi gue ketawa terus",
            "jelek amat wkwk, tapi ya udahlah",
        ],
        BENIGN: [
            "gapapa jelek yang penting jujur",
            "jangan bilang jelek {eh}, orang udah usaha bikin",
        ],
    },
    # ---------------- SCENE 3 -------------------------------------------------
    "goblok": {
        HARASS: [
            "{gv} {bgt} lu, gak bisa mikir ya",
            "dasar {gv}, komen aja gak nyambung",
            "lu {gv} apa pura pura {gv}",
            "makanya sekolah biar gak {gv} kayak gini",
            "{gv} emang lu, disuruh baca dulu malah ngegas",
            "orang {gv} emang gitu, ngotot padahal salah",
            "otak lu dimana {eh}, {gv} {bgt}",
            "gausah komen kalo {gv}",
            "{gv} banget lu jadi orang, kasian",
            "sumpah {gv} {bgt} yang komen di atas",
            "kalo {gv} tuh diem aja, jangan nambah nambahin",
            "cuma orang {gv} yang percaya beginian",
        ],
        AMBIG: [
            "wkwkwk {gv} lu \U0001F62D",
            "{gv} {bgt} {eh} lu, ngakak gue",
            "gue yang {gv} apa emang susah ya soalnya",
            "{gv} nih gue, salah pencet mulu",
            "wkwk {gv} emang kita berdua",
            "anjay {gv} {bgt} wkwkwk",
        ],
        BENIGN: [
            "jangan panggil orang {gv} {eh}, gak sopan",
            "komen yang bilang {gv} tuh yang harusnya ditegur",
        ],
        CRITICISM: [
            "argumennya {gv} {eh}, gak ada datanya sama sekali",
            "kebijakannya {gv}, malah nyusahin orang banyak",
        ],
    },
    # ---------------- SCENE 1 -------------------------------------------------
    "death_wish": {
        HARASS: [
            "mending mati aja lu daripada bikin konten kayak gini",
            "orang kayak gini mending mati {eh}",
            "mending mati daripada malu maluin keluarga",
            "hidup lu gak guna, mending mati aja",
            "mending mati {eh} daripada ngemis perhatian terus",
            "serius mending mati aja, gak ada yang kehilangan",
            "kalo gue jadi lu mending mati",
            "mending mati aja deh, capek liat mukanya",
            "mati aja lu, gak usah balik lagi",
            "mendingan mati daripada jadi kayak dia",
            "mati aja sana, gak ada gunanya idup",
            "gue doain cepet mati aja orang begini",
        ],
        AMBIG: [
            "mending mati lampu daripada mati gaya wkwk",
            "mending mati kutu daripada gitu wkwkwk",
        ],
        BENIGN: [
            'jangan bilang "mending mati" dong, jahat banget',
            'komen "mending mati" gitu gak lucu, kasian orangnya',
            'yang nulis "mending mati" tolong dihapus deh, keterlaluan',
            'gue laporin yang komen "mending mati", itu udah kelewatan',
        ],
    },
    # ---------------- mine.py worked example ----------------------------------
    "anjir": {
        HARASS: [
            "anjir muka lu kayak gitu masih berani komen",
            "anjir sok tau {bgt} orang ini, malu maluin",
            "anjir bego {bgt} lu, dibilangin ngeyel",
            "anjir najis liat kelakuan lu",
            "anjir dasar tukang bohong, ketauan lagi",
        ],
        CRITICISM: [
            "anjir {konten} nya turun kualitasnya, sayang {bgt}",
            "anjir kok audionya pecah gini {eh}",
        ],
        AMBIG: [
            "anjir keren {bgt} \U0001F525",
            "anjir kaget gue",
            "anjir wkwkwk",
            "anjir gokil {eh}",
            "anjir gue baru tau",
            "anjir mahal {bgt} harganya",
            "anjir seru {bgt} sumpah",
            "anjir kesel gue liatnya",
        ],
    },
    "insult": {
        HARASS: [
            "tolol {bgt} lu, baca dulu baru komen",
            "dasar tolol, gitu aja gak paham",
            "bego {bgt} {eh} orang ini",
            "bangsat lu, ngerugiin orang banyak",
            "kampret emang, janji mulu gak ditepatin",
            "idiot {bgt} yang bikin aturan begini",
            "sampah {eh} kelakuan lu",
            "muka lu bikin males liatnya",
            "gak usah sok pinter deh, tolol keliatan",
            "otak udang emang lu",
        ],
        AMBIG: [
            "tolol wkwk gue juga sering gitu",
            "bego bego gini gue sayang {eh} sama lu",
            "sarap {bgt} lu wkwkwk",
        ],
        BENIGN: [
            "jangan ngatain tolol {eh}, kasar {bgt}",
        ],
    },
    "threat": {
        THREAT: [
            "gue tau alamat lu, awas aja",
            "nomor lu udah gue sebar, rasain",
            "tunggu gue datengin rumah lu",
            "gue screenshot semua data lu, siap siap",
            "awas kalo ketemu di jalan",
            "gue laporin ke kantor lu, nama lengkap udah gue pegang",
            "gue sebar foto keluarga lu kalo masih ngeyel",
            "siap siap didatengin ya",
        ],
        AMBIG: [
            "awas ya kalo bohong, gue tagih janjinya wkwk",
        ],
    },
    "sexual": {
        SEXUAL: [
            "cuma modal body doang kontennya, gak ada isinya",
            "buka baju terus biar rame ya",
            "yang dijual cuma paha, ilmunya nol",
            "pantes rame, jualan sensasi doang",
            "kalo gak pake baju ketat gak ada yang nonton kali ya",
        ],
        AMBIG: [
            "outfitnya berani {bgt} {eh}",
        ],
    },
    # ---------------- SCENE 5 -------------------------------------------------
    "ingroup": {
        AMBIG: [
            "dasar {gv} emang lu wkwk, kangen gue",
            "anjay kocak {bgt} lu bro",
            "sarap emang lu berdua wkwkwk",
            "edan {eh} lu, nekat {bgt}",
            "gila lu ya, gue kira bercanda",
            "{gv} nih temen gue, jangan diseriusin",
            "gendeng emang, tapi gue ketawa",
            "bocil kematian wkwkwk",
        ],
    },
}

#: SCENE 4. Same families, different frames. Anything not overridden here falls back to
#: the dev templates, so the held-out figure is a genuine surface shift on the phrases
#: that matter rather than a completely different corpus.
HELDOUT_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "death_wish": {
        HARASS: [
            "udah lah mending mati aja, gak ada yang nyariin",
            "buat apa idup kalo cuma bikin susah, mending mati",
            "mending mati {eh} kalo tiap hari drama terus",
            "orang model begini tuh mending mati aja beneran",
            "gue sih mending mati daripada ketauan bohong segitunya",
            "mati aja lah, capek gue liat mukanya tiap hari",
            "mendingan mati aja daripada terus terusan cari sensasi",
            "lebih baik mati daripada jadi beban orang tua",
        ],
        AMBIG: [
            "mending mati lampu daripada mati listrik sekampung wkwk",
        ],
        BENIGN: [
            'yang komen "mending mati" itu sadar gak sih itu jahat',
            'stop nulis "mending mati" ke orang, itu bukan bercandaan',
        ],
    },
    "goblok": {
        HARASS: [
            "duh {gv} {bgt} {eh} orang ini",
            "yang di atas {gv} ya, gak ngerti konteks",
            "{gv} tuh kalo ngomong dipikir dulu",
            "beneran {gv} apa cuma cari ribut",
            "kelakuan {gv} dipelihara terus",
            "jangan sok tau kalo {gv}",
        ],
        AMBIG: [
            "wkwkwk {gv} {bgt} kita ya \U0001F602",
            "{gv} nih gue kalo udah ngantuk",
            "anjay {gv} lu wkwk",
        ],
        BENIGN: [
            "ngatain {gv} tuh gak nyelesein apa apa",
        ],
        CRITICISM: [
            "aturannya {gv} {eh}, gak masuk akal buat warga",
        ],
    },
    "jelek": {
        CRITICISM: [
            "{konten} jelek {bgt} hasilnya, buram semua",
            "suaranya jelek {eh}, kayak direkam di kamar mandi",
            "{produk} jelek, gak sesuai foto sama sekali",
            "jelek {bgt} pencahayaannya, gelap terus",
            "cetakannya jelek, warnanya luntur",
            "jelek {eh} kualitas kainnya buat harga segitu",
            "gue bilang jelek karena emang hasilnya gak rapi",
            "streamingnya jelek, patah patah terus",
        ],
        HARASS: [
            "jelek {bgt} orangnya, males liat",
            "muka jelek gitu masih sok ngartis",
        ],
        AMBIG: [
            "jelek tapi ngangenin {eh}",
            "jelek jelek gitu laku keras loh",
        ],
        BENIGN: [
            "gausah bilang jelek, hargain usaha orang",
        ],
    },
    "anjir": {
        HARASS: [
            "anjir tukang tipu emang lu",
            "anjir norak {bgt} kelakuannya",
            "anjir malu maluin aja lu jadi orang",
        ],
        CRITICISM: [
            "anjir kok kualitas {konten} nya beda jauh sama yang dulu",
        ],
        AMBIG: [
            "anjir mantap {eh}",
            "anjir gue merinding",
            "anjir gak nyangka",
            "anjir wkwk parah",
        ],
    },
    "insult": {
        HARASS: [
            "tolol {eh} kalo gak ngerti jangan komen",
            "bego amat, udah dijelasin masih salah",
            "bangsat, duit orang dibawa kabur",
            "sampah {bgt} kelakuannya",
        ],
        AMBIG: [
            "bego wkwk gue juga gitu kemarin",
        ],
        BENIGN: [
            "jangan ngatain bego {eh}, gak enak dibaca",
        ],
    },
    "threat": {
        THREAT: [
            "alamat lu udah gue kantongi, tunggu aja",
            "gue sebar nomor lu ke grup",
            "siap siap gue samperin ke kantor",
            "data lu udah lengkap di gue",
        ],
        AMBIG: [
            "awas kalo gak dateng ya, gue tagih wkwk",
        ],
    },
}


# --------------------------------------------------------------------------- config
def _load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    import yaml  # pyyaml is a hard dependency; kept local so imports stay cheap

    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"config not found: {p}")
    with p.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"config {p} did not parse to a mapping")
    return cfg


def _author_hash_salt(cfg: Mapping[str, Any]) -> str:
    """R8. The salt is what makes `author_hash` non-reversible by anyone holding the
    output, so a missing one is a privacy failure, not a default to paper over."""
    salt = (cfg.get("privacy") or {}).get("author_hash_salt")
    if not salt or not str(salt).strip():
        raise SystemExit(
            "privacy.author_hash_salt is missing from the config.\n"
            "R8 requires author ids to be hashed at ingest with a salt; refusing to "
            "run without one."
        )
    return str(salt)


def _figures_from_config(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Parse the `figures:` block (build contract §1). Mirrors the shape
    `src.ingest.base.load_figures` is expected to produce, kept local so the generator
    (`generate_records`) has no import-time dependency on that module.

    R9: a figure may not have sources in both splits. `split` lives on the figure, not
    the source, so the only way to violate this is the SAME figure id appearing twice
    with two different `split` values - which is exactly what this checks for.
    """
    entries = cfg.get("figures") or []
    if not entries:
        raise SystemExit("config lists no figures under `figures:` (build contract §1)")
    seen_split: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for entry in entries:
        fid = str(entry["id"])
        split = str(entry.get("split", SPLIT_DEV))
        if split not in SPLITS:
            raise SystemExit(f"figure {fid!r} has unknown split {split!r}")
        if fid in seen_split and seen_split[fid] != split:
            raise RuntimeError(
                f"R9 violation: figure {fid!r} appears with both "
                f"split={seen_split[fid]!r} and split={split!r} - a figure must be "
                "held out (or not) across every platform, never split by source"
            )
        seen_split[fid] = split
        sources = []
        for src in entry.get("sources") or []:
            platform = str(src["platform"])
            if platform not in PLATFORMS:
                raise SystemExit(f"figure {fid!r} has unknown platform {platform!r}")
            sources.append({
                "platform": platform,
                "account": str(src.get("account", "")),
                "mode": str(src.get("mode", "import")),
            })
        if not sources:
            raise SystemExit(f"figure {fid!r} lists no sources")
        out.append({"id": fid, "genre": str(entry.get("genre", "")), "split": split, "sources": sources})
    return out


def _figure_alias_map(figures: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """figure_id -> "Figure A" / "Figure B" / ... - the pseudonym `provenance` shows on
    screen (R9). Dev figures get letters in config order, the held-out figure(s) continue
    the sequence, matching AGENTS.md's own "Figure A" / "Channel E"-style examples."""
    dev_ids = [f["id"] for f in figures if f["split"] == SPLIT_DEV]
    heldout_ids = [f["id"] for f in figures if f["split"] == SPLIT_HELDOUT]
    letters = list(string.ascii_uppercase)
    alias = {fid: f"Figure {letters[i]}" for i, fid in enumerate(dev_ids)}
    for j, fid in enumerate(heldout_ids):
        alias[fid] = f"Figure {letters[len(dev_ids) + j]}"
    return alias


# --------------------------------------------------------------------------- helpers
def _quota(total: int, mix: Mapping[str, float]) -> dict[str, int]:
    """Exact integer allocation by largest remainder. Deterministic, and it is why the
    `anjir` intent mix lands on exactly 40%/10%/50% instead of in expectation."""
    keys = list(mix)
    weight = float(sum(mix[k] for k in keys))
    if total <= 0 or weight <= 0:
        return {k: 0 for k in keys}
    raw = {k: total * mix[k] / weight for k in keys}
    out = {k: int(raw[k]) for k in keys}
    remainder = total - sum(out.values())
    order = sorted(keys, key=lambda k: (-(raw[k] - int(raw[k])), k))
    for k in order[:remainder]:
        out[k] += 1
    return out


class _VariantCycler:
    """Deterministic exact-quota sampler over the spelling variants.

    A weighted random draw would leave a rare variant like `g o b l o k` absent on some
    seeds, and scene 3 is specifically about that variant existing. A shuffled quota bag
    guarantees it appears while still looking unordered.
    """

    def __init__(self, weights: Mapping[str, float], rng: random.Random, bag: int = 100):
        counts = _quota(bag, weights)
        for k, v in counts.items():  # every listed variant must be reachable
            if v == 0:
                counts[k] = 1
        self._bag = [k for k, n in counts.items() for _ in range(n)]
        rng.shuffle(self._bag)
        self._i = 0

    def next(self) -> str:
        v = self._bag[self._i % len(self._bag)]
        self._i += 1
        return v


def _leet(rng: random.Random, word: str) -> str:
    return "".join(
        LEET[c] if c in LEET and rng.random() < 0.6 else c for c in word
    )


def _decorate(rng: random.Random, text: str, *, mutate: bool) -> str:
    """Slang, emoji, elongation, leetspeak - the surface noise a real comment section has.

    `mutate` is off for the scene-critical groups: the demo phrases carry their own
    controlled variation (the goblok family) and mangling them here would make the
    scenes depend on the seed.
    """
    if rng.random() < 0.05:
        text = rng.choice(PREFIXES) + text
    if rng.random() < 0.08:
        text = text + rng.choice(SUFFIXES)
    if mutate:
        words = text.split(" ")
        if words and rng.random() < 0.10:
            i = rng.randrange(len(words))
            if words[i] and words[i][-1].isalpha():
                words[i] = words[i] + words[i][-1] * 2   # elongation: goblokkk
            text = " ".join(words)
        words = text.split(" ")
        if words and rng.random() < 0.06:
            i = rng.randrange(len(words))
            if words[i].isalpha() and len(words[i]) > 3:
                words[i] = _leet(rng, words[i])
            text = " ".join(words)
        if rng.random() < 0.03:
            text = text.upper()
    if rng.random() < 0.14:
        text = text + " " + rng.choice(EMOJI)
    if rng.random() < 0.03:
        text = text + "!!!"
    return text


def _platform_wrap(rng: random.Random, platform: str, text: str, *, scene_critical: bool) -> str:
    """Give each platform its real textual character (R9 / D35).

    X replies open with `@handles` and stay short and clipped; Instagram comments are
    emoji-heavy with `#hashtags`; YouTube keeps the longer prose the templates already
    produce, unchanged. `scene_critical` groups skip the X clipping so the phrase
    substring a demo scene depends on (`mending mati`, `jelek`, `goblok`, `anjir`) is
    never trimmed away.
    """
    if platform == "x":
        core = text
        if not scene_critical:
            # Clip the CORE before the handle goes on, not the combined string - a demo
            # phrase never loses characters this way, and the clip is checkable by
            # stripping the leading @tokens back off in a test.
            words = core.split(" ")
            if len(words) > 5:
                core = " ".join(words[:5])
        n_handles = 1 if rng.random() < 0.75 else 2
        handles = " ".join(f"@{rng.choice(X_HANDLES)}" for _ in range(n_handles))
        text = f"{handles} {core}"
    elif platform == "instagram":
        n_tags = 1 if rng.random() < 0.55 else 2
        tags = " ".join(f"#{t}" for t in rng.sample(IG_HASHTAGS, k=n_tags))
        extra_emoji = "".join(rng.choice(EMOJI) for _ in range(rng.randint(1, 2)))
        text = f"{text} {extra_emoji} {tags}"
    # youtube: unchanged - the templates already read as the longest-form of the three
    return text


def _templates_for(split: str, group: str, intent: str) -> list[str]:
    if split == SPLIT_HELDOUT:
        override = HELDOUT_TEMPLATES.get(group, {}).get(intent)
        if override:
            return override
    return DEV_TEMPLATES[group][intent]


def _adjusted_shares(genre: str) -> dict[str, float]:
    weights = GENRE_WEIGHTS.get(genre, {})
    return {g: BASE_SHARES[g] * weights.get(g, 1.0) for g in BASE_SHARES}


def _group_mix_for(platform: str, group: str) -> Mapping[str, float]:
    override = PLATFORM_GROUP_MIX_OVERRIDE.get(platform, {}).get(group)
    return override if override is not None else GROUP_MIX[group]


def _post_url(platform: str, account: str, index: int) -> str:
    if platform == "youtube":
        return f"https://www.youtube.com/watch?v=fx_{account}_{index:02d}"
    if platform == "instagram":
        return f"https://www.instagram.com/p/fx_{account}_{index:02d}/"
    if platform == "x":
        return f"https://x.com/{account}/status/fx{index:02d}"
    raise ValueError(f"unknown platform {platform!r}")


# --------------------------------------------------------------------------- records
@dataclass(frozen=True, slots=True)
class FixtureComment:
    """One generated comment, before schema projection.

    `group` and `intent` exist for composition, statistics and tests. They are NOT part
    of the contract schema and `to_rows` does not carry them forward - see the module
    docstring. `author_id` is the synthetic RAW handle; `to_rows` hashes it (R8) and this
    class never leaves the fixture module with anything derived from it left unhashed.
    `native_id` / `parent_native_id` are platform-local ids `to_rows` turns into
    `comment_id` / `parent_id` via `base.make_comment_id`.
    """
    platform: str
    figure_id: str
    genre: str
    split: str
    source_account: str
    post_id: str
    post_url: str
    post_index: int
    post_total: int
    native_id: str
    parent_native_id: str | None
    author_id: str
    published_at: str
    text: str
    group: str
    intent: str


def generate_records(cfg: Mapping[str, Any], seed: int = 0) -> list[FixtureComment]:
    """Build the whole corpus in memory. Deterministic in `seed` and `cfg`.

    No dependency on `src.ingest.base` anywhere in this function or anything it calls -
    see the module docstring.
    """
    fx = cfg.get("fixture") or {}
    n_dev = int(fx.get("comments_per_dev_figure", 4000))
    n_heldout = int(fx.get("comments_heldout_figure", 4500))
    n_authors = int(fx.get("authors_per_split", 4200))
    n_posts = int(fx.get("posts_per_source", 14))
    span_days = int(fx.get("span_days", 150))
    reply_prob = float(fx.get("reply_probability", 0.12))
    platform_weights = {
        str(k): float(v) for k, v in (fx.get("platform_weights") or {}).items()
    } or dict(DEFAULT_PLATFORM_WEIGHTS)
    start = datetime.strptime(str(fx.get("start_date", "2026-04-01")), "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )

    rng = random.Random(seed)
    figures = _figures_from_config(cfg)
    records: list[FixtureComment] = []
    #: native-id counters are per-platform so ids stay short and dense; comment_id
    #: (built downstream from platform + native id) is what makes them globally unique.
    platform_counters = {p: 0 for p in PLATFORMS}

    for fig in figures:
        split = fig["split"]
        genre = fig["genre"]
        figure_id = fig["id"]
        total = n_heldout if split == SPLIT_HELDOUT else n_dev
        sources = fig["sources"]

        slots = HELDOUT_SLOTS if split == SPLIT_HELDOUT else DEV_SLOTS
        variants = HELDOUT_GOBLOK_VARIANTS if split == SPLIT_HELDOUT else DEV_GOBLOK_VARIANTS

        weights = {s["platform"]: platform_weights.get(s["platform"], 0.1) for s in sources}
        platform_totals = _quota(total, weights)

        for src in sources:
            platform = src["platform"]
            account = src["account"]
            p_total = platform_totals.get(platform, 0)
            if p_total <= 0:
                continue

            posts = [
                (f"fx-{platform}-{figure_id}-{i:02d}", _post_url(platform, account, i), i + 1)
                for i in range(n_posts)
            ]
            post_threads: dict[str, list[tuple[str, str]]] = {p[0]: [] for p in posts}
            post_cycle = 0

            cycler = _VariantCycler(variants, rng)
            group_quota = _quota(p_total, _adjusted_shares(genre))

            for group, g_total in group_quota.items():
                mix = _group_mix_for(platform, group)
                for intent, k in _quota(g_total, mix).items():
                    templates = _templates_for(split, group, intent)
                    mutate = group not in SCENE_CRITICAL
                    for _ in range(k):
                        post_id, post_url, post_index = posts[post_cycle % n_posts]
                        post_cycle += 1

                        values = {name: rng.choice(vals) for name, vals in slots.items()}
                        values["gv"] = cycler.next()
                        text = rng.choice(templates).format(**values)
                        text = _decorate(rng, text, mutate=mutate)
                        text = _platform_wrap(rng, platform, text, scene_critical=not mutate)

                        native_id = f"c{platform_counters[platform]:07d}"
                        platform_counters[platform] += 1

                        thread = post_threads[post_id]
                        parent_native_id: str | None = None
                        if thread and rng.random() < reply_prob:
                            parent_native_id, parent_published = rng.choice(thread)
                            parent_dt = datetime.strptime(
                                parent_published, "%Y-%m-%dT%H:%M:%SZ"
                            ).replace(tzinfo=timezone.utc)
                            published_dt = parent_dt + timedelta(seconds=rng.randrange(60, 3 * 86400))
                        else:
                            published_dt = start + timedelta(seconds=rng.randrange(span_days * 86400))
                        published_at = published_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        thread.append((native_id, published_at))

                        # R8: the synthetic handle is a plain string here and never
                        # written to disk as-is - to_rows() hashes it before any row
                        # reaches comments_raw.parquet.
                        author_id = f"{platform}-fx-{split}-{int(n_authors * rng.random() ** 2.2):06d}"

                        records.append(FixtureComment(
                            platform=platform, figure_id=figure_id, genre=genre, split=split,
                            source_account=account, post_id=post_id, post_url=post_url,
                            post_index=post_index, post_total=n_posts,
                            native_id=native_id, parent_native_id=parent_native_id,
                            author_id=author_id, published_at=published_at, text=text,
                            group=group, intent=intent,
                        ))

    # Interleave everything so comment ids/order are not a contiguous block per figure or
    # platform. A held-out leak into the dev manifest then shows up as a real bug rather
    # than being masked by an ordering that happens to sort the right way.
    rng.shuffle(records)
    return records


def to_rows(records: Sequence[FixtureComment], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project generated records onto the contract schema (build contract §1).

    Lazily imports `src.ingest.base` - the shared hashing / row-building helpers a
    different lane owns. `generate_records` above has no such dependency, so the corpus's
    composition is testable without `base.py` existing yet; this function, and `build`
    below, are the only places that need it.
    """
    from . import base  # deliberately lazy - see module + function docstrings

    salt = _author_hash_salt(cfg)
    figures = _figures_from_config(cfg)
    alias = _figure_alias_map(figures)

    rows: list[dict[str, Any]] = []
    for r in records:
        comment_id = base.make_comment_id(r.platform, r.native_id)
        parent_id = (
            base.make_comment_id(r.platform, r.parent_native_id) if r.parent_native_id else None
        )
        provenance = base.provenance_string(r.platform, alias[r.figure_id], r.post_index, r.post_total)
        row = base.build_row(
            platform=r.platform,
            figure_id=r.figure_id,
            source_account=r.source_account,
            post_id=r.post_id,
            post_url=r.post_url,
            parent_id=parent_id,
            author_id=r.author_id,
            salt=salt,
            published_at=r.published_at,
            text_original=r.text,
            split=r.split,
            genre=r.genre,
            collection_mode="fixture",
            provenance=provenance,
        )
        row["comment_id"] = comment_id
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- statistics
#: Every probe a demo scene depends on. Tests assert these are non-empty; the CLI prints
#: them so a broken generator is visible at collection time, not three stages later.
SCENE_PROBES: dict[str, tuple[str, str]] = {
    # key: (scene description, case-insensitive substring of text_original)
    "1 mending mati": ("death wish -> SUGGESTED", "mending mati"),
    "1 mati aja": ("death wish, sibling variant", "mati aja"),
    "2 jelek": ("criticism -> REJECTED", "jelek"),
    "3 goblok": ("family: plain", "goblok"),
    "3 g0blok": ("family: leetspeak", "g0blok"),
    "3 goblokkk": ("family: elongation", "goblokkk"),
    "3 g*blok": ("family: censored", "g*blok"),
    "3 g o b l o k": ("family: separated", "g o b l o k"),
    "5 wkwkwk goblok lu": ("ambiguous reel", "wkwkwk goblok lu"),
    "anjir": ("mine.py worked example, 40/10/50", "anjir"),
    'F2 condemns "mending mati"': ("no stance detection - hidden anyway",
                                   'bilang "mending mati"'),
}


def scene_report(records: Sequence[FixtureComment]) -> dict[str, Any]:
    """Counts the demo scenes depend on, per split. Fixture composition, not labels."""
    out: dict[str, Any] = {}
    for split in (SPLIT_DEV, SPLIT_HELDOUT):
        texts = [r.text.lower() for r in records if r.split == split]
        out[split] = {
            key: sum(needle in t for t in texts) for key, (_, needle) in SCENE_PROBES.items()
        }
    return out


def intent_report(records: Sequence[FixtureComment]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in records:
        out.setdefault(r.split, {}).setdefault(r.intent, 0)
        out[r.split][r.intent] += 1
    return {s: {i: counts.get(i, 0) for i in INTENT_ORDER} for s, counts in sorted(out.items())}


def group_intent_report(records: Sequence[FixtureComment], split: str = SPLIT_DEV) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in records:
        if r.split != split:
            continue
        out.setdefault(r.group, {}).setdefault(r.intent, 0)
        out[r.group][r.intent] += 1
    return dict(sorted(out.items()))


def platform_report(records: Sequence[FixtureComment]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in records:
        out.setdefault(r.split, {}).setdefault(r.platform, 0)
        out[r.split][r.platform] += 1
    return dict(sorted(out.items()))


def figure_report(records: Sequence[FixtureComment]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        out[r.figure_id] = out.get(r.figure_id, 0) + 1
    return dict(sorted(out.items()))


def reply_report(records: Sequence[FixtureComment]) -> dict[str, int]:
    return {
        s: sum(1 for r in records if r.split == s and r.parent_native_id is not None)
        for s in (SPLIT_DEV, SPLIT_HELDOUT)
    }


def goblok_platform_harmful_share(records: Sequence[FixtureComment], split: str = SPLIT_DEV) -> dict[str, float]:
    """R9: `goblok`'s harmful share by platform, in-memory intent only (not annotator
    output) - the generation-time signal the `platform_support` reporting is meant to
    surface downstream."""
    out: dict[str, float] = {}
    for platform in PLATFORMS:
        subset = [r for r in records if r.split == split and r.group == "goblok" and r.platform == platform]
        if not subset:
            continue
        harmful = sum(1 for r in subset if r.intent in HARMFUL_INTENTS)
        out[platform] = harmful / len(subset)
    return out


def _print_stats(records: Sequence[FixtureComment]) -> None:
    intents = intent_report(records)
    print("\nintended composition (fixture design, NOT annotator output - R5):")
    for split, counts in intents.items():
        total = sum(counts.values()) or 1
        harmful = sum(v for k, v in counts.items() if k in HARMFUL_INTENTS)
        print(f"  {split}: n={total}  harmful={harmful / total:.1%}")
        for intent in INTENT_ORDER:
            n = counts.get(intent, 0)
            print(f"      {intent:<22} {n:>6}  {n / total:>6.1%}")

    print("\nby platform:")
    for split, counts in platform_report(records).items():
        total = sum(counts.values()) or 1
        parts = " ".join(f"{p}={counts.get(p, 0)} ({counts.get(p, 0) / total:.1%})" for p in PLATFORMS)
        print(f"  {split:<8} {parts}")

    print("\nby figure:")
    for fid, n in figure_report(records).items():
        print(f"  {fid:<12} {n:>7}")

    print("\nreply chains (parent_id set):")
    for split, n in reply_report(records).items():
        print(f"  {split:<8} {n:>7}")

    print("\ngoblok harmful share by platform (R9: at least one must differ):")
    for platform, share in goblok_platform_harmful_share(records).items():
        print(f"  {platform:<10} {share:.1%}")

    print("\ndemo-scene phrase counts (substring of text_original):")
    scenes = scene_report(records)
    width = max(len(k) for k in SCENE_PROBES)
    print(f"  {'probe':<{width}}  {'dev':>6} {'heldout':>8}   what it is for")
    for key, (why, _) in SCENE_PROBES.items():
        print(f"  {key:<{width}}  {scenes[SPLIT_DEV][key]:>6} "
              f"{scenes[SPLIT_HELDOUT][key]:>8}   {why}")

    print("\ndev intent mix by group:")
    for group, counts in group_intent_report(records).items():
        total = sum(counts.values()) or 1
        parts = " ".join(f"{i}={counts[i] / total:.2f}" for i in INTENT_ORDER if counts.get(i))
        print(f"  {group:<12} n={total:<6} {parts}")


def _print_summary(rows: Sequence[Mapping[str, Any]], raw_path: Path, manifest_path: Path) -> None:
    per_split: dict[str, int] = {}
    per_platform: dict[str, int] = {}
    per_figure: dict[str, int] = {}
    for r in rows:
        per_split[r["split"]] = per_split.get(r["split"], 0) + 1
        per_platform[r["platform"]] = per_platform.get(r["platform"], 0) + 1
        per_figure[r["figure_id"]] = per_figure.get(r["figure_id"], 0) + 1
    print(f"\nwrote {len(rows)} comments -> {raw_path.as_posix()}")
    print(f"dev manifest (dev rows only, R3) -> {manifest_path.as_posix()}")
    print("  by split:    " + ", ".join(f"{k}={v}" for k, v in sorted(per_split.items())))
    print("  by platform: " + ", ".join(f"{k}={v}" for k, v in sorted(per_platform.items())))
    print("  by figure:   " + ", ".join(f"{k}={v}" for k, v in sorted(per_figure.items())))


# --------------------------------------------------------------------------- CLI
def build(config: str | Path = DEFAULT_CONFIG, seed: int | None = None,
          root: str | Path = ".") -> tuple[list[dict[str, Any]], list[FixtureComment]]:
    cfg = _load_config(config)
    if seed is None:
        seed = int((cfg.get("fixture") or {}).get("seed", 0))
    records = generate_records(cfg, seed=seed)
    rows = to_rows(records, cfg)

    from . import base  # lazy - see to_rows()
    import pandas as pd  # lazy - only the CLI path needs a DataFrame

    df = pd.DataFrame(rows, columns=list(base.COLUMNS))
    base.validate_frame(df)

    raw_path = Path(root) / "data/raw/comments_raw.parquet"
    base.write_corpus(rows, raw_path)
    manifest_path = Path(root) / "data/processed/dev_manifest.json"
    base.write_dev_manifest(df, cfg, manifest_path)

    print(f"fixture corpus, seed={seed}")
    _print_summary(rows, raw_path, manifest_path)
    _print_stats(records)
    return rows, records


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m src.ingest.fixture",
        description="Seeded offline Indonesian comment corpus spanning YouTube, "
                    "Instagram and X, with the same schema the real connectors write. "
                    "No API key, no network.",
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--seed", type=int, default=None,
                    help="overrides fixture.seed in the config")
    ap.add_argument("--root", default=".", help="repo root the output paths hang off")
    args = ap.parse_args(argv)
    build(args.config, args.seed, args.root)


if __name__ == "__main__":
    main()
