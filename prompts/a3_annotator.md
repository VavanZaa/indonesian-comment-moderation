# A3 annotator prompt (FROZEN — see AGENTS.md R3)

A3 is the only annotator that can vote `legitimate_criticism` (TRAP 1). If demo scene 2
looks wrong during the screen check, fix this file — not the ensemble.

Emit exactly one label, lowercase, no punctuation, no explanation:

    harassment_personal | sexual_harm | threat_privacy | legitimate_criticism |
    public_figure_invective | benign_other

Guidance:
- **`harassment_personal` is about a PRIVATE target.** Ask who is being attacked before
  asking how crude the words are. A creator, another commenter, or a named private
  person → `harassment_personal`. A politician, official, police force, court, ministry,
  government, or company, attacked in their PUBLIC role → `public_figure_invective`,
  no matter how obscene. "prabowo kntl", "pemerintah memang anjing", "ah ngentot kamu
  kejaksaan!" are `public_figure_invective`. The same words aimed at the creator or a
  fellow commenter ("kamu memang ngentod ye phey!") are `harassment_personal`.
- **`public_figure_invective` vs `legitimate_criticism` is about substance, not target.**
  Both are aimed at power and both are protected here. A comment making an argument,
  however angry ("monopoli komoditas hutan perlu dibahas", "harga TBS langsung anjlok
  setelah pidato") is `legitimate_criticism`. A comment that is only an insult with no
  claim in it ("prabowo kntl", "memang anjing rezim sekarang") is
  `public_figure_invective`. Neither is harm.
- **Profanity as an intensifier is not harassment.** Indonesian speakers use `anjing`,
  `ngentod`, `kntl` the way English speakers use "fucking" — including in ENTHUSIASM.
  "lucu banget ngentod" (so funny), "ngakak ngentod" (cracking up), "garry so asik
  ngentod" (Garry is so fun) are `benign_other`: the words are crude, the sentiment is
  positive. Look at what the sentence is doing, not which words appear in it.
- **`threat_privacy` needs an actor and an intent — naming violence is not enough.** It
  covers a threat to do harm ("besok pagi kamu saya siram air keras", "bang, kamu tidak
  takut kalau saya dateng?"), a call for others to do it ("patungan nyewa sniper untuk
  nembak pejabat", "siram balik air keras"), and exposing or hunting someone's private
  information ("kami sudah mendapatkan lokasi anda", "saya punya fotomu", "bisa
  didoxing tidak sih auditornya?"). It does NOT cover:
  - **worry for someone's safety** — "hati-hati bang, nanti disiram air keras", "saya
    takut ditarget kalian berdua", "jangan pernah keluar sendirian" are concern *for* the
    person, not a threat *against* them → `benign_other`;
  - **predictions and idioms** — "kerusuhan 98 akan terulang", "tunggu bom waktu akan
    meledak" (a ticking time bomb) → label by whatever else the comment does;
  - **curses that leave the harm to God or fate** — "semoga di laknat allah", "semoga
    jaksa2 itu hidup tidak tenang" → `public_figure_invective` if aimed at a public role,
    `harassment_personal` if at a private person;
  - **fiction and games** — anime, film, and in-game "bantai" / "bunuh" → `benign_other`.
- Criticism of a *work, product, or performance* is `legitimate_criticism`, however harsh
  — but only when the comment expresses actual dissatisfaction, disagreement, or a
  negative judgment. A neutral or positive remark about the content ("first", "kegunaan
  nya apa", a question, a compliment, an answer to the video's own question) is
  `benign_other`, even though it is "about" the work — `legitimate_criticism` requires a
  negative stance, not just topical engagement.
- `harassment_personal` requires the comment to be demeaning or hostile IN TONE toward a
  person, not merely about a person's body, intelligence, appearance, or relationships.
  Real Indonesian fan comments constantly mention a creator's looks, love life, or
  resemblance to someone else as affectionate banter, not attacks — "bucin banget"
  (teasing about being smitten), "mirip mantannya kaesang" (a resemblance comparison),
  "gas nikahin jer" (a shipping joke) are `benign_other`. Reserve `harassment_personal`
  for comments whose evident purpose is to belittle, insult, or demean the person, e.g.
  "dasar goblok lu anjing" or "tipikal orang malas mikir" said as a put-down.
- Rough joking between peers (`goblok lu wkwk`) is often `benign_other`, not harassment.
- When genuinely torn, pick the least severe applicable label. The consensus layer will
  surface the disagreement; do not resolve it by guessing severity upward.
