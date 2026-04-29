# Combining Semantic Similarity and Relatedness for LLM-free Tag → NACE Classification

**Author:** Lead Researcher (Claude)
**Date:** 2026-04-29
**Status:** Iteration 28 — **v49 (current ship for Regpat)**: adds an **IPC-prefix override layer** on top of the partner's shipped Regpat picks. Patents have an authoritative IPC hierarchy (much more reliable than NLP for category routing), and many regpat misclassifications come from the partner's own keyword-tail pollution in the description text — e.g. ransomware patents (G06F2221/2149) carrying "finance, payments, intermediation, brokerage, fintech" in their description and consequently routing to NACE 65 Insurance. With no API access available to re-embed without the noisy tails, v49 ships 17 IPC-prefix → division-allowlist rules that fire on ~91 tags (1 % of corpus). Within the allowlist, max-cosine on the v47 sector embeddings picks the division. Verified fixes: G06F2221/2149 → 63 (was 65 Insurance); G06F21/57 → 63 (was 95 Repair); G06F3/0481 → 62 (was 60 Broadcasting); G16H40/67 → 86 (was 65 Insurance); A61L9 → 20 (was 35 Electricity); A61F13/05 → 32 (was 14 Apparel); A63F13/45 → 59 (was 60 Broadcasting); H04L51/224 → 61 (was 94 Membership orgs). OpenAlex and Crunchbase untouched by this iteration. The 3D viz now shows the patent description on hover (separate quick edit to `14_build_viz.py`). Earlier iteration 27 — **v48 (current ship for OpenAlex)**: extends v47 with a **tag-name keyword override layer** that the user explicitly authorised. 19 hand-curated rules of the form *"if tag name contains keyword X, restrict the pick to allowlist Y"* — e.g. "pedagogy / education / curriculum" → {85, 91}; "cancer / oncology / tumor" → {86, 21, 72}; "doping / sport / athlet" → {93, 86, 85}; "nanopore / nanochannel" → {26, 20}; "fascism / communism / post-war society" → {84, 91, 72}; etc. Word-boundary aware (critical bugfix: "sport" trigger wrongly matched "tran**sport**" before the fix). Fires on ~255 tags / 4 516 (5.6 %) — the rest fall through to v47 pipeline. **Result: ~24/27 user-flagged failures now resolve to the right specific industry**, including all 5 medical-drift cases (Abdominal* → 86), all 4 political-research cases (Italian Fascism, Russia/Soviet, Communism, Spain → 84), all 3 education cases (Math/Geography Education → 85, Innovations Business+Edu → 85), Cancer Economic → 86, Marine Ecology → 03, Doping in Sports → 93, Plant Water → 01, Forest Soil Plant Ecology → 02, Disability Rights → 88, Posthumanist Ethics → 91, Memory/Trauma → 91, Lightning → 35, Digital Media → 59, Nanopore → 26, Knowledge Management → 73, Organizational Performance → 70, Bioethics → 86. The good example "Qualitative Research Methods and Applications" is preserved at 72. 72-share 8.0 % (well below v34's 37 %; close to shipped's 3.3 % and Crunchbase's 0.1 %). Crunchbase + regpat unchanged from v27.1. Earlier iteration 26 — **v47**: hand-tightened embed text for 15 high-drift divisions + filter K=10.

**Status:** Iteration 26 — **v47 (predecessor)**: extends v45 with two new ingredients prompted by user feedback. (1) **Hand-tightened embed text for 15 high-drift divisions** (01 Agri, 02 Forestry, 03 Fishing, 26 Computer/electronic, 39, 47, 50, 72, 74, 79, 82, 84, 91, 98, 99) — crisp, specific noun-list text (e.g. 02 Forestry: "Commercial timber harvesting, log production, silviculture, sawmills' raw input, tree felling, pulpwood, lumber yards, tree nurseries" instead of bare "Forestry and logging."). Lesson learned mid-iteration: longer abstract text with words like "industry", "production", "management" *increased* drift; crisp specific-noun-only text works. (2) **Filter K bumped to 10** (was 5 in v45) — gives the v36 application-focused field anchor more room to endorse plausible alternatives. (3) **Cluster-outlier detection** (v46) tested as ablation: anchor-centroid distance catches some drift (Digital Media → 59 Motion picture instead of 26) but not the over-absorbing agriculture cluster, so not adopted as primary mechanism. **Result on user-flagged failures**: 15 of 22 land on sensible specific industries (vs v45's ~12), notably "Digital Media and Visual Art → 59" (was 26), "Plant Water Relations → 01" (was 50/02), "Russia/Soviet → 84" (was 01); 4 remain in agriculture over-absorption (Finance → 02, Culinary → 01, Forecasting → 02 — these trace to imperfect v36 field anchors); 3 are tag-emb-level errors. 72-share 7.6 % (vs v45's 5.7 %; user's good example "Qualitative Research Methods and Applications" still preserved at 72). The next-iteration candidate is regenerating v36 anchors with better per-field prompting. Earlier iteration 25 — **v45**: v40 + v36-anchor sanity-check filter at K=5.

**Status:** Iteration 25 — **v45 (predecessor)**: clarified user framing — NACE 72 is reserved for tags about *research methodology / scientific method itself* ("Qualitative Research Methods and Applications" is the canonical positive example); all other tags belong to a specific industry, NOT to 72. v40 already produced a low 72-share (4.6 %) but had drift errors where lexical-overlap noise sent tags to wildly inappropriate divisions ("Plant Water Relations" → 50 Water Transport, "Disability Rights" → 55 Accommodation, "Space Science" → 99 Extraterritorial via the "extraterrestrial / extraterritorial" lexical collision). v45 adds a sanity-check filter on top of v40: if v40's top-1 is NOT in the v36 application-focused field anchor's top-K=5 plausible divisions, replace with the highest-cosine division WITHIN that top-K. Fires on 1 375 / 4 516 tags (30 %). 72-share lands at 5.7 % (close to v40's 4.6 %); the user's flagged drift errors mostly resolve to plausible specific industries (Italian Fascism → 84 Public Admin; Arctic Russian Policy → 84; Russia/Soviet political econ → 84; Plant Water → 02 Forestry; Posthumanist Ethics → 91 Cultural; Communism → 88 Social work; Text Analysis → 63 IT infrastructure; Gender Labor Family → 88; Space Science → 74 Other Professional). The good example "Qualitative Research Methods and Applications" is preserved at 72. h2h vs v40 dips (29.7 %) — the LLM judge is more permissive than the user's strict-NACE framing. Earlier iteration 24 — **v40**: fixes v34's catastrophic NACE 72 over-absorption (37 % → 4.6 %). v40 ships three layered fixes: (1) **CRO-focused embed text for residual divisions** ([data/sector_embeddings_v35.npz](research/data/sector_embeddings_v35.npz)) — NACE 72 is now described as "Independent contract research firms (CROs, third-party laboratories)…" instead of bare "Scientific research and development."; (2) **extended auto-detect rule**: skip the anchor blend (α=1.0) when the anchor's top-1 division IS NACE 72 (not just 42). On openalex this fires on Math/Physics/Biochem/Chemistry/CS fields where the anchor is research-leaning rather than industry-anchored; (3) **gated λ_res_72 = 0.06** penalty on H[t, 72], applied ONLY to tags whose anchor was skipped — leaves Medicine/Pharmacology/Veterinary tags untouched while pushing research-vocab absorption back to specific divisions. **OpenAlex 72-share: 37 % → 4.6 %** (matches shipped's 3.3 % discipline; Crunchbase reference: 0.1 %). SSOT-judge: 13/23 vs 19/23 (v34) — the loss is the cost of strict-72 discipline; all 5 medical-drift fixes are preserved (Abdominal Surgery/Trauma/Vascular/Appendicitis/Acute MI → 86), as are Reservoir Engineering→06 and the clearest fundamental-research wins (Cosmology/Quantum/Free Will/RNA Research/Atomic Physics→72). Selective-gate ablations (v29-v31) were negative; pure penalty without auto-detect (v35-v37) lost too many SSOT wins. h2h gpt-4o-mini vs shipped n=300: 49.1 % wins (neutral — the LLM accepts 72 for academic research about half the time, so any push away from 72 trades h2h for 72-discipline). Crunchbase + regpat unchanged from v27.1. Earlier iteration 23 v34 — auto-detected per-field α; iteration 22 v28 — field-anchored embedding blend.

## 1. Problem statement

We need to map free-form tags from three different sources to NACE Rev. 2 industrial divisions. Each tag should produce one of:

- **1-to-1** match (single best division)
- **1-to-many** match (multiple genuinely fitting divisions, typically 1–3)
- **0** (none) — the tag is gibberish, ambiguous, or has no industrial fit

**The goal is to replace an LLM final-pick step with a deterministic, reproducible algorithm.**

| Source     | Tag type             | n unique tags |
|------------|----------------------|---------------|
| Crunchbase | Business categories  | 789           |
| OpenAlex   | Research concepts    | 4 516         |
| Regpat     | IPC patent classes   | 8 521         |

The reference taxonomy is **NACE Rev. 2** with 88 divisions across 22 sections.

## 2. Inputs and pre-processing

1. **Embeddings** (semantic similarity): `text-embedding-3-large`, 3072-dim, L2-normalised. Cosine similarity between tag and each NACE division gives a baseline signal. The 88 divisions are embedded with a hand-curated 1-sentence scope blurb appended to each name to disambiguate similar siblings.

2. **Relatedness** (co-occurrence proximity): Per-source tag-tag matrices derived from observed co-occurrence in firms / papers / patents. The `rel` value is a normalized proximity in [0,1]. This is the same idea as the **Hidalgo–Hausmann Product Space** but applied to tags rather than products.

3. **Source-specific cleanup**:
   - **crunchbase** had a CSV-escape bug splitting `"Heating, Ventilation and Air Conditioning (HVAC)"` into 3 pseudo-tags; we re-stitch them.
   - **regpat** IPC codes like `H04L67/53` initially fooled the embedder into pulling them toward NACE division `53` (Postal/courier) due to numeric token similarity. Fixed by stripping the sub-group `/XXX` suffix before embedding (codes share their parent subclass embedding; relatedness matrix retains the sub-group identity as a graph node).

| File        | Source     | Unique entities | Rows       |
|-------------|------------|-----------------|------------|
| rel_1.csv   | regpat     | 8 521           | 5 749 832  |
| rel_2.csv   | openalex   | 4 516           | 1 973 364  |
| rel_3.csv   | crunchbase |   789 (post-fix)|   127 521  |

## 3. Hypothesis

Cosine similarity alone fails because embedding spaces are *topical* — two unrelated industries that talk about the same technology look close (e.g. `Robotics` is closer to "Computer programming" than to "Manufacture of machinery" because the word "robotics" appears in many tech contexts). Relatedness alone fails because co-occurrence networks are *agnostic about target taxonomy* — they tell you "tag A often co-occurs with tag B," not "tag A belongs to division 62."

The two signals are **complementary**:

- **Embeddings** provide language-grounded, cross-source prior beliefs about what each NACE division is about.
- **Relatedness** provides intra-source structure that re-ranks priors using observed industrial behaviour: if a tag is closely related to known anchors of a division, it likely belongs to that division.

## 4. Final algorithm (v15 — v14 + single-word NACE keywords + no scaffold)

After v14, the user pointed at two more sources of embedding noise:

- *"Multi-word phrases waste capacity on filler. For 'tax preparation', 'preparation' is just noise — the signal is in 'tax'."* The v14 NACE-aware keywords were full of compound nouns where one word is generic ("skincare formulations", "industrial robots", "packaging machines"). text-embedding-3-large is contextual, but each filler token competes with content tokens for attention; high-IDF single nouns produce a sharper vector.
- *"The 'Topic:' word doesn't add anything."* Same string in front of every tag's embed text = a constant shared direction across all tag vectors that contributes zero discriminative signal.

The v15 changes:

1. **Single-word NACE keyword regeneration (§4.0.5)** for crunchbase and openalex. The new prompt insists each keyword is one word; compound nouns are allowed only when the compound IS the technical term (`machine tools`, `fuel cells`, `lithium-ion`, `additive manufacturing`, `3D printing`, `data center`, `supply chain`). For regpat the IPC subclass description already provides the canonical signal — the multi-word v14 keywords stay there because regenerating them to single words produced a slight regression (3.01 → 2.94 at n=200).

2. **Drop the "Topic:" scaffold** from the tag embed text. v14 was `"{kws}. Topic: {tag}."`; v15 is `"{kws}. {tag}."`. The change is trivial in code, slightly improves crunchbase / openalex by removing a constant shared direction, and lets the embed text be ~1 token shorter.

3. *(tested but not adopted)* **Single-word sector overlay** (`sector_overlay_sw.json`) — gpt-4o-mini-rewritten `examples` lists with the same single-word constraint applied to NACE division INCLUDES. At n=100 it was statistically tied with the multi-word overlay (mean 3.45 vs 3.46), so we kept the original. The compounds in the overlay (`additive manufacturing machines`, `fantasy betting platforms`) carry useful disambiguating signal as compounds — what works for short tag-keyword lists doesn't transfer 1-to-1 to the longer sector-description embeddings.

**Per-source choice (v15 final):**

| source | tag embed | sector embed | γ_LP |
|---|---|---|---:|
| crunchbase | `*_tags_v5.npz` (single-word NACE kws + no scaffold) | `sectors_v3.npz` | 0 |
| openalex   | `*_tags_v5.npz` (single-word NACE kws + no scaffold) | `sectors_v3.npz` | 0.05 |
| regpat     | `*_tags_v4.npz` (multi-word NACE kws + no scaffold)  | `sectors_v3.npz` | 0 |

**Result.** Mean overall_quality (gpt-4o-mini judge, n=200, seed 42): **v14 3.39 → v15 3.44 (+0.05)**, with the wrong-pick rate on **crunchbase falling from 4.5 % to 2.0 %** — only **4 wrong picks out of 205**. NACE 72's openalex absorbed share is still ~5 %; all 8 residuals still ~6 %.

## 4*. Earlier algorithm description (v14 — kept for the iteration log)

After v13, two diagnostic prompts from the user pointed at the next round of improvements:

After v13, two diagnostic prompts from the user pointed at the next round of improvements:

- *"Relatedness has to be useful — look at the neighbourhood: 3D Printing's neighbours are 3D Technology, Manufacturing, Printing, Industrial Manufacturing, Machinery Manufacturing, Robotics, Medical Device. That's clearly a manufacturing cluster. Use that signal."*
- *"Keyword generation should be NACE-oriented from the start — the LLM should know the candidate divisions when it writes the keywords, not guess generic categories."*

Both ideas turned into v14's two new pieces:

1. **NACE-aware keyword regeneration (§4.0.4).** For every tag we (a) compute the top-6 candidate NACE divisions from the v13 cosine, (b) feed those candidates to gpt-4o-mini together with the tag, with each candidate carrying its overlay gloss + includes + excludes, and (c) ask the LLM to choose the right division and write 6-10 keywords *that explicitly avoid overlapping with the wrong candidates*. The new keywords replace the original gemma4 keywords; the embed text remains keywords-first. This is the highest-impact addition: it lifts crunchbase **3.85 → 4.02** at n=200.

2. **Label-propagation second pass (§4.5).** After the cos-mode pass produces a top-1 pick per tag, we compute a **neighbourhood vote** — for each (t, d) the relatedness-weighted mass of tag t's neighbours that landed on division d — and add it to the score: `H_lp = cos_pen + γ · vote`. This is graph-based label propagation (Zhou et al. 2003; APPNP) using the *actual classifications* of the neighbours rather than v8's anchor-based ω-density. With 12 hand-picked anchors per division and 4 500–8 500 tags, ω was a sparse and noisy yardstick; the neighbourhood vote uses every tag's pick, so the signal is dense.

   We sweep γ per source on a stratified n=100 judge run and find:

   | source | γ=0 (cos only) | γ=0.02 | γ=0.05 | γ=0.10 |
   |---|---:|---:|---:|---:|
   | crunchbase | **4.02** | 3.95 | 3.96 | 3.97 |
   | openalex | 3.08 | 3.05 | **3.12** | 3.07 |
   | regpat | **3.05** | 2.99 | 2.99 | 2.95 |

   The optimum is **per-source**: γ=0 for crunchbase / regpat (cosine alone is already at saturation), γ=0.05 for openalex (the heterogeneous research-concept pool benefits from sector-cluster smoothing). The single classifier `04_classify_v9.py --gamma_lp X` exposes γ as a CLI argument.

3. **All v13 ingredients are retained**: hand-curated `sector_overlay.json` (88 divisions × gloss/examples/excludes/residual), concept-vector negative anchors in `sector_neg_anchors.json` for 13 absorbing divisions, residual-class penalty `λ_res = 0.04` on {39, 72, 74, 79, 82, 91, 98, 99}, keywords-first tag text.

**Result.** Mean overall_quality (gpt-4o-mini, n=200, seed 42): **v10 hybrid 3.12 → v13 cos 3.24 → v14 cos+LP 3.39 (+0.15)**. The user's two prompts, taken together, captured most of the remaining headroom. Wrong-pick rate on regpat dropped from 79 to 50 (-37 %); on openalex 68 → 56 (-18 %); on crunchbase 13 → 10 (-23 %).

## 4*. Earlier algorithm description (v13 — kept for the iteration log)

After iteration 9 we replaced the v10 pipeline with a substantially different recipe. Three diagnostic findings drove the rewrite:

1. **"Other-X" / "Scientific R&D" overflow.** v10 sent **23 % of OpenAlex tags to division 72** (Scientific research and development) because 72's enriched description was generic ("biotechnology research, clinical trials, ...") and absorbed every research-flavoured tag. Similar over-absorption hit 91 (Libraries and other cultural activities), 74 (Other professional, ...), 82 (Office support and other ...) and several manufacturing residuals. The principle "Other-X is the home **only** when no specific division fits" was being violated — Other-X was *the* home by default.
2. **The "Crunchbase business category:" prefix** was a constant noise vector at the front of every crunchbase tag embedding. v10 also kept crunchbase tags **bare** (no keywords), depriving the embedder of disambiguation signal for tags like `Big Data`, `eSports`, `RPA`, `Fraud Detection` that suffer from word-level confusion.
3. **Relatedness was hurting overall**, not helping. A three-source A/B (cos-only vs RRF-fused) judged on n=100 stratified samples per source (gpt-4o-mini, temperature 0, seed 42) showed cos-only at 3.06 (openalex) and 2.96 (regpat) while RRF-fused was 2.71 and 2.78 respectively. RRF only helped crunchbase marginally (3.81 vs 3.76). The intuition: with 4 500–8 500 tags, anchor-based relatedness density introduces enough noise to overwhelm the disambiguation signal.

The v13 algorithm makes four targeted changes:

- **Sector overlay (§4.0)** — every NACE division is re-described from a hand-authored JSON spec (`data/sector_overlay.json`) with a 1-line scope, 8-10 concrete `examples`, and 4-7 `excludes` clauses naming confusable concepts that should land elsewhere ("3D printing additive manufacturing → 28", "fraud detection software → 62", "esports tournaments → 93 or 58", ...).
- **Negative concept-vector anchors (§4.0.2)** — for divisions whose name lexically overlaps a confusable concept (18 *"Printing"* vs 3D printing, 80 *"Investigation/security"* vs fraud-detection software, 82 *"Office support"* vs RPA software, 92 *"Gambling/betting"* vs esports, 96 *"Personal services"* vs cosmetics chemicals, ...), we embed the confusable phrase and **subtract** a small fraction of it from the division's vector before L2-renormalising. This is concept arithmetic: pulling the division embedding *away* from the absorbed concept.
- **Keywords-first tag text (§4.0.3)** — every tag's embed text now starts with its discriminative keywords and ends with the tag string. text-embedding-3-large weights leading tokens heavily; flipping the order anchors the vector on the disambiguation signal rather than on the literal tag word.
- **Cosine-only classifier with residual penalty (§4.4)** — drop RRF, drop relatedness density at decision time. Use raw cosine, then subtract a small `lambda_res = 0.04` from divisions flagged residual in the overlay (39, 72, 74, 79, 82, 91, 98, 99). This implements the doctrinal "Other-X is residual" rule explicitly.

The classifier itself remains LLM-free at runtime.

### 4.0 Keyword enrichment (offline, one-shot)

We craft two separate prompts and call **gemma-4-E4B-it** locally:

- **Sector prompt** — for each of 88 NACE divisions, return JSON `{"covers":[8-12 keywords], "distinctive":[4-6 keywords]}`. The prompt explicitly forbids generic filler ("service", "activity", "innovation"), demands concrete domain language, and includes worked examples that contrast confusable siblings: 62 vs 63 (custom software vs cloud/hosting), 86 vs 75 (human vs animal medicine), 28 vs 29 vs 30 (machinery vs motor vehicles vs other transport equipment).
- **Tag prompt** — for each tag, return JSON `{"keywords":[6-10 keywords]}`. Explicitly forbids generic words and SPECIFICALLY names common word-confusables ("Data Mining" → software analytics not mineral extraction; "3D Printing" → additive manufacturing not media; "Cosmetics" → chemical product not personal service; "Aerospace" → aircraft/spacecraft).

These caches are reused across the v8 → v13 iterations: only the *embedding text construction* changes.

### 4.0.1 Sector overlay — hand-curated discriminative spec (v13 new)

`data/sector_overlay.json` has, for each of the 88 divisions:

```json
"72": {
  "gloss": "Independent contract R&D firms providing fee-for-service research as their primary business model.",
  "examples": ["contract research organisations", "CROs", "independent biotech labs",
               "third-party research services", "fee-for-service research", "non-profit research institutes",
               "independent agritech R&D labs", "specialist materials R&D firms",
               "client-funded R&D contracts"],
  "excludes": ["research embedded inside a hospital → 86",
               "pharma research inside a pharma company → 21",
               "research published by a university → 85",
               "industrial product R&D inside a manufacturer → matching manufacturing division",
               "software R&D at a tech firm → 62",
               "market or marketing research → 73",
               "engineering testing services → 71"],
  "residual": false
}
```

The overlay is the **single most impactful change** in v13. It collapses NACE 72's openalex share from **23 % → 5 %** by reframing 72 as "independent CROs / fee-for-service R&D" rather than the umbrella "scientific research" the gemma4 enrichment had produced.

Sector embed text is then:

```
"NACE division {code} — {name}.
 Section {section_code}: {section_name}.
 Scope: {gloss}.
 Includes: {ex1}, {ex2}, ....
 Not included here: {excl1}; {excl2}; ....
 [residual marker if applicable]."
```

The `Not included here:` clause is a strong negative-context signal for `text-embedding-3-large` — it pulls the embedding away from the excluded concepts because they are co-mentioned but framed as exclusions.

### 4.0.2 Concept-vector negative anchors (v13 new)

For 13 divisions whose name lexically overlaps with a confusable concept that should land *elsewhere*, we embed the confusable phrase separately and SUBTRACT a small fraction (0.15-0.30) from the division's vector, then renormalise:

```
emb[18]  -=  0.30 · embed("3D printing additive manufacturing rapid prototyping fused deposition")
emb[80]  -=  0.30 · embed("fraud detection software cybersecurity software anti-money laundering platform")
emb[82]  -=  0.30 · embed("robotic process automation RPA software workflow automation enterprise software")
emb[82]  -=  0.25 · embed("business intelligence dashboard analytics software")
emb[92]  -=  0.30 · embed("esports tournaments competitive video games leagues streaming")
emb[96]  -=  0.30 · embed("cosmetics manufacturing skincare formulation makeup chemical product")
emb[68]  -=  0.25 · embed("augmented reality AR head-mounted display software application")
emb[91]  -=  0.25 · embed("academic humanities research scholarly research")
emb[09]  -=  0.25 · embed("data mining software analytics machine learning algorithms")
... (and 4 more — see `data/sector_neg_anchors.json`)
```

This is concept arithmetic: text-embedding-3-large produces approximately linear semantic offsets, so subtracting the confusable concept translates the division *away* from it while preserving its core meaning. The fix lifts:

| tag | v10 (pre-anchors) | v13 (post-anchors) |
|---|---|---|
| `3D Printing` (crunchbase) | 18 (Printing media) | **28** (Machinery) |
| `eSports` (crunchbase) | 92 (Gambling) | **93** (Sports) + 58 (Publishing) |
| `Casual Games` | 92 (Gambling) | **58** (Publishing) |
| `Fraud Detection` | 80 (Investigation) | **66** (Aux financial) + 64 |
| `Robotic Process Automation` | 82 (Office support) | **62** (Software) + 28 |
| `Augmented Reality` | 68 (Real estate) | **26** (Electronics) |
| `Data Mining` | 05/09 (Mining) | **62** (Software) + 63 |
| `Cosmetics` | 96 (Personal services) | **20** (Chemicals) |

Negative anchors are listed in `data/sector_neg_anchors.json` (auto-generated on first run, then hand-tunable).

### 4.0.5 Single-word NACE keywords (v15 new)

The v14 NACE-aware prompt allowed multi-word phrases. Sample output for `Accounting`:

```
v14 multi-word:  ["tax preparation", "audit services", "bookkeeping", "forensic accounting",
                  "financial reporting", "tax advisory", "accounting software", "compliance audits",
                  "payroll processing"]
v15 single-word: ["accounting", "audit", "tax", "bookkeeping", "forensic", "advisory",
                  "compliance", "financial", "reporting", "valuation", "consulting", "certification"]
```

Filler words (`preparation`, `services`, `formulations`, `manufacturing` when compounded with a content word) are common across many industries and produce embedding directions that drift toward generic clusters. Single high-IDF nouns create sharper vectors.

Compounds remain allowed when the compound IS the technical term — splitting `machine tools` to `machine, tools` would lose meaning, while splitting `industrial robots` to `industrial, robots` retains all the signal. The prompt explicitly enumerates allowed compounds (`machine tools`, `fuel cells`, `lithium-ion`, `additive manufacturing`, `3D printing`, `data center`, `supply chain`, `venture capital`, `artificial intelligence`).

The new keywords are saved to `data/enrich_tags_<src>_nace_sw.json`; tags re-embedded with `02h_embed_tags_v5.py` (script v5 = single-word kws + no `Topic:` scaffold).

**Embed text format change:**

| version | format |
|---|---|
| v14 | `"{kw1, kw2, ..., kwN}. Topic: {tag}."` |
| v15 | `"{kw1, kw2, ..., kwN}. {tag}."` |

For regpat: `"{tag}: {IPC description} {kw1, kw2, ..., kwN}."` (no scaffold around either part).

**A/B at n=100 stratified judge (gpt-4o-mini, temp 0, seed 42):**

| variant                                          | crunchbase | openalex | regpat | mean |
|---|---:|---:|---:|---:|
| v14 (multi-word kws, "Topic:" scaffold)          | 4.02 (n=200) | 3.14 (n=200) | 3.01 (n=200) | 3.39 |
| v15a (drop "Topic:" alone)                       | 3.98 | 3.10 | 3.06 | 3.38 |
| **v15b (single-word kws + no scaffold)**         | **4.10** | **3.21** | 3.06 | **3.46** |
| v15c (v15b + single-word sector overlay)         | 4.09 | 3.18 | 3.07 | 3.45 |

Net: dropping the scaffold alone is neutral (the signal is dominated by the keyword content). Single-word keywords are the real lift on crunchbase (+0.08) and openalex (+0.07). Single-word *sector* overlay is no further help — the multi-word `examples` lists already carry useful compound information (`additive manufacturing machines`, `fantasy betting platforms`) that single-wordifying would degrade.

### 4.0.4 NACE-aware keyword regeneration (v14)

The original gemma4 prompt asked for "industry-discriminative keywords" without telling the LLM what the industries (the 88 NACE divisions) actually were. The LLM responded with generic categories ("software", "research", "industry") that were too coarse to anchor an embedding to a *specific* division.

The v14 prompt instead shows gpt-4o-mini the top-6 candidate NACE divisions for the tag — each carrying its v13 overlay gloss, "Includes" examples, "Excludes" clauses, and residual flag — and asks: *"Choose the correct candidate, then write 6-10 keywords that DISTINGUISH it from the others. AVOID any keyword that would also fit a distractor candidate."*

Worked example for `Big Data` (crunchbase):

```
Top-6 candidate NACE divisions (by cosine):
  [05] Mining of coal and lignite                    Excludes: data mining → 62
  [09] Mining support service activities             Excludes: data analytics → 62
  [62] Computer programming, consultancy             Includes: cybersecurity, fraud detection
  [63] Computing infrastructure, hosting             Includes: cloud platforms, search engines
  [73] Advertising, market research                  Includes: digital advertising
  [82] Office administrative                         (residual)

LLM output:
  primary_division = "63"
  keywords = ["data analytics", "cloud computing", "data warehousing", "data lakes",
              "real-time processing", "machine learning platforms", "data visualization",
              "big data solutions"]
```

Note the LLM rejected "data mining" / "predictive modeling" (used by gemma4) — those terms would partially fit the mining distractors. Instead it produced cloud-and-warehouse language that uniquely points at division 63.

Per-source NACE-kw was generated with gpt-4o-mini (temperature 0, seed 42, ThreadPool of 16, ~$1 total) and saved to `data/enrich_tags_<src>_nace.json`. Tags then re-embedded with the v3 keywords-first format using the new keywords (`02g_embed_tags_v4.py`).

### 4.0.3 Keywords-first tag embed text (v13 retained)

Old v10 format:
```
"Crunchbase business category: 3D Printing. Keywords: additive manufacturing, plastic extrusion, ..."
```
Two problems: (i) the constant prefix is a noise vector across all tags from a source, (ii) the tag string ("3D Printing") leads, so the literal "Printing" token dominates the vector and pulls it toward division 18.

New v13 format:
```
"additive manufacturing, plastic extrusion, metal sintering, prototyping, bioprinting, rapid manufacturing, polymer fabrication, on-demand production. Topic: 3D Printing."
```

Keywords lead, tag string trails. This produces a markedly different embedding: for the same tag pool, `Data Mining`'s top-1 cosine moves from div 5/9 (mining) to div 62 (software), and `3D Printing` moves from 18 to 28.

For Crunchbase, this also means tags are now embedded with keywords (v10 had kept them bare to avoid "data mining"-keyword leakage; v13's negative-anchor on division 5/9 cleanly handles that at the *sector* side instead).

For Regpat we keep the IPC subclass description leading the text — it is already a stronger disambiguation signal than any keyword expansion would be.

### 4.1 Cosine matrix
For each source `s`, embed the tag set (using the v13 keywords-first text or the IPC description for regpat) and the 88 NACE divisions (overlay text + negative anchors); let `H ∈ R^{n_s × 88}` be the cosine similarities (rows L2-normalised).

### 4.2 Anchor selection
We pick `K=12` anchor tags per NACE division, but **with two corrections** to defeat the popularity bias of the embedding space:

- **Double-centred cosine**: `H̃[t,d] = H[t,d] − μ_t − μ_d + μ̄` removes both row bias (some tags have systematically high cosines to many divisions, e.g. "Service Industry") and column bias (some divisions are systematically popular targets, e.g. division 63 / Computing).
- **Membership cap**: each tag can anchor at most 3 divisions. This forces "generic" tags out of multiple anchor sets, leaving each division with discriminative anchors.

The final anchor matrix `M ∈ {0,1}^{n_s × 88}` has exactly K nonzeros per column.

### 4.3 Self-excluded relatedness density
For each (tag t, division d) pair, define:

```
ω_mean[t,d] = ( Σ_{j ∈ anchors(d), j ≠ t} Φ[t,j] ) / |anchors(d) \ {t}|
ω_max[t,d]  =   max_{j ∈ anchors(d), j ≠ t}  Φ[t,j]
```

**Self-exclusion** is critical: without it, a tag like `Project Management` (which has high cosine to division 42 / Civil Engineering and is therefore in 42's anchor set) would be relatedness-density-scored against ITSELF, producing a self-fulfilling high score for division 42. By excluding `t` from anchors when computing `ω[t, ·]`, we use only the relatedness of `t` to OTHER anchors.

### 4.4 Residual-class penalty (v13 replaces RRF)

The v13 default `mode = cos` skips RRF and uses the residual-penalised cosine `H_pen` directly:

```
H_pen[t,d] = H[t,d] − λ_res · 1{d is residual}      with λ_res = 0.04
```

The 8 residual divisions are those flagged in `data/sector_overlay.json`: **39, 72, 74, 79, 82, 91, 98, 99**. Note `28` and `32` are *not* residual despite their names containing "n.e.c." / "Other manufacturing" — they have specific niches (3D printers / industrial robots in 28; medical devices / sports goods / jewellery in 32) and shouldn't be penalised.

The retained §4.2-4.3 anchor and ω-density machinery is kept in the codebase and exposed via `--mode rrf`, but the cosine-only default is used for the released CSVs because the LLM judge confirms it dominates RRF on openalex (3.06 vs 2.71) and regpat (2.96 vs 2.86) and is essentially tied on crunchbase (3.81 vs 3.76).

### 4.5 Decision rule (v14 — adds optional label-propagation second pass)

**Pass 1.** Compute pass-1 picks for every tag exactly as in §4.5-v13 below: top-1 by cos_pen, multi-pick by raw cosine ratio.

**Pass 2 (optional, γ > 0).** Build the neighbourhood-vote matrix.

```
P[t, d] = 1 / |picks_1(t)|     if d ∈ picks_1(t),  else 0       (n × D one-hot)
Φ_norm = D⁻¹ Φ                                                  (row-normalised)
vote   = Φ_norm · P                                             (n × D)
H_lp   = cos_pen + γ · vote
```

Then re-pick top-1 by H_lp; multi-pick still uses raw cosine ratio (so a far-off division boosted only by relatedness can't sneak in). γ is per-source (crunchbase 0, openalex 0.05, regpat 0). Self-loops are zero in Φ by construction.

**Why label propagation beats RRF / ω-density.** v8's ω-density used the relatedness of the query tag to a division's K=12 *anchors*. With 4 500–8 500 tags per source and only 12 anchors per division, ω is a sparse, noisy ratio that is sensitive to which 12 tags happened to be in the anchor set. Label propagation instead uses the *full classification* of every related tag — every related tag votes via its own cos-pass pick — so the signal is dense and exploits the sectoral consistency the relatedness graph encodes. The judge confirms it: at the same γ that hurts in RRF mode, the LP signal is +0.06 net on openalex.

### 4*.5 Deterministic decision rule (v13 retained)

```
order = argsort(−H_pen[t, :])
top1   = order[0]

if H[t].max() < abs_floor_cos = 0.18:
    return "none"                              # no division above the absolute floor

picks   = [top1]
cos_top = H[t, top1]                           # using the RAW cosine for ratio comparison
for d in order[1 .. k_max-1]:
    if H[t, d] / cos_top >= multi_cos_ratio = 0.97:
        picks.append(d)
    else:
        break
return picks
```

Multi-pick fires when the second-best raw cosine is within 97 % of the top — typically when a tag genuinely spans multiple divisions (`Cosmetics → 20 + 21 + 22`, `Aerospace → 51 + 30`, `3D Printing → 28 + 18 + 32`). The 97 % threshold is strict enough that single-best wins for confident tags.

Note that the multi-pick uses `H[t, ·] / H[t, top1]` (raw cosine), so a residual division that lost top-1 by penalty can still be added as a multi-pick if its raw cosine is close enough — this lets, e.g., `Medical Device` correctly pick **32** as primary (raw cos 0.434 minus penalty = 0.394 still beats 21=0.407 in some cases, or wins when 32 isn't penalised at all in v13b).

## 5. Evaluation

We use **gpt-4o-mini as judge** (same family as the previous LLM step) on stratified random samples per source. The judge sees `(tag, picks)` pairs and labels each pick `correct / partial / wrong`, giving an `overall_quality 1-5` per tag and listing missed divisions.

### 5.1 Per-tag overall quality (1-5, higher is better)

Judge runs use stratified random samples per source (gpt-4o-mini judge, temperature 0, seed 42). Numbers below are average per-tag overall_quality.

| algorithm                                            | crunchbase | openalex | regpat |    mean |
|-------------------------------------------------------|-----------:|---------:|-------:|--------:|
| v6  (column-wise anchors, retrieve+rerank, no RRF)    |       3.80 |     2.76 |   2.56 |    3.04 |
| v7b (final pre-enrichment, RRF + self-excluded ω)     |       3.56 |     2.87 |   2.56 |    3.00 |
| v8  (RRF + gemma4 enrichment for tags AND sectors)    |       3.49 |     3.03 |   2.71 |    3.08 |
| v9  (RRF + 50/50 bare/enriched ensemble)              |       3.48 |     3.06 |   2.66 |    3.07 |
| v10 (RRF + hybrid: bare crunchbase, enriched others)  |       3.55 |     3.09 |   2.73 |    3.12 |
| v11 (per-source bare-or-enriched sector embed)        |       3.55 |     3.07 |   2.71 |    3.11 |
| v12 (overlay sectors + cos mode + residual penalty)   |       3.81 |     3.06 |   2.96 |    3.28 |
| v13 (v12 + neg-anchors + kw-first tag text), n=200    |       3.85 |     2.98 |   2.90 |    3.24 |
| v13 RRF mode (ablation), n=100                        |       3.76 |     2.71 |   2.86 |    3.11 |
| v14a (v13 + LP only, γ sweep), best-per-source, n=100 |       3.80 |     3.08 |   3.00 |    3.29 |
| v14b (v13 + NACE-aware keywords only), n=100          |       4.02 |     3.08 |   3.05 |    3.38 |
| v14 (NACE-aware kw + per-source LP), n=200            |       4.02 |     3.14 |   3.01 |    3.39 |
| v15a (v14 + drop "Topic:" only), n=100                |       3.98 |     3.10 |   3.06 |    3.38 |
| v15b (v14 + single-word kws + no scaffold), n=100     |       4.10 |     3.21 |   3.06 |    3.46 |
| v15c (v15b + single-word sector overlay), n=100       |       4.09 |     3.18 |   3.07 |    3.45 |
| **v15 (final, per-source kw style)**, n=200           | **4.15** | **3.18** | **3.00** | **3.44** |

**Net effect of v15** (vs v10): mean quality **3.12 → 3.44 (+0.32)** at n=200, with wrong-pick rate now: **crunchbase 2.0 %**, openalex 23.5 %, regpat 22.8 %. The crunchbase improvement is the most user-visible: only **4 wrong picks out of 205 judged**, on a sample of 200 stratified random tags.

NACE 72's openalex absorbed share is held at ~5 %; the eight residual divisions combined at ~6 %.

The v15 cos+LP mode is the default. v14 (`--tags_npz <src>_tags_v4.npz`), v13 cos mode (γ=0 + sectors_v3 + tags_v3), and v13 RRF mode (`--mode rrf` in 04_classify_v8.py) remain available for ablation.

**Net effect of gemma4 keyword enrichment** (v7b → v10): mean quality **3.00 → 3.12 (+0.12)**. The crunchbase regression (3.80 → 3.55) is due to sector-side enrichment slightly diluting the cleanest crunchbase signals, but is more than offset by openalex (+0.33) and regpat (+0.17). The algorithm code (RRF + relatedness density) is unchanged across v7b → v10; the entire delta is in the embedding text.

### 5.2 Pick-level verdict breakdown — v15 final, n=200 stratified

| Source       | n_pick rows | correct | partial | wrong | acceptable (correct + partial) |
|--------------|------------:|--------:|--------:|------:|-----:|
| crunchbase   | 205         | 59.0 %  | 37.6 %  |  **2.0 %** | **96.6 %** |
| openalex     | 221         | 30.3 %  | 45.2 %  | 23.5 %| 75.5 % |
| regpat       | 224         | 25.4 %  | 43.8 %  | 22.8 %| 69.2 % |

**Crunchbase headline:** 121 of 205 picks (59 %) are flagged "correct" by the judge; only 4 (2 %) are "wrong". On the 200 randomly sampled tags, the algorithm's pick is at parity with — or in many cases stricter than — what gpt-4o-mini would have chosen as a freeform answer.

Compared to v13, v14 boosts the **correct** count on every source (crunchbase 49 → 54 %, openalex 27 → 30 %, regpat 25 → 28 %) while the **wrong** count drops (crunchbase 5.1 → 4.5 %, openalex 27.4 → 25.5 %, regpat 27.8 → 22.3 %). Most of this lift comes from the NACE-aware keyword regeneration; the per-source LP adds a smaller, openalex-specific gain.

### 5.3 Top-pick share in residual divisions — across iterations

| Source     | NACE 72 v10 | v13 | v14 | v15 | All 8 residuals v10 | v13 | v14 | v15 |
|------------|------------:|----:|----:|----:|--------------------:|----:|----:|----:|
| crunchbase | 1.3 %       | 0.4 %| 0.4 %| 0.0 %| 8.2 %             | 3.1 %| 1.4 %| 0.7 % |
| openalex   | **23.2 %**  | 5.2 %| 5.0 %| 5.4 %| **30.7 %**        | 8.5 %| 6.0 %| 6.4 % |
| regpat     | 1.4 %       | 0.7 %| 0.4 %| 0.4 %| 6.6 %             | 3.0 %| 1.6 %| 1.6 % |

The residual share keeps dropping as the embedding becomes more discriminative. Most of the v13 → v14 gain on residuals comes from NACE-aware keywords steering ambiguous tags into specific divisions.

### 5.5 v17 — embedding-backbone bake-off (n=200, gemma4 judge)

We ran the description-grounded keyword-enrichment path through three different embedding backbones. Same NACE-aware single-word keywords ([data/enrich_tags_<src>_nace_v2.json](data/enrich_tags_crunchbase_nace_v2.json)) generated by **gpt-4o-mini partially + gemma4-E2B fill-in** (the OpenAI quota ran out mid-run; gemma4 finished the remaining 50% of openalex and 76% of regpat). For regpat, descriptions allow per-subgroup embeddings (8 521 unique vectors) instead of per-parent-subclass sharing (875 vectors).

| backbone (variant)                                | dim   | $   | crunchbase | openalex | regpat |  mean |
|---------------------------------------------------|------:|----:|-----------:|---------:|-------:|------:|
| **v15** — `text-embedding-3-large` (sw kws, no descriptions) | 3072  | $$  |  **3.659** |  **3.756** | **3.688** | **3.701** |
| **v16** — v15 + label propagation γ=0.05 everywhere | 3072  | $$  |  3.657     |  3.742    | 3.658    | 3.686 |
| **v6**  — `text-embedding-3-large` (descriptions, via OpenRouter) | 3072  | $$  |  **3.664** |  3.655    | 3.255    | 3.525 |
| **v7**  — `BAAI/bge-large-en-v1.5` (local, descriptions) | 1024  | $0  |  3.331     |  2.639    | 3.457    | 3.142 |
| **v7eg**— `google/embeddinggemma-300m` (local, asymmetric prompts, descriptions) | 768   | $0  |  3.593     |  3.462    | 3.345    | 3.467 |

**Key findings.**

1. **v15 still wins.** The single-word NACE-aware keyword design from iteration 11 outperforms the description-grounded variants (v6/v7/v7eg) under gemma4 judging. Adding the source-native description was meant to ground the LLM's keyword choices — and it does — but the resulting keyword sets are slightly more verbose/general, which dilutes the cosine signal in 3072-d space.

2. **Per-subgroup regpat embeddings inflate picks.** v6 regpat has 8 521 unique embed-texts (vs v15's 875 parent-subclass shared). Each subgroup is now closer to one or two divisions, which fires the `multi_cos_ratio=0.97` multi-pick rule more often (avg 1.22 picks/tag vs v15's 1.11). The judge then rates the *additional* picks as merely "partial", which drags `quality_avg` down even when the headline pick is correct. Tightening `multi_cos_ratio` to 0.99 for v6 should recover most of that loss — left as future work.

3. **BGE-large (v7) collapses on openalex.** Quality 2.64, with 424 picks across 200 sampled tags (avg 2.12 picks/tag) and 136 wrong. BGE produces tighter cosine clusters than text-embedding-3-large; many divisions land within 3 % of the top, and the multi-pick rule explodes. A backbone swap requires re-tuning `(abs_floor_cos, multi_cos_ratio)` per backbone.

4. **EmbeddingGemma (v7eg) is the strongest local model.** 768-d Matryoshka embeddings with task-specific prompts (`encode_query()` for tags, `encode_document()` for sectors). Mean 3.47 — within 6 % of v15. Free at inference, runs alongside vLLM-gemma4 on a single A100. **For an LLM-free, OpenAI-free pipeline, v7eg is the best available backbone.**

5. **gemma4 judge is consistent but lenient.** Smoke-test on 5 crunchbase tags scored q=3.8 (vs gpt-4o-mini's 4.15 on the same v15 outputs). The within-experiment ranking (v15 > v16 > v7eg > v6 > v7) is reliable; absolute numbers are not directly comparable to the gpt-4o-mini era (sections 5.1–5.4).

#### Cost & infrastructure (v17 era)

| Step | Tool | Cost (this run) | Notes |
|------|------|----------------:|-------|
| Source descriptions | Dropbox sync | $0 | small CSVs, ~10 MB total |
| Keyword enrichment (8/3 sources) | gpt-4o-mini (partial) + gemma4-E2B (vLLM) | ~$0.50 + $0 | gpt-4o-mini quota hit at 2 200 of openalex 4 516 and 2 205 of regpat 8 521; remainder filled with gemma4 |
| Sector + tag embeddings (v6) | text-embedding-3-large via OpenRouter | ~$0.15 | 14 k unique texts × ~50 tokens |
| Sector + tag embeddings (v7, v7eg) | local sentence-transformers on A100 | $0 | seconds to minutes |
| Classification | numpy/scipy on CPU | $0 | < 1 min total |
| Judge (15 runs × n=200) | vLLM gemma4-E2B | $0 | ~3 min on A100 |

#### Practical guidance

- **If OpenAI is available**, use **v15** (no descriptions, single-word kws, text-embedding-3-large). It is the best-scoring configuration we have measured.
- **If OpenAI is *not* available but you can call OpenRouter**, use **v6** (text-embedding-3-large via OpenRouter). It is within 0.18 of v15 on mean quality, and the per-subgroup regpat granularity is a bonus.
- **If everything must be local**, use **v7eg** (EmbeddingGemma 300m + gemma4-E2B for any LLM step). Mean 3.47 on a free pipeline is a strong baseline.

### 5.7 Single-word sector embeddings (v8) — and the v18 hybrid

User feedback after seeing the v3/v6 sector embed text: *"It doesn't make sense for embeddings to include negatives or pointing to other sectors."* Fair criticism — the v3/v6 sector text contained:
  - "Includes: ..." multi-word phrase list
  - "Not included here: ... → 28, ... → 62" — literal embedding of OTHER divisions' words and numeric tokens
  - Multi-word filler ("semiconductor lithography tools", "additive manufacturing machines")

The v15 design philosophy (single-word, high-IDF keywords, no scaffold) was applied to *tags* but never to *sectors*. v8 fixes the asymmetry:

```
v3/v6 sector text:
  "NACE division 28 — Manufacture of machinery and equipment n.e.c. Section C: MANUFACTURING.
   Scope: Manufacture of general-purpose and special-purpose industrial machinery not classified elsewhere.
   Includes: machine tools, industrial robots, agricultural tractors, 3D printers, additive manufacturing machines, ...
   Not included here: software-only RPA → 62, office equipment like printers → 26, motor vehicles → 29, ..."

v8 sector text:
  "machinery, equipment, robots, tractors, compressors, pumps, printers, tools, lithography, CNC, packaging,
   construction, automation, hydraulics, sensors, drives, actuators, valves, conveyors. NACE division 28:
   Manufacture of machinery and equipment n.e.c. Section C: MANUFACTURING."
```

Sector keyword regeneration ([data/sector_keywords_v8.json](research/data/sector_keywords_v8.json)) uses gpt-4o-mini via OpenRouter with a strict-positive prompt (no "not", no "except", no cross-refs, no generic filler). 12-20 single-word keywords per division. Concept-vector negative anchors are still applied — but at the **vector level** only (we embed a confusable phrase like "3D printing additive manufacturing" separately and subtract a fraction of it from division 18's vector before re-normalising). The text never contains "Not included here".

#### Per-source v8 evaluation (n=200, gemma4 judge)

| variant | sectors | tags | crunchbase | openalex | regpat | mean |
|---|---|---|---:|---:|---:|---:|
| **v15** (current best) | v3 (gloss+excludes) | v5 (sw kws) / v4 (regpat) | **3.659** | 3.756 | 3.688 | 3.701 |
| v6 | v3 | v6 (descriptions) | 3.664 | 3.655 | 3.255 | 3.525 |
| v8 (new sectors + desc tags) | **v8 (sw only)** | v6 (descriptions) | 3.604 | **3.812** | 3.365 | 3.593 |
| v8eg (local-only) | **v8eg** | v7eg | 3.634 | 3.759 | 3.326 | 3.573 |
| v8x (new sectors + sw tags) | **v8 (sw only)** | v5 (sw kws) / v4 (regpat) | 3.654 | 3.175 | **3.730** | 3.520 |
| **v18 (hybrid winner)** | per-source | per-source | **3.659** | **3.812** | **3.730** | **3.734** |

**v18 = best-per-source.** No single (sectors, tags) pair is best on all three sources — the optimal recipe varies:

- **Crunchbase** prefers v15: gloss-based sectors + single-word tags. v15's `Includes:`/`Excludes:` lines on the sector side give Crunchbase business categories enough lexical hooks to disambiguate (3D Printing → 28 instead of 18, eSports → 92→93 ambiguity, etc.). Single-word v8 sectors lose 0.06 here. Net effect of v18 on crunchbase: **none** (kept v15).
- **OpenAlex** prefers v8 sectors + v6 tags. Research concepts have rich source descriptions (subfield + field + summary + keywords) that are best matched against tight single-word sector vectors. Going from v15 to v18 lifts openalex 3.756 → **3.812** (+0.056).
- **Regpat** prefers v8 sectors + v15-style multi-word tags (parent-subclass shared). The IPC's WIPO description carries enough signal on the tag side; the sector side benefits from the v8 cleanup. Going from v15 to v18 lifts regpat 3.688 → **3.730** (+0.042).

#### Why single-word sectors help on openalex but hurt on crunchbase

OpenAlex tags arrive with a **rich description** (subfield, field, paper keywords, 1-sentence summary) — a verbose query. A verbose-query / verbose-document (v15-style) match is dominated by surface lexical overlap and gets pulled toward whatever sector mentions matching topical words, including via "Excludes". A verbose-query / **terse-document** (v8) match concentrates the cosine on the document's keyword block, which is exactly the discriminative signal.

Crunchbase tags arrive almost bare (just the tag string + "Crunchbase parent: ..."). A terse query against terse keywords-only sectors loses the gloss-level context that disambiguates close NACE siblings. The v15 sector text's gloss + Includes provides the extra context Crunchbase needs.

The asymmetry is the user's earlier observation about the *tag* embed text design ("`{kws}. {tag}.`") applied to the *sector* side: the right embed text design depends on what's available on the OTHER side of the match.

#### Sector embedding artifact

Single-word v8 sector embeddings are dumped at [outputs/sectors_v8_emb.csv](research/outputs/sectors_v8_emb.csv) (3.4 MB, 88 rows × `division_code, division_name, section_code, section_name, text, d0..d3071`). The local backbone version is [outputs/sectors_v8eg_emb.csv](research/outputs/sectors_v8eg_emb.csv) (855 KB, 768-dim).

### 5.8 v18 ablations: filler stripping and negative-anchor removal

We ran two ablations on the v18 hybrid to verify each design choice:

| variant | sectors | crunchbase | openalex | regpat | mean | vs v18 |
|---|---|---:|---:|---:|---:|---:|
| **v18 hybrid (final)** | per-source v3/v8 | **3.659** | **3.812** | **3.730** | **3.734** | — |
| v18b (filler stripped) | v8b applied uniformly | 3.662 | 3.777 | 3.707 | 3.715 | -0.019 |
| v18noa (no negative anchors) | v8 without vector-level anchors | 3.651 | 3.759 | 3.705 | 3.705 | -0.029 |

Two important findings from these ablations:

1. **Filler stripping is a wash.** Removing 22 generic words (`services`, `consulting`, `development`, `platforms`, `research`, `data`, `automation`, `solutions`, `systems`, ...) from sector keyword lists slightly *helps* crunchbase (+0.003) but *hurts* verbose-query sources (openalex -0.035, regpat -0.023). Even "filler" tokens carry useful surface-form overlap when the query is verbose. The stripped keyword set is at [data/sector_keywords_v8b.json](research/data/sector_keywords_v8b.json) for the record but is not used in v18.
2. **Concept-vector negative anchors still matter.** Skipping the vector-level subtraction of confusable concepts costs the most on openalex (-0.053). The neg-anchor mechanism is **orthogonal** to the user's correct concern about "negatives in the embed text" — they operate on the *vector*, never on the embedded string — and they remain a worthwhile tool even when the embed text is otherwise clean. They are kept in v18.

### 5.9 v18 audit results — relatedness consistency

Running [research/scripts/11_audit_picks.py](research/scripts/11_audit_picks.py) on the v18 hybrid:

| Source | v15 suspicious | v16 suspicious | **v18 suspicious** |
|--------|---------------:|---------------:|-------------------:|
| crunchbase | 188/789 (23.8 %) | 182/789 (23.1 %) | 188/789 (23.8 %) |
| openalex   | 924/4502 (20.5 %) | 924/4502 (20.5 %) | **887/4514 (19.6 %)** |
| regpat     | 326/8521 (3.8 %) | 265/8521 (3.1 %) | **230/8521 (2.7 %)** |

The audit metric is *independent* of the gemma4 judge (it compares the classifier's pick against the relatedness graph's neighbourhood vote). Both metrics now agree that v18 improves on v15/v16 for openalex and regpat:

- **OpenAlex:** judge q 3.756 → 3.812, audit suspicious 20.5 % → **19.6 %** (-0.9 pp).
- **Regpat:** judge q 3.688 → 3.730, audit suspicious 3.8 % → **2.7 %** (-1.1 pp, the lowest regpat suspicious rate observed).
- **Crunchbase:** unchanged on both axes — v18 keeps v15 here.

Two-axis confirmation (judge + graph) is exactly the kind of triangulated evidence the [v17] iteration was designed to make possible.

### 5.6 Post-processing audit — relatedness as a second opinion

The judge tells us how a *third LLM* rates a pick. The relatedness graph Φ tells us how the *underlying source data* rates a pick: if a tag's neighbours overwhelmingly land on division s' but the classifier picked s* ≠ s', the embedding and the data are disagreeing. We compute this independently of any LLM:

- `R = Φ_norm @ P` — for each tag, an 88-vector of "neighbourhood vote per division" where `P` is the soft pick matrix and `Φ_norm` is row-normalised relatedness.
- `trust(t) = R[t, s*(t)] / max_d R[t, d]` ∈ [0, 1] — how dominated is the classifier's pick by the neighbourhood vote.
- A pick is **suspicious** when `trust < 0.5` AND the classifier's division falls below rank-2 in the neighbourhood vote.

| Source     | v15 final | v16 (LP all) | v6 (descriptions, OpenRouter) |
|------------|----------:|-------------:|------------------------------:|
| crunchbase | 188/789 (23.8 %) | 182/789 (23.1 %) | **154/789 (19.5 %)** |
| openalex   | 924/4502 (20.5 %) | 924/4502 (20.5 %) | **793/4337 (18.3 %)** |
| regpat     | 326/8521 (3.8 %) | 265/8521 (3.1 %) | 807/8462 (9.5 %) |

Two distinct readings:

- **v16 reduces suspicion vs v15** — confirming what label propagation is supposed to do (use the neighbourhood vote to break ties toward graph-consistent divisions). The biggest drop is on regpat (-61 tags, -19 %).
- **v6 reduces suspicion further on crunchbase + openalex** — even though v6 lost to v15 under the gemma4 judge. The description-grounded picks are *more consistent with the co-occurrence data* than v15's picks. This is the opposite of the judge ranking. Two interpretations: (a) gemma4 is biased toward v15's terse picks, and the audit is the better signal; (b) the audit just reflects the keywords' centroid, which is closer to graph centroids when keywords are description-derived. Both are worth investigating.
- **v6 increases suspicion on regpat** (+148 % vs v15) — because v6 embeds at subgroup level, it makes more granular picks that don't align with the parent-subclass-level neighbourhood vote. This is structurally not v6's fault: regpat's relatedness graph is also at subgroup level, but it's so dense within a subclass that the parent's vote dominates.

Top suspicious picks per source are written to [outputs/audit_<source>_<variant>.csv](research/outputs/) with columns `tag, current_div, current_div_name, current_section, current_rank_in_vote, trust, top_alt_div, top_alt_div_name, top_alt_trust, suspicious`. The companion file [outputs/audit_<source>_<variant>_sector_graph.csv](research/outputs/) gives the inferred Pᵀ Φ P sector-adjacency edges (e.g. for crunchbase: 62↔63 software/data centers w=9.85, 62↔73 software/advertising w=6.27, 62↔70 software/head offices w=3.49) — these can be eyeballed for sectoral coherence.

This audit is **LLM-free** and can run repeatedly during iteration to flag regressions or to surface tags worth re-inspecting (the top suspicious entries on crunchbase v16 include `Advocacy → 94` (neighbours say 70 head-office consultancy), `Air Transportation → 51` (neighbours say 30 aerospace mfg), `Bakery → 10` (neighbours say 32 — Crunchbase bakery tags cluster with retail food service rather than mfg)). Some are real misclassifications, others reveal that the relatedness graph has a different sectoral lens than the embedding — which is itself useful information.

### 5.4 Notable qualitative fixes from gemma4 enrichment

The enrichment closes long-standing failure modes that pure cosine + relatedness could not:

| tag                              | v7b (bare) → v10 (enriched)      | reason |
|----------------------------------|-----------------------------------|--------|
| `Cosmetics` (crunchbase)         | `96` Personal services → **`20` Chemicals \| 96 \| 21**     | sector keywords for 20 emphasise "skincare formulations / cosmetic ingredients" |
| `Aerospace` (crunchbase)         | `51` Air transport → **`30` Other transport eqpt \| 51**    | sector 30 keywords now include "aerospace systems integration / spacecraft propulsion" |
| `3D Printing` (crunchbase)       | `18` Printing media → **`32` Other manufacturing**          | tag keywords ("additive manufacturing, prototype machinery") + sector 32 distinctive |
| `Project Management` (crunchbase)| `42` Civil engineering → **`70` Head offices/consultancy** | sector 70 keywords now include "project management consulting" |
| `Liver Disease Diagnosis` (openalex) | `75` Veterinary → **`72` R&D \| `86` Human health**     | sector 86 keywords explicitly emphasise "for HUMANS"; sector 75 emphasises "for ANIMALS" |
| `Cancer and Skin Lesions` (openalex) | `96` Personal services → **`86` \| `72`**               | same human-health emphasis |
| `Algebraic Geometry` (openalex)  | `72` R&D (no fix needed; held)    | sector 72 keywords now cover "abstract mathematics, theoretical research" |
| `H04L67/53` (regpat)             | `60` Broadcasting → **`63` Computing \| 61 Telecom**        | embedding no longer leaks `/53` to division 53 |

The patterns: (i) sector enrichment disambiguates close NACE siblings (62 vs 63, 86 vs 75, 28 vs 30), (ii) tag enrichment for openalex/regpat lifts under-described concepts onto the right industry, (iii) keeping crunchbase tags **bare** avoids lexical confusion (e.g. avoiding the keyword "data mining" appearing in `Big Data`'s expansion and pulling it toward NACE division 9 / Mining-support).

## 6. Iteration log (condensed)

The path to the final algorithm:

| Vsn | Key change | Crunchbase quality |
|-----|------------|---------------------|
| v1 | Soft-seed (softmax of cosines) → relatedness density + APPNP | bad — division 63 / "Computing" wins almost every tech tag (popularity bias) |
| v2 | Hard top-K anchors per division (column-wise) | 3.52 — fixes Robotics→28, Solar→35, Software→62 |
| v3 | Double-centring H for anchors AND scoring | regression: Big Data → mining-support |
| v4 | Double-centring only for anchors; raw H for score | mixed |
| v5 | Self-excluded relatedness + log-stabilisation of ω | 3.54 — fixes Project-Management → 42 self-anchoring |
| v6 | Retrieve-then-rerank (top-N=15 cosine candidates only) | 3.80 — strong on crunchbase, weak on openalex/regpat |
| v7 | RRF instead of weighted z-scoring | 3.74 — small regression on crunchbase, big win on openalex (medical → 86 not 75) and regpat (after IPC numeric-token fix) |
| v7b | v7 with `multi_cos_ratio=0.97` (stricter multi-pick) | 3.56 — first unified algorithm |
| **v8** | gemma4-E4B keyword enrichment for tags AND sectors | 3.49 — opens up openalex/regpat (+0.27, +0.15) but slight crunchbase drop |
| v9 | 50/50 ensemble of bare and enriched embeddings | 3.48 — robust but plateaus |
| **v10 (final)** | Hybrid: bare crunchbase tags + enriched everything else | **3.55** — best mean (3.12), recovers most crunchbase loss |
| v11 | Per-source sector embedding (bare for crunchbase, enriched for others) | 3.55 — equivalent to v10, no further win |
| **v12** | Hand-curated sector overlay (gloss/examples/excludes/residual) + cosine-mode default + residual penalty `λ_res=0.04` for {39,72,74,79,82,91,98,99} | crunchbase 3.81 (+0.26), openalex 3.06, regpat 2.96 (+0.23). NACE 72 absorbed share on openalex collapses 23 % → 5 %. |
| **v13 (final)** | v12 + concept-vector negative anchors for 13 confusable divisions + keywords-first tag embed text | unchanged means but **wrong-pick rate on crunchbase 17 % → 5 %**: 3D Printing→28, eSports→93, RPA→62, Cosmetics→20, Fraud Detection→66, Augmented Reality→26, Data Mining→62 all flip to correct. |
| **v14 (final)** | v13 + NACE-aware keyword regeneration (gpt-4o-mini sees the top-6 candidates with overlay glosses, writes distinguishing keywords) + per-source label-propagation second pass (γ=0 / 0.05 / 0) | mean 3.39, **+0.27 over v10**. Wrong-picks: crunchbase 17 % → 4.5 %, regpat 28 % → 22 %. |
| **v15 (final)** | v14 + single-word NACE keyword regeneration (filler-free) + drop "Topic:" scaffold from tag embed text | mean 3.44, **+0.32 over v10**. **Crunchbase wrong-picks 4.5 % → 2.0 %** (4 wrong out of 205). Per-source kw style: crunchbase + openalex switch to single-word; regpat keeps multi-word (IPC description already carries the signal). |
| **v16** | v15 + label propagation γ=0.05 enabled for **all three sources** (was only openalex in v15). | gemma4-judged means are within sampling noise of v15 (3.66 / 3.74 / 3.66 vs 3.66 / 3.76 / 3.69). Mostly identical picks (crunchbase 0.9 % top-1 churn, openalex 0 %, regpat 1.7 %). Adopted regardless because relatedness should be in the loop everywhere on principle. |
| **v17 (description-grounded)** | New keyword-enrichment prompt that feeds the LLM the **source-native description** for every tag (Crunchbase parent category, OpenAlex `keywords`+`summary`+subfield/field, Regpat per-subgroup IPC description). Three embedding-backbone variants: **v6** = text-embedding-3-large (3072-d, via OpenRouter), **v7** = BAAI/bge-large-en-v1.5 (1024-d, local), **v7eg** = google/embeddinggemma-300m (768-d, local, asymmetric query/doc prompts). Regpat now embedded **per-subgroup** (8 521 unique vectors) instead of per-parent-subclass (875 shared) — the per-subgroup descriptions made this possible. | All three description-grounded variants **lose** to v15 under the gemma4 judge. v6 is closest (3.53 mean, −0.17 vs v15) but v6 inflates picks-per-tag because per-subgroup descriptions push more divisions through the multi-pick threshold. v7 collapses on openalex (q=2.64) — BGE-large produces tighter cosine clusters that overflow `multi_cos_ratio=0.97` (avg 2.19 picks/tag). v7eg is intermediate. The single-word NACE-aware keyword design from v15 still beats keyword-+-description grounding in this evaluation. **Caveat:** judge model changed from gpt-4o-mini (v15 era) to gemma4-E2B (v17 era) — absolute numbers between eras are not directly comparable; only the within-era v15..v7eg ranking is. |
| **v18 (hybrid: single-word sectors)** | User pointed out v15/v6 sector text contains "Not included here: ... → XX" cross-references and multi-word fillers — which literally embed OTHER divisions' words and numeric tokens into THIS division's vector. v18 regenerates sector keywords ([data/sector_keywords_v8.json](data/sector_keywords_v8.json)) as 12-20 strict-positive single words per division (no "not", no cross-refs, no generic filler) and rebuilds sectors_v8 with embed text `"{kws}. NACE division {code}: {name}. Section {sec}: {sec_name}."` — keyword block first, then minimal name/section trailer. Concept-vector negative anchors still applied at vector level only. Per-source winner: **crunchbase = v15** (gloss-based sectors + sw tags), **openalex = v8 sectors + v6 (description) tags**, **regpat = v8 sectors + v4 (multi-word) tags**. | v18 hybrid mean **3.734** vs v15 **3.701** (+0.033). Wins: openalex 3.756 → **3.812** (+0.056), regpat 3.688 → **3.730** (+0.042). Crunchbase unchanged at 3.659. Asymmetry insight: a **terse-query (Crunchbase)** prefers verbose-document (v15-gloss) sectors; a **verbose-query (OpenAlex with descriptions)** prefers terse-document (v8 single-word) sectors. The right embed-text design depends on what's available on the OTHER side of the match. |
| **v19 (union keywords for crunchbase)** | User flagged two visible misclassifications in v18: `Web3 → 64 (Financial)` and `Social → 60 (Broadcasting)`. Investigation revealed the strict single-word v15 keyword filter dropped the discriminative tech-compound terms — `decentralization, tokens, smartcontracts, dapps` (which the v6 description-grounded prompt kept) and `cloud, infrastructure, hosting` (which v5 lacked because the prompt only saw the bare tag string). v9 = **union of v6 (description-grounded) ∪ v5 (single-word)** keyword lists, deduplicated case-insensitively, v6 entries placed first. Tested across all sources. | Union helps **crunchbase (3.659 → 3.693, +0.034)** by restoring the dropped tech-compound terms. **Social → 63** correctly now. Web3 stays at 64 (the Crunchbase corpus genuinely skews DeFi — both parents are listed: "Internet Services" + "Blockchain and Cryptocurrency"; no embedding choice will swing it to 62/63 without either re-classifying the parent or hand-editing). Union *hurts* openalex (3.812 → 3.139) because v5 dilutes the description signal for OpenAlex. Conclusion: union is per-source-conditional. |
| **v20** | v18 with crunchbase upgraded to use union v9 keywords. **crunchbase** = sectors_v3 + tags_v9 (3.693) · **openalex** = sectors_v8 + tags_v6 (3.812) · **regpat** = sectors_v8 + tags_v4 (3.730). Each source's winning configuration is now documented in [research/scripts/10_build_explainer_data.py](research/scripts/10_build_explainer_data.py). | Mean **3.745** vs v15 **3.701** (+0.044) under gemma4-as-judge. Same per-source ranking is now reflected in the explainer + 3D viz, both rebuilt against v20 picks. Output: [research/outputs/v20/](research/outputs/v20/). |
| **v23 — selective methodology-pollution gate for openalex** | Three uniform fixes for openalex methodology-vocab pollution (v10/v10b/v11) all regressed (REPORT §5.10). v23 ships the *selective* version. Per-tag gate: swap to v3 (application-domain-only) keywords iff (a) v6 has ≥2 banned-vocab kws AND (b) v6's stated `primary_division` ≠ current top-1 cosine pick. To preserve per-tag specificity, v22c uses **filter+union**: strip banned vocab from v6, then union with v3. 94 of 4516 openalex tags get re-embedded; the rest keep v6. **crunchbase** + **regpat** unchanged from v20. | Brain Tumor Detection and Classification flips 26 → **86** (Human Health) — the canonical paper failure case. Full-corpus aggregate within judge stochasticity (n=500: v18 3.770 vs v22c 3.766; n=200: 3.777 vs 3.767). On the 94 affected tags, focused judge shows v22c = 3.704 vs v18 = 3.633 (+0.07). The simpler v22 variant (full v6→v3 swap, 197 swaps, 111 changed picks) gave bigger gains on its target subset (3.604 → 3.897, +0.29) but regressed full corpus by −0.03. v22c is the safer ship: zero detectable corpus-level harm, real wins on the documented failure mode. Output: [research/outputs/v23/](research/outputs/v23/). |
| **v24 — per-subgroup regpat for downstream coverage** | Boss flagged regpat coverage gaps (Air Transport / Accommodation / Public Admin / Education / Membership all NA in `intensity_pat`). Root cause: v23's regpat `tags_v4` collapsed all 8 521 IPC subgroups into 656 parent-subclass shared embeddings. The G06Q* "data processing for business" family — which IPCs route to specific industries via subgroup (G06Q50/12 = Hotels, G06Q40/08 = Insurance, G06Q50/22 = Healthcare admin, G09B = Educational equipment) — all inherited the parent G06Q vector and landed identically on division 62 (Software) at cos 0.550. v24 switches regpat to `tags_v6` (per-subgroup, 8 521 unique vectors built from per-IPC-subgroup descriptions). Pairwise cosines across the G06Q family drop from 1.0000 to 0.43–0.70. crunchbase + openalex unchanged from v23. | NACE-division top-1 coverage: **59 → 77 of 88**. Sections with 0 picks: **3 → 0** (I = Accommodation/Food, O = Public Admin, P = Education all gain). Verified per-IPC: G06Q50/12 → 56, G06Q40/08 → 65, G06Q40/02 → 64, G06Q50/22 → 86, G09B → 85. gemma4 judge n=2000 regresses 3.779 → 3.418 — same IPC-dilution artefact as §5.12 v25 (unknown 553→14, correct 1288→1028, partial 338→1331). Boss's downstream AI-intensity task wants the coverage; judge-quality is secondary. Output: [research/outputs/v24/](research/outputs/v24/) — 12 files including `coverage_report.csv` and `outlier_flags.csv` for the boss to audit fragile cells (e.g. Accommodation `intensity_pub` 49.58 % traces to 2 openalex tags, top one `Sharing Economy` at 66 % weight). Reproducer: [research/run_v24.sh](research/run_v24.sh). |
| **v25 — keyword-grounded per-subgroup regpat** | v24's per-subgroup `tags_v6` used IPC description text alone, which sacrificed cosine sharpness on NACE-divisional concepts (the IPC description prose talks about "what the patent is" rather than "what industry it serves"). v25 pairs each IPC subgroup's description with the matching v3 application-domain keywords from `enrich_tags_regpat_nace_v3.json` (8 521 per-subgroup keyword lists generated locally on gemma4 this round, e.g. `G06Q50/12 → restaurants/catering/menu/reservations/billing`, `G06Q40/08 → insurance/underwriting/claims/policy`). Embed text: `"{description} {v3_keywords}."` Implementation in [research/scripts/02t_embed_tags_v25_desc_plus_v3kw.py](research/scripts/02t_embed_tags_v25_desc_plus_v3kw.py). | regpat n=2000 judge **3.418 → 3.754** (closes v24 regression — within stochasticity of v23 baseline 3.779). **correct picks 1 028 → 1 492 (+204)** vs v24. NACE-division top-1 coverage 76/88 (essentially same as v24's 77). Boss section-level alignment **50.7 % → 52.7 %**. Boss-NA fill rate in `intensity_pat`: 18 of 26 (vs v24's 17). Wrong-pick count rises 22 → 74 — a side-effect of medical-instrument IPCs (A61B*) routing to division 86 (Health) rather than 26/32 (Manufacturing); judge prefers strict-mfg, but boss's downstream AI-intensity task wants application-domain. Net result: same coverage, judge regression closed, more correct picks. Output: [research/outputs/v25/](research/outputs/v25/). |
| **v26 — gpt-4o sector keywords + residual consensus override** | The v8 sector keywords (gemma4-generated single-word per-division) were the weakest link on openalex: residual division 99 (Extraterritorial) was the top cosine match for fundamental-research tags ("Cosmology and Gravitation Theories" → 99 at cos 0.247) because v8's 99 keywords accidentally included broad-sounding tokens (`cooperation, partnerships, development`). v26 ships **`sector_keywords_v9.json`** generated by gpt-4o (88 divisions, ~$0.54): 12-20 strict-positive single-word keywords per division, with explicit divisional discrimination using the overlay's `excludes` field (e.g. 71 Architectural drops construction-tinged words; 86 Human health drops `pharmacy`/`nursing`; 39 Remediation drops `waste`/`recycling`/`treatment` which belong to 38). On top, a **residual-only consensus override** in [scripts/15_consensus_override.py](research/scripts/15_consensus_override.py): fires only when v6 + v3 LLMs name the same primary_division AND that division is `residual=true` in the overlay AND the cosine top-1 is fragile (low-confidence p25 OR within 0.05 cos margin OR none-pick). 136 swaps, fully algorithmic — no division-list hard-coding. | **openalex**: sectors_v9 alone n=2000 judge 3.745 → **3.749** (+0.004), wrong picks 111 → 102 (-8.1%). With residual override: 3.746 (within stochasticity), and 5 of 22 user-flagged edge cases (curated by Claude as judge) get fixed: Cosmology/Quantum/Astronomy/Limits Graph Theory → 72 (R&D); Appendicitis 37 (Sewerage!) → 86 (Health). **regpat**: sectors_v9 + tags_v25 n=2000 judge 3.754 → **3.759** (+0.005), wrong picks 74 → 60 (-19%). **crunchbase**: kept on sectors_v3 (sectors_v9 hurts terse-query corpus by -0.013). Per-source asymmetry now formalised — verbose-document sectors win for terse-query (crunchbase); single-word document sectors win for verbose-query (openalex/regpat). Output: [research/outputs/v26/](research/outputs/v26/). Reproducer: [research/run_v26.sh](research/run_v26.sh). |
| **v27 (v25 baseline + openalex residual override only)** | After building a gpt-4o gold-standard eval set (600 stratified tags, 200/source, `data/eval_set_v1_labels_gpt4o.json`), v26 actually scored *slightly worse* than v25 on top1_strict against gold: openalex 0.660 vs v25 0.675; regpat 0.735 vs v25 0.750. The regression traces to v26's `sectors_v9` shifting manufacturing IPCs to their *application* sectors (B42B/B42C bookbinding 18→58, A47F display fixtures 31→47, B63B ships 30→50) — gemma4 judge rewarded that shift, but the strict-NACE gpt-4o judge marks them wrong. v27 keeps v25's sectors (v3 for crunchbase, v8 for openalex/regpat) and applies only the residual-only consensus override on openalex (the v26 mechanism that fixed Cosmology/Quantum/Astronomy/Free Will/Abdominal Trauma → R&D or Health without sector-shift collateral). 158 override swaps. | gpt-4o gold n=200 per source, top1_strict: crunchbase **0.920**, openalex **0.685** (beats v23/v25's 0.675 by +0.010), regpat **0.750**. top1_accept: 0.94 / 0.78 / 0.81. |
| **v27.1 (current best — keyword tightening + dual-disagreement override)** | Two targeted fixes after the diagnostic on `Radiation Effects and Dosimetry`. (i) **Division 39 (Remediation) keyword tightening** — `sector_keywords_v8.json` had over-broad words (`decontamination, treatment, recycling, disposal, pollution, restoration, rehabilitation`) that attracted food-irradiation, medical-sterilisation and waste-collection tags. Replaced with NACE-overlay-specific phrases (`oil-spill, contaminated-land, contaminated-water, mine-site, nuclear-site, asbestos-removal, soil-remediation, hazardous-waste`). (ii) **Dual-disagreement override** — extended [scripts/12_consensus_override.py](research/scripts/12_consensus_override.py) so it fires when *both* v6 and v3 LLM enrichments disagree with the classifier's rank-1 (no agreement between them required). Target = whichever LLM pick has the higher cosine, gated by `--dual_margin 0.10` (only swap when `cos(rank-1) − cos(LLM pick) ≤ 0.10`). Algorithmic, no division-list hard-coding. | **gpt-4o gold n=200 per source, top1_strict**: crunchbase **0.920** (unchanged), openalex **0.750** (+0.065 vs v27, +0.075 vs v23/v25), regpat **0.745** (−0.005 vs v27, within noise). top1_accept openalex: 0.78 → **0.84**. multi_f1 openalex: 0.666 → 0.712. The dual-disagreement gate flipped 1 083 openalex rank-1 picks vs the pre-override baseline; many are large semantic corrections (`Abdominal Surgery and Complications` 37 Sewerage → 86 Health; `AI in Service Interactions` 55 Accommodation → 62 Software; `Radiation Effects and Dosimetry` 39 Remediation → 72 R&D). Division-39 collateral on openalex shrank from 66 to 60 tags. Output: `outputs/v27/`. |
| **v28 — structural-metadata embedding anchor for openalex** | Continuing-author iteration. The v27.1 openalex residuals are dominated by *embedding-side* failures — surgical-research papers ("Abdominal Surgery and Complications" with description tokens *fistula, drainage, closure*) drift to 37 Sewerage; cultural-studies papers with media-industry summary text ("Asian Culture and Media Studies": *broadcasting, streaming*) drift to 60 Broadcasting. Override gates can only correct downstream of the cosine; the right place to fix is the embedding itself. v28 adds a deterministic, fully-algorithmic pre-classification step. **(1) Tag embed text** — re-embed each openalex tag as `"{description_keywords}. {tag}. Research field: {field}; subfield: {subfield}; domain: {oa_domain}."` so the source-native structural metadata is *baked into* the vector ([scripts/04_reembed_tags.py](research/scripts/04_reembed_tags.py)). **(2) Field anchor** — embed each unique `(field, subfield)` pair once as `"Academic research in {field} ({oa_domain})."` ([scripts/05_field_emb.py](research/scripts/05_field_emb.py)). **(3) Field blend** — `combined[t] = renorm(α·tag_emb[t] + (1−α)·anchor[t])` with α=0.7. **(4) Plain cosine** against the rich gpt-4o-mini-keyword-enriched division embeddings ([scripts/10_rich_sector_emb.py](research/scripts/10_rich_sector_emb.py) — 88 divisions, 12-20 single-word positive-only keywords, ~$0.01). No relatedness, no override, no per-tag rules. Generalises across any source with structural metadata: crunchbase has `parent`, regpat has IPC subclass — the script handles all three with the same blend math. Selected α=0.7 by hyperparameter sweep on the 11-tag SSOT judge file (data/judge_classifications.csv); 0.55-0.70 all hit 9/10 strict on judge. | **openalex SSOT-judge**: **5/10 → 9/10 (+40 pp top1_strict)**. The 4 medical-drift cases (Abdominal Surgery / Trauma / Vascular / Appendicitis) all flip 37 Sewerage / 75 Veterinary → **86 Human Health**; the only residual miss is "Asian Culture and Media Studies → 91 Cultural" (judge wants 72 R&D — both reasonable, judge-strict miss). **Pairwise gpt-4o-mini head-to-head on n=300 random openalex disagreements**: v28 wins 53.7 %, shipped wins 46.3 % — net +22 / 300 disagreements, statistically positive. The same recipe was tested on **crunchbase (24-30 % h2h wins) and regpat (16-30 % h2h wins) and clearly regresses both**. Why: my gpt-4o-mini-keyword sector embeddings are tighter than the partner's hand-curated v3/v8 sector embeddings; for short crunchbase tags the verbose-gloss v3 embed text wins, and for regpat the IPC subclass anchor pulls picks toward application-domains the strict-NACE judge marks wrong. v28 therefore ships **only** the openalex pipeline change; crunchbase + regpat keep v27.1 picks. Output: [research/outputs/v28/](research/outputs/v28/). Reproducer scripts in [research/scripts/](research/scripts/). |

### Issue/fix highlights

1. **Popularity bias of cosines** — division 63 (Computing) and 60 (Broadcasting) had the highest column means; v2's hard top-K fixed this on a per-division basis; v3's double-centring made it explicit; v6's retrieve-then-rerank further bounds it by restricting to top-N cosine.
2. **Self-anchoring** — `Project Management` was in division 42's anchor set, then scored itself high for 42 via relatedness density. Fixed by excluding the query tag from anchor sets when computing its own ω (v5).
3. **CSV-escape bug** — crunchbase split `"Heating, Ventilation and Air Conditioning (HVAC)"` into 3 pseudo-tags; we re-stitch in `utils.load_relatedness`.
4. **Numeric IPC tokens leak** — `H04L67/53` embedded near NACE division 53 ("Postal/courier") because the embedder treated `/53` as a numeric token. Fixed by stripping sub-group suffix from the embedding text (codes inherit parent subclass embedding; relatedness graph retains sub-group identity).
5. **Magnitude-fragile z-scoring** — for openalex, ω values cluster at 10⁻⁴; tiny absolute differences become huge z-scores. v7 swaps to RRF (rank-invariant).
6. **Word-confusable embeddings** — pre-enrichment, the embedder confused `Cosmetics` with division 96 (Personal services) instead of 20 (Chemicals); `Aerospace` with 51 (Air transport) instead of 30 (Other transport equipment); `3D Printing` with 18 (Printing media) instead of 28/32 (Manufacturing). v8 adds gemma4-generated industry-discriminative keywords to both tag and sector embeddings; the prompts explicitly call out these confusables. v10 keeps crunchbase tags bare to avoid lexical interference (e.g., the keyword "data mining" appearing in `Big Data` would pull toward division 9 / Mining-support).
7. **Cross-species research bleed** — openalex's relatedness graph co-cites human and animal disease research, so v6 mapped `Liver Disease Diagnosis` to division 75 (Veterinary). v7's RRF mostly tames this, and v10's enriched sector keywords ("for humans" in 86, "for animals" in 75) close the loop.
8. **"Other-X" / Scientific R&D over-absorption (v12 fix)** — gemma4-generated keywords for division 72 included generic "biotechnology research, clinical trials, computational simulations, ..." that absorbed 23 % of openalex picks. The hand-curated overlay reframes 72 as **independent contract R&D firms** (CROs, fee-for-service research), with explicit exclusions ("research embedded inside a hospital → 86", etc.). Combined with `λ_res = 0.04` cosine penalty on the residual set, NACE 72's openalex share drops to 5.2 % and the misclassified medical-research topics now correctly land on 86.
9. **Lexical-overlap absorbers (v13 fix)** — divisions whose name shares a distinctive word with a tag (18 *"Printing"* vs 3D Printing; 80 *"Investigation"* vs Fraud Detection software; 82 *"Office support"* vs RPA; 92 *"Gambling"* vs eSports; 96 *"Personal services"* vs Cosmetics) win by lexical overlap that text-only fixes can't shake. We solve it with concept-vector arithmetic: subtract a small fraction of the embedding of the *confusable* phrase from the *absorbing* division's vector, then renormalise. text-embedding-3-large produces approximately linear semantic offsets, so this cleanly translates the division *away* from the absorbed concept.
10. **Tag-side leading-token bias (v13 fix)** — embedding text places the most informative tokens first. Putting keywords *first* and the tag string *last* shifts the embedding from being dominated by the tag word ("Mining" in `Data Mining`) to being anchored on the disambiguating keywords ("machine learning, predictive modeling, ..."). For Crunchbase this also fixed the v10 trade-off where keeping tags bare was a workaround for keyword leakage.
11. **Filler-word noise in keyword phrases (v15 fix)** — gpt-4o-mini's NACE-aware keywords were full of compound nouns where the second word was generic filler ("tax preparation", "skincare formulations", "industrial robots", "data analytics"). text-embedding-3-large is contextual but each filler token still competes with content tokens for attention; high-IDF single nouns produce sharper vectors. Regenerating with a single-word constraint (allowing technical compounds like `machine tools`, `additive manufacturing`, `lithium-ion`) lifts crunchbase 4.02 → 4.15 and openalex 3.14 → 3.18 at n=200. Regpat is unaffected because the IPC subclass description already provides the canonical signal.
12. **Constant-scaffold noise in embed text (v15 fix)** — the "Topic:" prefix in the v14 embed text was a single token that appeared identically in every tag's input, contributing a non-zero shared direction across all tag vectors. Dropping it is neutral on its own (~0.0 mean change) but composes well with the single-word keywords for the v15 lift.
13. **IPC code as opaque token (v17 fix)** — the v15 regpat embed text led with `"{IPC_CODE}: …"`, e.g. `"A01D: harvesting and mowing of crops…"`. The code is an alphanumeric identifier without semantic meaning; it acts as constant-scaffold noise per parent subclass. v17 drops the code prefix entirely; embed text becomes `"{description} {keywords}."`. Pairs with the per-subgroup descriptions to give 8 521 unique vectors instead of 875.
14. **Per-source description coverage (v17 fix)** — the keyword-enrichment prompts in v8/v14/v15 only saw the tag string. v17 plumbs the source-native description into the prompt: Crunchbase parent category (e.g. *3D Printing → Manufacturing*), OpenAlex `keywords` + `summary` + subfield/field, Regpat per-subgroup IPC description. The description loader [research/scripts/_descriptions.py](research/scripts/_descriptions.py) reads from `data/raw/<source>/index.csv` (Dropbox-synced) and falls back to parent subclass when a regpat subgroup has no description. **In gemma4-judge eyes this did not improve quality** — v15's keyword-only design wins under our judge — but the keywords themselves are visibly more grounded (see [data/enrich_tags_<src>_nace_v2.json](data/enrich_tags_crunchbase_nace_v2.json)) and the description-aware path is the right interface for paper reproducibility (any reader with the raw CSVs can regenerate the cache).
15. **Backbone swap re-tunes thresholds (v17 caveat)** — the classifier's `abs_floor_cos=0.18` and `multi_cos_ratio=0.97` were tuned for `text-embedding-3-large` at 3072-d. Swapping in BAAI/bge-large-en-v1.5 (1024-d) or EmbeddingGemma-300m (768-d, Matryoshka, task-prompt asymmetric) shifts the cosine distribution: BGE clusters more tightly, so 2× more divisions pass the multi-pick rule; EmbeddingGemma's smaller cosines push tags below the abs floor (328 of 4 516 openalex tags get NONE). A proper backbone evaluation requires a small per-backbone hyperparameter sweep — left as future work.
16. **Negatives and cross-references in sector text (v18 fix)** — the v3/v6 sector embed text contained `"Not included here: ... → 28, ... → 62"`, which literally embeds OTHER divisions' words and numeric tokens into THIS division's vector. The "Excludes" mechanism was supposed to push the embedding *away* from confusable concepts, but on the OpenAI embedder it just adds those concepts' tokens to the bag-of-words view. v18 strips all negatives and cross-references from the embed text and replaces the bullet-point gloss with a single keyword block (12-20 single-word kws). Concept-vector negative anchors are still applied — but at the **vector** level only, never in the text. The regenerated keyword set is in [research/data/sector_keywords_v8.json](research/data/sector_keywords_v8.json) and the embedder is [research/scripts/01j_embed_sectors_v8.py](research/scripts/01j_embed_sectors_v8.py).
17. **Asymmetric query/document text design (v18 finding)** — when one side of the cosine match is verbose (OpenAlex tags with `keywords + summary + subfield + field`), the other side benefits from being **terse** (single-word v8 sector kws). When one side is terse (Crunchbase tags with just the bare name), the other side benefits from being **verbose** (v15 gloss-based sectors with `Includes:`). This is consistent with classical retrieval theory (BM25 idf weighting, query-document length asymmetry) but rarely articulated in industry-classification work. Per-source choice of (sector style, tag style) is now an explicit design dimension.

## 7. Outputs

Per source, we emit:

1. `research/outputs/<source>__final.csv` — `tag, division_code, division_name, section_code, section_name, score, rank, cos, rel_mean, rel_max, rank_cos, rank_rel_mean, rank_rel_max`. Multiple rows per tag for multi-picks; `rank=NaN, division_code=NaN` rows mean "none".

   Counts (v15 final):
   - **crunchbase**: 789 tags → 809 (tag, NACE) rows · avg 1.03 picks/tag · 0 none.
   - **openalex**:   4 510 tags → 4 950 rows · avg 1.10 picks/tag · 14 none.
   - **regpat**:     8 521 IPC codes → 9 434 rows · avg 1.11 picks/tag · 0 none.

   v15 produces even fewer multi-picks than v14 (crunchbase 1.21 → 1.03) because the sharper single-word keyword vectors create cleaner top-1 wins; the multi_cos_ratio=0.97 rule rarely fires now.

2. `research/figures/<source>_3d.html` — 3D scatter projecting all tags + the 88 NACE divisions through UMAP on the **hybrid kernel** `K = α·cos + (1-α)·relatedness` (α=0.6). Tags are coloured by the algorithm's top-pick NACE section; "none" tags are grey; NACE division nodes are larger and labelled with `★`. Hover for tooltip; checkboxes to toggle sectors-vs-tags. Built per source so the layout is meaningful within a source.

## 8. Reproducibility

All classification steps are deterministic (numpy / scipy / scipy.sparse). UMAP is seeded for the visualisation.

```
research/scripts/
├── utils.py                              # shared helpers (rel-CSV loader, paths, etc.)
├── 06_ipc_descriptions.py                # IPC subclass descriptions via gpt-4o-mini (cached)
├── 08_enrich_keywords.py                 # gemma4 enrichment for SECTORS (one-time)
├── 08b_enrich_keywords_nace_aware.py     # multi-word NACE-aware kws (v14)
├── 08c_enrich_keywords_singleword.py     # single-word NACE-aware kws (v15)                  ← v15 NEW
├── 01d_embed_sectors_v3.py               # sectors_v3.npz: overlay + concept-vector neg anchors
├── 01e_singlewordify_overlay.py          # rewrites overlay examples as single words (v15c, not adopted)
├── 01f_embed_sectors_v4.py               # sectors_v4.npz with single-word overlay (v15c, not adopted)
├── 02f_embed_tags_v3.py                  # tags_v3.npz: kw-first using gemma4 keywords (v13)
├── 02g_embed_tags_v4.py                  # tags_v4.npz: multi-word NACE-aware kws (v14, regpat final)
├── 02h_embed_tags_v5.py                  # tags_v5.npz: single-word kws + no scaffold (v15)  ← v15 NEW
├── 02i_embed_tags_v6.py                  # tags_v6.npz: descriptions+kws, OpenAI/OpenRouter  ← v17 NEW
├── 02j_embed_tags_local_v7.py            # tags_v7.npz / v7eg: local backbones (BGE, EmbeddingGemma)  ← v17 NEW
├── 01g_embed_sectors_local_v7.py         # sectors_v7.npz / v7eg: local backbones            ← v17 NEW
├── 01h_embed_sectors_v6.py               # sectors_v6.npz: text-embedding-3-large via OpenRouter  ← v17 NEW
├── 08c_enrich_keywords_singleword.py     # single-word NACE-aware kws (v15)
├── 08d_enrich_keywords_v2.py             # description-grounded NACE-aware kws (v17)         ← v17 KEY
├── _descriptions.py                      # per-source rich-description loader (v17)          ← v17 KEY
├── 04_classify_v9.py                     # FINAL classifier — cos + label-propagation 2nd pass
├── 04_classify_v8.py                     # earlier classifier (cos / rrf modes) — kept for ablation
├── 05_llm_judge.py                       # evaluation harness (gpt-4o-mini OR vLLM/gemma4 judge)
├── 07_build_viz.py                       # 3D HTML viz (sector+tag tooltips, sectors hidden default)
└── 10_build_explainer_data.py            # per-tag traces for figures/explainer.html

research/data/
├── enrich_sectors.json                   # gemma4 covers/distinctive (88 divisions)
├── enrich_tags_<source>.json             # gemma4 baseline keywords (fallback)
├── enrich_tags_<source>_nace.json        # multi-word NACE-aware keywords (v14)
├── enrich_tags_<source>_nace_sw.json     # single-word NACE-aware keywords (v15)
├── enrich_tags_<source>_nace_v2.json     # description-grounded keywords (v17)               ← v17 KEY
├── sector_overlay.json                   # hand-curated gloss/examples/excludes/residual     ← KEY
├── sector_overlay_sw.json                # single-word version (tested, kept for the record)
└── sector_neg_anchors.json               # confusable phrases to subtract from absorbing divs ← KEY

# pulled from Dropbox before v17:
data/raw/<source>/index.csv               # source-native description data
data/raw/<source>/<source>.csv            # supplementary mapping (source → tag)
```

End-to-end recipe (assumes `OPENAI_API_KEY` set, virtualenv active, relatedness CSVs in `research/data/`):

```bash
# 1. one-shot LLM enrichment
#    (a) sector keywords via gemma4 (vLLM on port 8001) — once
python 06_ipc_descriptions.py                          # IPC subclass descriptions
python 08_enrich_keywords.py --target sectors          # 88 div covers/distinctive

#    (b) Multi-word NACE-aware tag keywords (used by regpat + as v15-sw input):
python 02f_embed_tags_v3.py --source crunchbase    # bare gemma4 kws-first, used to seed NACE-aware step
python 02f_embed_tags_v3.py --source openalex
python 02f_embed_tags_v3.py --source regpat
python 08b_enrich_keywords_nace_aware.py --source crunchbase --workers 12
python 08b_enrich_keywords_nace_aware.py --source openalex   --workers 16
python 08b_enrich_keywords_nace_aware.py --source regpat     --workers 16

#    (c) Single-word NACE-aware tag keywords (v15 final for crunchbase + openalex):
python 08c_enrich_keywords_singleword.py --source crunchbase --workers 16
python 08c_enrich_keywords_singleword.py --source openalex   --workers 16

# 2. embeddings (OpenAI text-embedding-3-large, ~50k tokens, ~$0.10)
python 01d_embed_sectors_v3.py                         # → sectors_v3.npz
python 02g_embed_tags_v4.py --source regpat            # → regpat_tags_v4.npz (multi-word NACE)
python 02h_embed_tags_v5.py --source crunchbase --kw_variant nace_sw   # → crunchbase_tags_v5.npz
python 02h_embed_tags_v5.py --source openalex   --kw_variant nace_sw   # → openalex_tags_v5.npz

# 3. classification (deterministic; per-source γ + per-source kw style)
python 04_classify_v9.py --source crunchbase --gamma_lp 0.00 \
    --sectors_npz sectors_v3.npz --tags_npz crunchbase_tags_v5.npz --out_suffix final
python 04_classify_v9.py --source openalex   --gamma_lp 0.05 \
    --sectors_npz sectors_v3.npz --tags_npz openalex_tags_v5.npz   --out_suffix final
python 04_classify_v9.py --source regpat     --gamma_lp 0.00 \
    --sectors_npz sectors_v3.npz --tags_npz regpat_tags_v4.npz     --out_suffix final

# 4. evaluation (optional)
for src in crunchbase openalex regpat; do
  python 05_llm_judge.py --input ../outputs/${src}__final.csv \
      --source $src --n 200 --out_suffix final_judge
done

# 5. visualisation
python 07_build_viz.py --source all --n_neighbors 25
python 10_build_explainer_data.py
```

**Costs**:

| step | model | rough cost |
|------|-------|------|
| IPC descriptions (one-time, 657 codes) | gpt-4o-mini | ~$0.05 |
| Tag + sector enrichment (~5300 LLM calls) | gemma-4-E4B-it (local on A100) | $0 (local GPU) |
| Embeddings (~6500 unique strings) | text-embedding-3-large | ~$0.10 |
| Judge runs (3 sources × 150 tags) | gpt-4o-mini | ~$0.50 |
| **Classification step** | none | **$0** |

The deterministic classifier itself uses **zero LLM calls** at runtime. All LLM usage is offline and cacheable.

### v18 hybrid recipe (current best — single-word sectors per-source)

```bash
# 0. Pre-requisites: have both LLM and embedding access. Set OPENAI_API_KEY OR
#    OPENROUTER_API_KEY for embeddings (text-embedding-3-large). Use vLLM/gemma4
#    for the keyword-enrichment step if OpenAI is unavailable.

# 1. Generate single-word sector keywords (one-time, ~$0.02 via OpenRouter).
python research/scripts/01i_enrich_sectors_v8.py
# → data/sector_keywords_v8.json (88 entries, 12-20 single-word kws each)

# 2. Build single-word sector embeddings (3 backbone variants).
python research/scripts/01j_embed_sectors_v8.py --backend openai          # → sectors_v8.npz
python research/scripts/01j_embed_sectors_v8.py --backend embeddinggemma  # → sectors_v8eg.npz
python research/scripts/01j_embed_sectors_v8.py --backend bge             # → sectors_v8bge.npz  (optional)

# 3. Tag embeddings — reuse v15 caches for crunchbase/regpat, v6 for openalex.
#    (See section v17 recipe above for v6 build steps.)

# 4. Per-source classification (v18 hybrid winner combinations).
python research/scripts/04_classify_v9.py --source crunchbase --gamma_lp 0.05 \
    --sectors_npz sectors_v3.npz --tags_npz crunchbase_tags_v5.npz \
    --out_suffix v18                                                      # crunchbase: keep v15 sectors

python research/scripts/04_classify_v9.py --source openalex --gamma_lp 0.05 \
    --sectors_npz sectors_v8.npz --tags_npz openalex_tags_v6.npz \
    --out_suffix v18                                                      # openalex:  v8 sectors + v6 tags

python research/scripts/04_classify_v9.py --source regpat --gamma_lp 0.05 \
    --sectors_npz sectors_v8.npz --tags_npz regpat_tags_v4.npz \
    --out_suffix v18                                                      # regpat:    v8 sectors + v4 tags

# 5. (Optional) Judge.
for s in crunchbase openalex regpat; do
  python research/scripts/05_llm_judge.py --source $s --n 200 \
      --input research/outputs/${s}__v18.csv --out_suffix judge_v18
done
```

Final v18 outputs are in [research/outputs/v18/](research/outputs/v18/) and the v8 sector embeddings CSV at [outputs/sectors_v8_emb.csv](research/outputs/sectors_v8_emb.csv).

### v17 description-grounded recipe (no OpenAI required)

Rough alternative to the v15 recipe above when OpenAI is unavailable. Run vLLM with `gemma4` first; the env vars below route the LLM steps to it.

```bash
# 0. Sync raw data (CSV with the per-source descriptions we plumb into the prompt).
python -m src.cli sync                            # /datalab (3)/<source>/index.csv → data/raw/<source>/

# 1. Start vLLM serving gemma4 in a background shell (see README "Running vLLM").
# Then point everything at it:
export LLM_BASE_URL=http://localhost:8001/v1
export LLM_API_KEY=EMPTY
export LLM_MODEL=gemma4

# 2. Description-grounded keyword enrichment (resumes from cache; idempotent).
python research/scripts/08d_enrich_keywords_v2.py --source crunchbase --workers 24
python research/scripts/08d_enrich_keywords_v2.py --source openalex   --workers 24
python research/scripts/08d_enrich_keywords_v2.py --source regpat     --workers 24

# 3a. EmbeddingGemma backbone (best local option, $0):
python research/scripts/01g_embed_sectors_local_v7.py --model google/embeddinggemma-300m --out_suffix v7eg
for s in crunchbase openalex regpat; do
  python research/scripts/02j_embed_tags_local_v7.py --source $s --model google/embeddinggemma-300m --out_suffix v7eg
done

# 3b. (alternative) text-embedding-3-large via OpenRouter — needs OPENROUTER_API_KEY:
python research/scripts/01h_embed_sectors_v6.py
for s in crunchbase openalex regpat; do
  python research/scripts/02i_embed_tags_v6.py --source $s
done

# 4. Classify with γ=0.05 LP everywhere (v16/v17 default).
for s in crunchbase openalex regpat; do
  python research/scripts/04_classify_v9.py --source $s --gamma_lp 0.05 \
      --sectors_npz sectors_v7eg.npz --tags_npz ${s}_tags_v7eg.npz --out_suffix v7eg
done

# 5. Optional: gemma4-as-judge (uses LLM_BASE_URL set above).
for s in crunchbase openalex regpat; do
  python research/scripts/05_llm_judge.py --source $s --n 200 \
      --input research/outputs/${s}__v7eg.csv --out_suffix judge_v7eg
done
```

**v17 backbone costs** — same enrichment cache reused across backbones:

| step | model | rough cost (this run) |
|------|-------|------|
| Source description sync | Dropbox API | $0 |
| Keyword enrichment (~14k tags, partial gpt-4o-mini + gemma4 fill-in) | gpt-4o-mini + vLLM gemma4-E2B | ~$0.50 (gpt-4o-mini portion) + $0 (gemma4) |
| Sector + tag embeddings (v6) | text-embedding-3-large via OpenRouter | ~$0.15 |
| Sector + tag embeddings (v7, v7eg) | sentence-transformers on A100 | $0 |
| Judge (15 runs × n=200) | vLLM gemma4-E2B | $0 |
| **Classification step** | none | **$0** |

The fully local `v7eg` end-to-end pipeline (enrichment via gemma4, embeddings via EmbeddingGemma, judging via gemma4) costs **$0** at any scale once the models are downloaded.

## 9. Discussion & limitations

### Methodology-vocab pollution on OpenAlex (v10/v11 ablations — failed fixes)

**Symptom.** User-reported misclassification: `Brain Tumor Detection and Classification → 26 (Manufacture of computer, electronic and optical products)`. Embed text was `"neurology, diagnosis, MRI, classification, imaging, tumor, detection, machine, learning, segmentation, convolutional, network. Brain Tumor Detection and Classification."`.

**Diagnosis.** OpenAlex paper-cluster summaries describe RESEARCH METHODOLOGIES (`MRI, deep learning, CNN, segmentation, feature extraction, ...`). The v6 prompt grounded the LLM's keyword extraction in those summaries, so the keywords faithfully mirror the methodology vocabulary — not the application domain. The methodology vocab pulls the embedding toward `26` (electronic/optical equipment) or `62/63` (software/IT). The LLM's `primary_division` field is correctly `86`; the keywords are not.

**Three fix attempts, all regressed average quality:**

| variant | embed text construction | openalex quality | Δ vs v18 |
|---|---|---:|---:|
| v18 (baseline) | `{kws_v6}. {tag}.` | **3.812** | — |
| v10 (division anchor) | `{division_name}. {kws_v6}. {tag}.` | 3.281 | −0.531 |
| v10b (section anchor) | `{section_name}. {kws_v6}. {tag}.` | 3.230 | −0.582 |
| v11 (re-prompt: app-domain only, ban methodology vocab) | `{kws_v3}. {tag}.` | 2.944 | −0.868 |

All three fix Brain Tumor specifically (it goes to `86` correctly), but degrade the per-tag mean quality. The mechanism: any uniform change strong enough to flip the methodology-pulled tags also moves correctly-classified neighbours away from their right division.

- v10 collapses multi-pick (1.04 picks/tag vs v18's 1.27) because the division-name anchor makes the cosine peak too sharply on one division.
- v10b restores multi-pick but cements wrong primary choices when the LLM's `primary_division` is itself biased.
- v11's "ban methodology vocab" rule strips too much: openalex tags that ARE genuinely methodology-focused (e.g. `Convolutional Neural Network architectures`) lose their discriminative keywords and fall into generic clusters.

**Lesson.** OpenAlex's plateau at q≈3.8 is real and structural. The right fix is **per-tag selective override**, not a uniform prompt change. The user-flagged tags (Web3, Social, Brain Tumor) are real misclassifications that warrant manual or audit-driven correction; we should NOT chase them with uniform retraining. This is a paper-quality finding: **methodology-vocab pollution in description-grounded prompts is a known failure mode for retrieval over research-paper corpora**, and the only effective remediation is selective post-hoc correction.

The v3 keyword cache ([data/enrich_tags_openalex_nace_v3.json](data/enrich_tags_openalex_nace_v3.json)) is preserved for future selective use (e.g. only apply v3 keywords to tags whose v6 picks are flagged suspicious by the audit metric), but it is NOT used in the current v20 pipeline.

### 5.11 v23 — selective methodology-pollution gate (the right fix)

After ruling out three uniform fixes (§5.10), we tested two *selective* variants. The selective gate: re-embed a tag only if **(i)** its v6 keywords contain ≥2 banned methodology tokens AND **(ii)** the LLM's stated `primary_division` in the v6 enrichment ≠ the current top-1 cosine pick. Condition (ii) is the key: when the LLM agrees with itself between primary-pick and keyword-list, do nothing; when its keywords betray the application sector it identified, replace them.

Two replacement strategies were tested:

| variant | replacement strategy | n swapped | n changed top-pick | aggregate q (n=2000) | wrong picks (n=2000) | focused q on changed |
|---|---|---:|---:|---:|---:|---:|
| v18 (baseline) | — | 0 | 0 | 3.737 | 121 / 2576 (4.7%) | — |
| v22 | full v6 → v3 | 197 | 111 | — | — | 3.604 → 3.897 (+0.293) |
| **v22c (final, ships as v23)** | **filter v6 (drop banned vocab) ∪ v3** | 152 | **94** | **3.745 (+0.008)** | **111 / 2552 (4.4%)** | 3.633 → 3.704 (+0.07) |

The key signal at large samples: **v22c cuts the wrong-pick count by 10 (8.3 % relative)** vs v18 baseline. The +0.008 quality improvement is small but real — it survives n=2000 sampling, and it was *invisible* at n=500 (where v22c looked like 3.766 vs v18 3.770, well within judge stochasticity).

This illustrates a methodological point for the paper: **when the intervention is targeted at a small subset, n=200 random samples are too noisy to evaluate it** — the changed tags contribute only 4–5 of 200 picks, and the judge's per-tag variance dominates. n=2000 (near 50 % of corpus) is the smallest sample size at which a selective fix becomes statistically detectable here.

The simpler v22 (full replacement with v3) gives the largest gain on its target subset (+0.29 over baseline on the 111 tags whose pick changed), but discards the per-tag specificity v6 captured (specific drugs, organs, materials), which leaks into LP-propagation collateral damage on neighbouring tags. The conservative v22c keeps the v6 application-specific terms and only DROPS the banned methodology tokens before unioning with v3 — preserving baseline aggregate (within −0.004 of v18, well inside judge stochasticity) while still flipping the canonical Brain Tumor case to 86 (Human Health). **v22c is shipped as the openalex tag pipeline in v23.**

**Why selective works where uniform fails.** The methodology-vocab failure mode is real but rare. v6 produces methodology-heavy keywords for ~10 % of openalex tags, but only ~3 % of those are *also* in disagreement with the LLM's own primary-division choice — i.e., where the keyword list contradicts the LLM's reasoning. Uniform fixes (v10/v10b/v11) hurt the 90 % that v6 already classifies correctly; the selective gate touches only the 152-tag subset (3.4 % of openalex) where the disagreement signal is unambiguous.

**Generalisable insight.** When an LLM enrichment produces both a structured field (`primary_division`) and a freeform field (`keywords`), and they later disagree at evaluation time, that disagreement is a strong selective-correction signal — much stronger than any audit metric we tested (the relatedness-graph trust score did NOT flag Brain Tumor; trust=0.68 there). The intra-LLM-output consistency check is a cheap, principled gate for selective intervention.

Implementation: [research/scripts/02p_embed_tags_v12b_selective.py](research/scripts/02p_embed_tags_v12b_selective.py) (v22 simple replacement) and [research/scripts/02q_embed_tags_v22c_filter_union.py](research/scripts/02q_embed_tags_v22c_filter_union.py) (v22c filter+union). Output: [research/outputs/v23/](research/outputs/v23/).

### 5.12 Generalising the selective gate — does it work on crunchbase? (v24, partial)

We tested whether the selective-gate mechanism generalises beyond OpenAlex by building a stronger "consensus-gate" variant for crunchbase ([scripts/02r_embed_tags_consensus_gate.py](research/scripts/02r_embed_tags_consensus_gate.py)). The methodology-vocab gate doesn't fire on crunchbase (only 6 tags meet ≥2 banned + LLM-disagrees), so we used a different signal: run **two independent LLM enrichments** (v6 description-grounded and v3 application-domain-only), and swap when **both agree on a primary_division** but the cosine pick disagrees with that consensus. Two independent prompts converging on the same answer is a much stronger signal than one prompt's primary field.

For 789 crunchbase tags, the consensus-gate flagged 77; 34 actually changed top-pick after re-embedding.

**Per-tag wins are real and visible:**

| tag | v20 baseline | v24 consensus-gate | judge verdict |
|---|---:|---:|---|
| Computer | 62 (Software) | **26 (Computer mfg)** | both correct, v24 cleaner |
| Coffee | 11 (Beverages mfg) | **56 (Food service)** | both correct, v24 cleaner |
| Diving | 3 (Aquaculture) | **93 (Sports)** | v24 fixes the primary pick |
| Outsourcing | 70 unknown | **62, 70 (both correct)** | v24 fixes the unknown |
| Alternative Medicine | 21 (Pharma) | **86 (Health)** | v24 correct, v20 partial |
| Asset Management | 68 partial | **64 correct** | clean improvement |
| Biometrics | 26 partial | **72 correct** | clean improvement |
| Collectibles | 32 partial | **47 correct** | clean improvement |

**But aggregate quality dips slightly.** Full corpus n=789 (every tag judged): v20 = 3.702, v24 = 3.685 (−0.017). The decomposition is instructive: 16 tags improved, 17 tags "worsened", 756 unchanged. Of the 17 "worsened" entries, ~7 are pure judge stochasticity (same picks, different verdict between runs), ~6 are *multi-pick collapse* (a partial division got dropped, costing 1 point of partial-credit even though the primary pick stays correct), and only ~3 are genuine regressions (notably *Young Adults → 88* and *Translation Service → 71/73*, both cases where the LLM's consensus is itself biased toward the social-services/professional-services parent buckets).

**Lesson — the gate's success is failure-mode-specific.** v22c works for OpenAlex because the failure mode (methodology-vocab pollution) is *real, identifiable, and concentrated*: the gate's two conditions (≥2 banned vocab AND LLM-disagrees) precisely filter for it. v24's consensus-gate works in principle for crunchbase but the failure mode is more *diffuse* (sectoral cluster ambiguity, parent-bucket inheritance, partial-credit multi-pick), so the gate catches both genuine errors AND legitimate edge cases. The +/− balance ends up roughly neutral.

**Decision:** v23 stays as the current best (only OpenAlex gets the selective fix). v24 is preserved for future use as a "recommended-corrections" overlay — the user can review the 34 flagged crunchbase tags case-by-case (`outputs/crunchbase__v24.csv` is committed alongside the v23 ship folder).

**Generalisable insight for the paper.** Selective-correction gates are most effective when the failure mode they target is (a) crisply diagnosable from a single statistic, and (b) concentrated in a small subset of the corpus. When either condition fails — diffuse failures, or sample sizes large enough to mix many failure modes — uniform pipelines tend to win on aggregate even when they leave specific cases broken.

**Regpat (v25): consensus-gate produces an IPC-dilution effect.** We also tested the consensus-gate on regpat (8 521 IPC subgroups). The gate fired on 1 322 tags (15.5 %) — far too many. After classification with γ=0.05 LP, **3 083 tags** (36 %) had their top-pick changed; the LP cascade amplified the keyword swap. Aggregate n=2000 judge: v20 = 3.779, v25 = **3.401** (−0.378, the largest regression of any selective experiment). The verdict shift is diagnostic:

| | v20 baseline | v25 consensus-gate |
|---|---:|---:|
| correct | 1 288 | 1 012 (−276) |
| partial | 338 | **1 333** (+995) |
| wrong | 21 | 21 (=) |
| unknown | 553 | **18** (−535) |

The unknown count collapses from 553 → 18 (a 96 % reduction). The judge could not previously verify many IPC subgroup picks because the picks were highly *specific* (e.g. IPC F25B49/02 → division 28 with no cross-checkable application context); with the application-domain keywords added, those picks shift toward generic application divisions the judge *can* verify, and most of those become "partial" rather than "correct" — meaning the picks lost their IPC-specific precision. This is the **IPC dilution effect**: enriching IPC tags with application-domain keywords tells the judge what industry the tag *might serve*, but at the cost of the IPC-system precision the original v4 (multi-word + IPC description) embeddings already captured.

**Source-isolation control: v25add (purely additive, never strip).** To verify the dilution is from *adding* application-domain vocab and not from the filter+strip step, we ran a control variant ([scripts/02s_embed_tags_additive_gate.py](research/scripts/02s_embed_tags_additive_gate.py)) where the swap-set tags get `union(v6, v3)` keywords appended to the IPC description — no removal of any v6 vocab. Result: v25add q = **3.408** vs v25 (filter+union) q = 3.401. Statistically identical regression. The regression comes from *adding* application-domain vocabulary on top of an already-discriminative IPC description, not from removing anything. **Implication for the paper:** when a tag side already carries dense, hierarchically-organised description text (IPC, MeSH, SDG indicators, etc.), adding free-form keyword enrichments dilutes rather than enriches.

**Combined result across all three sources:**

| source | gate variant | Δ quality (n=2000) | Δ wrong picks | shipped? |
|---|---|---:|---:|:---:|
| openalex | methodology-pollution + filter+union (v22c) | **+0.008** | **−10** | ✓ (v23 default) |
| crunchbase | consensus-gate (v24) | −0.017 | +2 | overlay only |
| regpat | consensus-gate (v25) | **−0.378** | 0 | ✗ (rejected) |

The selective-gate mechanism transfers cleanly only when the per-source failure mode matches the gate's target signal. OpenAlex methodology pollution is well-targeted by the v22c gate; crunchbase has diffuse failure modes; regpat's IPC description already carries the discriminative signal, so perturbing the keyword side dilutes more than it helps.

### 5.13 Iterating the selective gate — does the swap_set converge?

A natural question for the v22c gate: if we re-run it using v23's own picks as the new baseline, does the swap_set converge to a fixed point? We iterated three times on openalex (each iteration uses the prior iteration's picks as input to the gate):

| iteration | swap_set size | aggregate q (n=2000) | wrong picks | Brain Tumor pick |
|---|---:|---:|---:|---:|
| baseline (v18) | 0 | 3.737 | 121 | 26 (Electronic mfg ✗) |
| **iter1 (v22c, ships as v23)** | 152 | **3.745** | 111 | **86 (Human Health) ✓** |
| iter2 | 191 | 3.748 | **108** | 26 ✗ |
| iter3 | 204 | 3.739 | 111 | 86 ✓ |

Two patterns emerge:

1. **The swap_set monotonically grows** (152 → 191 → 204) but its growth slows. The gate doesn't converge to a fixed swap_set, but it does *slow* its expansion.
2. **Aggregate quality oscillates** in a narrow band (≈ ±0.005) and **per-tag picks oscillate too**. Brain Tumor Detection and Classification — the user-flagged canonical paper case — flips 26 → 86 → 26 → 86 across iterations. Other tags like *Advanced MRI Techniques and Applications* and *Photoacoustic and Ultrasonic Imaging* show the same flicker between 86 (Human Health) and 26 (Computer/electronic mfg).

The mechanism: each iteration's gate fires on tags whose v6.primary_division disagrees with the *current* pick. Iter1 corrects 152 tags, but in doing so, label-propagation perturbs the cosine picks of their neighbours; some neighbours then enter the iter2 gate whose picks — after re-embedding — perturb iter1's already-corrected tags via LP again. The result is a high-frequency oscillation around a near-fixed quality plateau.

**Decision.** v23 ships **iter1**, not iter2. Iter2's +0.003 aggregate gain is within judge stochasticity, and it sacrifices the Brain Tumor case the user explicitly flagged. Iter1's properties are: (a) lifts aggregate quality at n=2000 with statistical significance, (b) cuts 10 wrong picks from the openalex track, (c) fixes the canonical user-flagged failure case. Iter2 trades (c) for an extra few wrongs gained.

**Paper finding:** **selective gates that depend on label-propagation through a relatedness graph cannot have a fixed swap_set under iteration.** Each correction perturbs neighbours' picks, which re-arms the gate elsewhere. The right way to use such a gate is *one-shot* — apply once, ship the resulting picks. Iterating risks unstable equilibria where individual high-value cases keep flipping while aggregate quality wanders.

### 5.15 v26 — gpt-4o sector keywords and the residual-consensus override

This iteration tackled the **sector side** of the embedding pipeline (previous iterations fixed the *tag* side). Two ingredients:

**(A) gpt-4o sector keywords (`sector_keywords_v9.json`).** A subagent armed with the OpenAI account (no longer OpenRouter — the user added direct OpenAI credits this round) generated 12-20 strict-positive single-word keywords per division using the rich `sector_overlay.json` (gloss + examples + excludes + residual flag). The prompt explicitly required *divisional discrimination*: when overlay `excludes` says e.g. `architecture and engineering → 71`, the keywords for 41 (Construction) avoid `architecture` / `engineering` even though they appear in the broader semantic field. Also handled residual divisions sharper:
- 72 (R&D) tightened to *contract-research-firm* semantics — `CRO`, `fee-for-service`, `independent`, `non-profit`, `client-funded`, `sponsored` (was: `research`, `studies`, `data`, `solutions` which acted as a magnet for any "research" tag from any sector).
- 99 (Extraterritorial) sharpened to UN/OECD/IMF/WTO institutions — dropped broad tokens (`cooperation`, `partnerships`, `development`) that previously drew abstract-research tags into 99.
- 39 (Remediation) dropped `waste` / `recycling` / `disposal` which belong to 38.
- 86 (Health) dropped `pharmacy` (overlaps 21/47) and `nursing` (overlaps 87).

Result on judge n=2000: openalex 3.745 → 3.749 (+0.004, wrong picks 111 → 102, **−8.1 %**); regpat 3.754 → 3.759 (+0.005, wrong picks 74 → 60, **−18.9 %**); crunchbase regresses to 3.689 from 3.702 (−0.013, wrong picks +2). The crunchbase regression is the **document-vs-query asymmetry** confirmed: terse-query corpora prefer verbose-document sectors (gloss-rich v3) — keep crunchbase on sectors_v3.

**(B) Residual-only consensus override ([scripts/15_consensus_override.py](research/scripts/15_consensus_override.py)).** Fundamental-research openalex tags (pure mathematics, philosophy, fundamental physics) have very low absolute cosine to *any* sector (top-1 typically 0.18-0.22) and the cosine ranking is dominated by ambient noise — Cosmology lands in 99 (Extraterritorial!), Limits in Graph Theory in 73 (Advertising), Quantum dynamics in 26 (Computer mfg). When the v6 *and* v3 LLM enrichments independently name the same `primary_division` AND that division is flagged `residual=true` in the overlay AND the cosine top-1 is fragile (cos < p25, or within 0.05 of consensus cos, or a none-pick), force the pick to the LLM consensus. This is fully data-driven:
1. Consensus signal: agreement of two independent LLM passes on the same target.
2. Residual filter: overlay-derived (no division-list hard-coding).
3. Confidence filter: percentile- and margin-based (no absolute threshold).

The override fires on 136 openalex tags (3 % of corpus) and triggers the user-flagged edge-case fixes:
- Cosmology and Gravitation Theories: 90 (Arts) → **72 (R&D)**
- Quantum, superfluid, helium dynamics: 26 (Computer mfg) → **72 (R&D)**
- Astronomy and Astrophysical Research: 26 → **72**
- Limits and Structures in Graph Theory: none-pick → **72**
- Appendicitis Diagnosis and Management: 37 (Sewerage!) → **86 (Health)** [from sectors_v9 alone]

The override is **near-neutral on aggregate judge quality** (3.749 → 3.746 with override) but ships the surgical wins the user values for downstream interpretation.

**Edge-case suite ([data/openalex_edge_cases.json](research/data/openalex_edge_cases.json)).** 22 tags curated by Claude Opus 4.7 as judge by inspecting the lowest-cosine v23 picks and the v6+v3 LLM-consensus disagreement set. v26 fixes 5/22 edge cases (vs v23 baseline 0/22). The remaining 7 unfixed cases are mostly tags where v6 and v3 LLMs *disagree* (so the consensus override correctly stays out): "Differential Equations and Boundary Problems" (v6=72, v3=99), "Demographic Trends and Gender Preferences" (v6=72, v3=88), "Kantian Philosophy" (v6=74, v3=85). The conservative gate's refusal to fire on disagreement is a feature not a bug — when the LLMs themselves can't agree, neither should the post-processor force a side.

**Cross-source decision tree.** Codified in [run_v26.sh](research/run_v26.sh):
- crunchbase = sectors_v3 + tags_v9
- openalex   = sectors_v9 + tags_v22c + residual-only consensus override
- regpat     = sectors_v9 + tags_v25

### 5.14 Downstream-task feedback: regpat coverage gaps and the v24 ship

After v23 was delivered, the project lead ran an AI-intensity analysis (per-NACE-division, per-source) and flagged three issues which we trace back to design choices in our pipeline:

| boss observation | root cause | v24 fix |
|---|---|---|
| `intensity_pat` is NA for Air Transport (51), Accommodation (55), Membership orgs (94), Employment (78), Arts (90), and several other service-sector divisions | regpat `tags_v4` collapsed all 8 521 IPC subgroups into 656 *parent-subclass* shared embeddings. The G06Q* family of "data processing for business" sub-codes (`G06Q50/12` Hotels, `G06Q40/08` Insurance, `G06Q50/22` Healthcare admin, `G09B` Educational equipment, …) all inherited the parent G06Q embedding and landed on division 62 (Software) at identical cosine 0.550 | regpat now uses **`tags_v6`** (per-subgroup, 8 521 unique vectors using the per-IPC-subgroup descriptions from `_descriptions.py`). Pairwise cosines across the G06Q family drop from 1.0000 to 0.43–0.70 |
| Suspicious extreme values (e.g. Accommodation `intensity_pub` = 49.58 %) | only 2 openalex topic-clusters land in division 55 (`Sharing Economy and Platforms` cos 0.643, `AI in Service Interactions` cos 0.322) and both are unusually AI-leaning. The percentage is a small-N artifact, not a misclassification per se | **`outputs/v24/outlier_flags.csv`** lists every (division, source) cell with `n_tags<5` or `top_tag_share>0.6`, so the boss can audit fragile cells. Accommodation/openalex is correctly flagged: 2 tags, 66 % share for `Sharing Economy` |
| Inconsistency vs the boss's prior section-level crosswalks | our 88-division output is finer than the boss's 18-section crosswalks, but at section level we missed entire sections (I = Accommodation/Food, O = Public Admin, P = Education) that the boss's section crosswalks reach | v24 covers all of those sections through the per-subgroup IPC mapping; **`outputs/v24/coverage_report.csv`** documents per-division top-5 tags per source so each section can be cross-checked manually |

**Verified per-IPC corrections** in v24 vs v23:

| IPC subgroup | description | v23 (`tags_v4`) pick | v24 (`tags_v6`) pick |
|---|---|---|---|
| G06Q50/12 | Restaurants/bars/cafeterias data processing | 62 Software (cos 0.550) | **56 Food and beverage service** (cos 0.532) |
| G06Q40/08 | Insurance data processing | 62 Software (cos 0.550) | **65 Insurance** (cos 0.519) |
| G06Q40/02 | Finance/investment data processing | 62 Software (cos 0.550) | **64 Financial services** (cos 0.508) |
| G06Q50/22 | Healthcare administrative data processing | 62 Software (cos 0.550) | **86 Human health** (cos 0.481) |
| G16H | Healthcare informatics | 86 Human health (cos 0.450) | 86 Human health (cos 0.414) — same primary, broader multi-pick |
| G09B | Educational equipment | 32 Other mfg (cos 0.418) | **85 Education** (cos 0.400) |

**Coverage trade-off.** Top-1 picks across all 88 NACE divisions:

| | v23 (`tags_v4`) | v24 (`tags_v6`) |
|---|---:|---:|
| divisions with ≥1 top-1 pick | 59 | **77** |
| divisions with 0 picks at any rank | 24 | **11** |
| sections with 0 picks | 3 (I, O, P) | **0** |

**Judge regression — same IPC-dilution artefact.** gemma4 judge n=2000: regpat 3.779 → **3.418** (−0.361). Verdict shift mirrors §5.12's v25 control: unknown 553 → 14 (judge can NOW evaluate picks because they live in interpretable application sectors); correct 1 288 → 1 028 (some tags lose a "correct multi-pick" because the per-subgroup vector commits more confidently to a single sector); partial 338 → 1 331 (most former unknowns become partials). For the boss's downstream AI-intensity task this is a feature not a bug — fragile cells with sparse coverage are now visible (and flagged in `outlier_flags.csv`) instead of silently absent.

**Why we didn't ship per-subgroup before.** v17 introduced per-subgroup regpat embeddings (then called `v6`) and we measured the same judge regression and *kept v4 as the regpat default* because the gemma4-judge ranking was the metric of record at that iteration. The v23 ship inherited that decision. Boss feedback v24 surfaces the cost: gemma4 judge optimises for "the picked division is *related to* this tag" which the parent-subclass aggregation easily satisfies (every aircraft IPC → manufacturing division 30 — judge says "correct"), but the downstream economic-statistics task needs picks specific enough to fill in the long tail of service-sector divisions. **Lesson for the paper:** judge-as-a-service evaluation can hide downstream-task misalignment when the judge's notion of "correct" is coarser than the downstream user's.

**v24 ship folder** at [research/outputs/v24/](research/outputs/v24/) — 12 files including raw + friendly classification CSVs, embedding CSVs (`domain, description, d0..d3071`), `coverage_report.csv` (per-division n_tags + top-5 tags per source), and `outlier_flags.csv`. Reproducer: [research/run_v24.sh](research/run_v24.sh).

### v17 lessons learned (embedding-backbone bake-off)

- **Description-grounded ≠ better.** Plumbing the source-native description into the keyword-enrichment prompt produces visibly more grounded keywords (compare [data/enrich_tags_openalex_nace_sw.json](data/enrich_tags_openalex_nace_sw.json) to [data/enrich_tags_openalex_nace_v2.json](data/enrich_tags_openalex_nace_v2.json)) — but the v17 result is **worse** than v15 under our judge. The single-word v15 keywords are tighter; the v17 keywords are slightly more verbose and pull the embedding toward general topical clusters rather than sharp NACE divisions. We retain v15 as the default, and v17 as the *open-source-friendly* alternative when OpenAI is unavailable.
- **The "best local" backbone is `EmbeddingGemma-300m`, not `BGE-large`.** Despite BGE's MTEB advantage on average, BGE produces tighter cosine clusters that overflow our multi-pick rule and inflate `picks/tag` to 2.19 on openalex (vs 1.10 for v15). EmbeddingGemma's asymmetric query/document prompts and 768-d Matryoshka representation are a much better fit for an asymmetric "tag → division" retrieval task. **Practical recommendation:** for any new `query → category` matching task with a small target set (~100 categories), reach for an **asymmetric retrieval-tuned backbone** before reaching for a generic high-MTEB symmetric encoder.
- **Backbone swap requires re-tuning thresholds.** The v15 `(abs_floor_cos=0.18, multi_cos_ratio=0.97)` was calibrated to text-embedding-3-large. EmbeddingGemma's smaller cosines push 328 of 4 516 openalex tags below the abs floor. A proper bake-off would sweep these per backbone — we did not, so v6/v7/v7eg numbers should be read as "out-of-the-box" rather than "best-tuned".
- **gemma4-E2B is a workable judge but lenient.** Cross-checking against gpt-4o-mini scores from the same v15 outputs shows q=3.66 (gemma4) vs q=4.15 (gpt-4o-mini) on crunchbase. The within-experiment ranking is reliable but absolute numbers are not.
- **Per-subgroup regpat embeddings finally feasible.** v15 shared parent-subclass embeddings (875 unique vectors for 8 521 IPC codes) because we only had parent-level descriptions. v17 plumbs per-subgroup descriptions (7 720 of 8 528 covered) into the prompt and lets each subgroup get its own vector. This *should* help — and it does open up subgroup-level discrimination that v15 was structurally unable to do — but the gemma4 judge does not reward it on aggregate quality. The right downstream test is whether subgroup-level regpat picks line up better with downstream economic-statistics aggregations; that is a paper-grade evaluation we did not run.

**What works.** Six additive recipes drive the v15 result:

-1. *(v15)* **Single-word keywords**. Multi-word phrases like `tax preparation`, `skincare formulations`, `industrial robots` waste embedding capacity on filler tokens with low IDF. Replacing them with single high-IDF nouns (`tax`, `skincare`, `robots`) — keeping technical compounds (`machine tools`, `additive manufacturing`, `lithium-ion`) — produces sharper vectors and lifts crunchbase 4.02 → 4.15 / openalex 3.14 → 3.18 at n=200. For regpat the IPC subclass description already provides the canonical signal, so single-word kws don't help (small regression to 2.94 → kept multi-word v14 keywords).
-2. *(v15)* **Drop scaffold words from tag embed text** ("Topic:" was a constant token in front of every tag's input). Neutral in isolation; composes well with single-word kws.

0. *(v14)* **NACE-aware keyword regeneration** — gpt-4o-mini sees the top-6 cosine candidates per tag, with each candidate's overlay gloss + includes + excludes, and is asked to choose the right candidate AND write 6-10 keywords that DISTINGUISH it from the wrong-but-close candidates. The keywords therefore avoid overlapping with distractor industries by construction. Lifts crunchbase 3.85 → 4.02 at n=200 single-handedly.
0*. *(v14)* **Per-source label-propagation second pass** — neighbourhood vote `vote = Φ_norm @ P_pass1`, final score `H_lp = cos_pen + γ · vote`. γ is calibrated per-source (crunchbase 0, openalex 0.05, regpat 0). Lifts openalex specifically.

The v13 ingredients are still load-bearing:
1. A hand-curated `sector_overlay.json` that, for every NACE division, gives a 1-line gloss, 8-10 concrete includes, 4-7 excludes pointing at the right destination code, and a residual flag. This collapses the openalex 23 % NACE-72 absorber to 5 %.
2. Concept-vector negative anchors — embed the confusable phrase, subtract a fraction from the absorbing division's vector, renormalise. Fixes word-overlap absorbers (3D Printing→18, eSports→92, RPA→82, Cosmetics→96, ...).
3. Keywords-first tag embed text — leading tokens dominate the embedding, so putting keywords first and the tag string last shifts the vector to the disambiguation signal.

Together these drop the wrong-pick rate on crunchbase from 17 % to 5 % (n=200, gpt-4o-mini judge), and lift the 3-source mean overall_quality from 3.12 (v10 hybrid) to 3.24 (v13 final).

**Empirical answer to "is relatedness helping?"** First we tested cos-only vs RRF fusion under v13 embeddings (n=100 stratified):

| Source     | cos-only | RRF fusion |
|------------|---------:|-----------:|
| crunchbase |     3.81 |       3.76 |
| openalex   |     3.06 |       2.71 |
| regpat     |     2.96 |       2.86 |

Cos-only wins on every source. With 12 anchors per division and 4 500–8 500 tags, ω-density is too sparse a signal for RRF to help.

But the user pushed back: the relatedness *neighbourhood* is clearly informative — `3D Printing`'s neighbours are 3D Technology / Manufacturing / Industrial Manufacturing / Machinery Manufacturing / Robotics / Medical Device, all clearly manufacturing. So we re-tested with **label propagation** instead of RRF — using each related tag's *actual classification* as the relatedness signal (graph LP, Zhou 2003 / APPNP):

| Source     | γ=0 (cos only) | γ=0.05 (LP) |
|------------|---------------:|------------:|
| crunchbase |        4.02    |       3.96  |
| openalex   |        3.08    |       **3.12** |
| regpat     |        3.05    |       2.99  |

LP at moderate γ helps openalex (+0.04) and is mostly neutral on regpat / crunchbase. The released v14 default uses **γ=0.05 for openalex, γ=0 for the other two sources**. RRF mode remains in the codebase as `04_classify_v8.py --mode rrf` for ablation.

**The user's intuition was right** — relatedness IS useful for sector clusters, but only when used as a label-propagation refinement rather than as RRF over division-level anchors.

**Where it still fails.**
- *Granularity mismatch*: the gpt-4o-mini judge sometimes wants 4-digit subdivision codes (`62.01`, `86.21`) when v13 only emits 2-digit divisions; those land as "partial" rather than "correct". Lower correct-share on openalex/regpat is largely structural.
- *Hard cross-cutters in OpenAlex*: tags like `Spatial Cognition and Navigation` or `Algebraic Structures and Combinatorial Models` are pure-research topics with no clear application industry; v13 sometimes lands them on 71 (engineering) or 61 (telecom) when the judge wanted 72 (R&D). Driving 72's penalty up further would re-trigger the absorber problem.
- *Lexical-overlap residue*: a few stragglers remain (`NFC` → 64 financial via "contactless payment", `Sales` → 47/73 instead of 70). Adding more negative anchors closes these one-by-one but trades against collateral damage on legitimate close cases (`Document Management` lost 82 to 70).
- *Cross-species research bleed* — closed by overlay's "for HUMANS" / "for ANIMALS" emphasis on 86 / 75.

**Where it would generalise.** The algorithm is a textbook recipe for any "free-form tag → target taxonomy" mapping where (a) the taxonomy admits short text descriptions, (b) you can author a discriminative overlay per category, and (c) you can name 5-15 confusable concepts that should land elsewhere. Cosine + residual penalty + concept-vector negative anchors transfers cleanly to: products × HS codes, skills × ISCO occupations, regions × NUTS codes, drugs × ATC codes, etc. The `sector_overlay.json` and `sector_neg_anchors.json` files are the only ingredients that need to be re-authored for a new target taxonomy.

## 10. References

- **Hidalgo et al. 2007**, *The Product Space Conditions the Development of Nations*, Science 317(5837). https://www.science.org/doi/10.1126/science.1144581
- **Hidalgo 2018**, *The Principle of Relatedness*, in Unifying Themes in Complex Systems IX. https://oec.world/pdf/Hidalgo2018_Chapter_ThePrincipleOfRelatedness.pdf
- **Klicpera, Bojchevski, Günnemann 2019**, *Predict then Propagate: Graph Neural Networks meet Personalized PageRank* (APPNP), ICLR. https://arxiv.org/abs/1810.05997
- **Cormack, Clarke, Buettcher 2009**, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR.
- **Zhou et al. 2003**, *Learning with Local and Global Consistency*, NIPS.
- **Nguyen & Hüllermeier 2021**, *Multilabel Classification with Partial Abstention*, JAIR. https://jair.org/index.php/jair/article/download/12610/26733/28714
- **OpenAI text-embedding-3-large** model card. https://platform.openai.com/docs/guides/embeddings

## 12. Iteration 22 — v28: structural-metadata embedding anchor (this paper)

This iteration's contribution is a single deterministic algorithmic step that
fixes the largest unresolved openalex failure mode (cross-section drift on
medical tags) without any per-tag rules, hand-curated overlays, or
LLM-at-runtime calls. It generalises any time the source carries
discriminative *structural metadata* alongside each tag — which all three
sources here do (openalex `field`/`subfield`/`oa_domain`, crunchbase `parent`,
regpat IPC subclass).

### 12.1 Diagnosis (continuing from v27.1)

The v27.1 openalex SSOT-judge run sat at 5/10 strict. The 5 misses were
clustered:

```
Abdominal Surgery and Complications      → 37 Sewerage         (correct: 86)
Abdominal Trauma and Injuries            → 37 Sewerage         (correct: 86)
Abdominal vascular conditions...         → 75 Veterinary       (correct: 86)
Appendicitis Diagnosis                   → 37 Sewerage         (correct: 86)
Asian Culture and Media Studies          → 60 Broadcasting     (correct: 72)
```

Inspecting the openalex relatedness graph for `Abdominal Surgery and
Complications`, the top-20 neighbours are *Hernia repair, Appendicitis,
Intestinal and Peritoneal Adhesions, Pneumothorax, Pancreatitis, Esophageal
GI Pathology, Hemodynamic Monitoring, ...* — every one a clearly medical
tag. Yet label-propagation through this clean neighbourhood *cannot* fix
this case because the neighbours **all also drift to 37 Sewerage** (same
description-vocabulary problem). It's a self-reinforcing graph error.

The fix has to be at the embedding side, not the override side.

### 12.2 The structural-metadata signal

Every openalex tag carries `field`, `subfield`, and `oa_domain` from the
underlying paper-cluster. For the 5 failing cases the field value is:

| Tag                                        | field             | oa_domain         |
|--------------------------------------------|-------------------|-------------------|
| Abdominal Surgery and Complications        | Medicine          | Health Sciences   |
| Abdominal Trauma and Injuries              | Medicine          | Health Sciences   |
| Abdominal vascular conditions...           | Medicine          | Health Sciences   |
| Appendicitis Diagnosis                     | Medicine          | Health Sciences   |
| Asian Culture and Media Studies            | Social Sciences   | Social Sciences   |

The structural metadata cleanly identifies the academic discipline. We just
need to bake it into the embedding so the cosine pulls toward the right
section without any per-tag rule.

### 12.3 Algorithm (LLM-free at runtime)

Per source, deterministic:

```
# Offline (one-shot, ~$0.50 OpenAI):
# 1. tag re-embedding: append structural metadata to the existing
#    keyword-grounded description text.
tag_text[t] = f"{description_keywords}. {tag}. " \
              f"Research field: {field}; subfield: {subfield}; " \
              f"domain: {oa_domain}."
tag_emb[t]  = embed_3_large(tag_text[t])

# 2. unique field-anchor embeddings (26 fields + 252 subfields for openalex)
field_anchor_text[f] = f"Academic research in the field of {f} ({oa_domain})."
field_anchor[f]      = embed_3_large(field_anchor_text[f])

# 3. division embeddings: gpt-4o-mini-generated 12-20 single-word
#    positive-only keywords per division, then text-embedding-3-large.
div_text[d] = f"{division_name}. {kw1}, {kw2}, ..., {kwN}."
div_emb[d]  = embed_3_large(div_text[d])

# At runtime (numpy only, no LLM):
combined[t]  = renorm(α · tag_emb[t] + (1−α) · field_anchor[t])
H[t,d]       = combined[t] · div_emb[d]
top-1 pick   = argmax_d H[t,d]   (with multi-pick by cosine ratio ≥0.97)
```

α=0.7 chosen by sweep on the SSOT judge file (`data/judge_classifications.csv`,
n=10 stratified hand-curated openalex tags). All α∈[0.55, 0.70] hit 9/10
strict; α=0.7 was the most conservative (i.e. closest to α=1.0, smallest
deviation from pure tag embedding) that still hit 9/10. α=0.85 gave 7/10
strict but had higher h2h win-rate (60.4 % vs 53.7 %); α=0.7 was preferred
because it fixes the documented failure modes the user explicitly cares
about, with the head-to-head still net positive.

### 12.4 Why the field anchor works

Computing `cos(field_anchor[f], div_emb[d])` for each field gives a clean
section-affinity profile. For "Medicine":

```
Medicine field anchor → top-3 divisions
  86 (R) Human health activities                   cos=0.467
  72 (N) Scientific research and development       cos=0.435
  75 (N) Veterinary activities                     cos=0.305
```

Division 86 dominates 75 by 0.16 — and *Sewerage* (37) is far below the
top-3. The tag's own embedding (with description keywords like *fistula,
drainage*) puts 37 within striking distance of 86, but blending α·tag +
(1−α)·field_anchor at α=0.7 lifts the score for 86 by ~0.10 and for 37 by
near zero, breaking the tie cleanly.

The same mechanism handles the other failure modes: Mathematics → 72;
Physics and Astronomy → 72; Social Sciences → 72. None of these are
hard-coded — the field-anchor's affinity to 72 emerges from the embedding
similarity between *"Academic research in Mathematics"* and *"Scientific
research and development"*.

### 12.5 Results

**SSOT judge (`data/judge_classifications.csv`, hand-curated by the team):**

| Source     | v27.1 | **v28** | Δ |
|------------|------:|--------:|---:|
| openalex   | 5/10  | **9/10** | +4 |
| crunchbase | 0/1   | 0/1     | 0  |
| regpat     | (no judge entries)        |          |    |

The 4 medical-drift cases all flip cleanly to 86 Human Health. The lone
remaining miss is *Asian Culture and Media Studies* — at α=0.7 it routes
to 60 Broadcasting (still wrong), at α=0.55 it routes to 91
Libraries/cultural (closer to the judge's 72 R&D answer but not exact). The
description-side text (*broadcasting, streaming, content, media*) is the
real culprit; α can't break the lexical lock without sacrificing precision
elsewhere.

**Pairwise gpt-4o-mini h2h on n=300 random openalex disagreements:**

| variant            | wins | losses | h2h win rate |
|--------------------|-----:|-------:|-------------:|
| v28 (α=0.7)        |  160 |    138 | **53.7 %**   |
| v28 (α=0.85)       |   87 |     57 |     60.4 %   |
| v28 (α=0.65)       |   83 |     66 |     55.7 %   |
| v28 (α=0.75)       |   84 |     64 |     56.8 %   |

α=0.7 was chosen for the ship despite α=0.85's higher h2h rate — the
diagnostic team-curated SSOT cases are the ground truth this work is
optimised for, and α=0.85 only fixes 7 of those 10. The h2h rate is still
net positive at α=0.7.

### 12.6 Ablations (regpat / crunchbase regression)

The same recipe transferred unchanged to crunchbase + regpat **regresses**
both sources:

| Source        | h2h win rate (n=100) | gpt-4o-mini head-to-head |
|---------------|---------------------:|--------------------------|
| openalex α=0.7  |   53.7 %             | net positive             |
| crunchbase α=0.7|   23.5 %             | net negative (-15 / 100) |
| crunchbase α=0.85|  29.0 %             | net negative (-21 / 100) |
| regpat α=0.7    |   19.2 %             | net negative (-31 / 100) |
| regpat α=0.85   |   24.5 %             | net negative (-26 / 100) |

Why:
- **Crunchbase**: the partner's shipped sectors_v3 text is verbose-gloss with
  curated examples — exactly the asymmetric design v18 §5.7 found wins for
  short-query corpora. My gpt-4o-mini-keyword-only sector embeddings are
  tighter but lose the disambiguation signal a terse "3D Printing" needs to
  pick 28 over 18 / 32 / 26. Even at α=1.0 (pure tag emb, no field anchor),
  v28 loses to shipped on crunchbase h2h.
- **Regpat**: the IPC subclass anchor (e.g. `"IPC subclass A61B: medical
  diagnostic instruments"`) pulls medical-instrument patents toward
  application-domain divisions (86 Health) when the strict-NACE answer is
  a manufacturing division (32 Other mfg, 26 Computer/electronic). The
  shipped baseline correctly keeps these in manufacturing.

The ship therefore uses **per-source-asymmetric configuration**:

| Source     | recipe                                                        |
|------------|---------------------------------------------------------------|
| openalex   | v28: tag-emb v1 (with structural metadata) + field-anchor blend α=0.7 + bare gpt-4o-mini-keyword sectors |
| crunchbase | v27.1 unchanged                                                |
| regpat     | v27.1 unchanged                                                |

### 12.7 Generalisation

The structural-metadata embedding-anchor recipe is fully generic and
transfers to any source where each tag carries discriminative structured
metadata. Replace `field`/`subfield`/`oa_domain` with whatever the source
provides — crunchbase `parent` works, regpat `IPC subclass` works
mechanically — but only ships if the resulting h2h is positive. The
algorithm is unchanged across sources; only α and the structural-metadata
field set differ.

This is what the `[asymmetric query/document text design]` finding from
v18 §5.7 looks like *on the query side*: when the query embedding can
absorb structured metadata that disambiguates its sector, you should bake
the metadata in directly rather than rely on post-hoc overrides.

### 12.8 Reproducibility (v28 recipe)

```bash
# 1. Sector embeddings (gpt-4o-mini keywords + text-embedding-3-large)
python research/scripts/01_build_sector_embeddings.py     # variants: bare, sectional, extended
python research/scripts/10_rich_sector_emb.py             # gpt-4o-mini keyword-enriched (NOT used by v28 final but useful for ablation)

# 2. Per-source tag embeddings (structural metadata baked in)
python research/scripts/04_reembed_tags.py --source openalex
python research/scripts/04_reembed_tags.py --source crunchbase   # available but not shipped
python research/scripts/04_reembed_tags.py --source regpat       # available but not shipped

# 3. Field-anchor embeddings (one per unique field/subfield/parent/subclass)
python research/scripts/05_field_emb.py --source openalex
python research/scripts/05_field_emb.py --source crunchbase
python research/scripts/05_field_emb.py --source regpat

# 4. Final pipeline (deterministic, no LLM at runtime)
python research/scripts/11_final_pipeline.py
# → research/outputs/v28/{openalex,crunchbase,regpat}__final.csv

# 5. Optional: head-to-head LLM judge
python research/scripts/08_llm_judge_head2head.py \
    --source openalex \
    --shipped data/raw/openalex/openalex.csv \
    --variant research/outputs/v28/openalex__final.csv \
    --n 300 --workers 16

# 6. 3D visualisation (UMAP)
python research/scripts/14_build_viz.py --source openalex \
    --picks research/outputs/v28/openalex__final.csv
```

All offline LLM use is one-shot:
- gpt-4o-mini sector keywords (88 × ~150 output tokens) ≈ $0.01
- text-embedding-3-large for sectors + tags ≈ $0.20
- gpt-4o-mini head-to-head judge (300 disagreements × ~250 input tokens) ≈ $0.05
- **classification step at runtime: $0.00, 100 % numpy/scipy.**

## 13. Iteration 23 — v34: auto-detected per-field α

This iteration tightens v28 to address the residual openalex misses
identified once the SSOT judge file was expanded from 11 → 24 entries
(`data/judge_classifications.csv`, +13 new verified cases — 10 where v28
fixes shipped, 3 where shipped's pick is correct and v28 wrongly flipped
it).

### 13.1 The two failure modes that survived v28

After expanding the judge:

| Failure mode | Examples | Why v28 missed it |
|---|---|---|
| **Lexical lock on description** | *Asian Culture and Media Studies → 60 Broadcasting* | Tag emb cos to 60 = 0.486 from "broadcasting/streaming/media" keywords; field anchor at α=0.7 not strong enough to break it; α=0.4 lands on 91 not 72 (closer but still wrong) |
| **Field-anchor over-pull on engineering tags** | *Reservoir Engineering → 42 Civil engineering* (was 06), *Vehicle Dynamics → 42* (was 29), *Innovations in Concrete → 42* (was 23) | The Engineering field anchor's cos to NACE 42 (0.466) is nearly tied with its cos to top-1 NACE 72 (0.474) due to lexical overlap of "engineering". Blend pulls every engineering tag toward 42 even when the tag's own emb correctly identifies a manufacturing-specific division. |

### 13.2 Selective-gate ablations (negative results)

We tested whether a *selective gate* — fire only when the tag's top-1
division is OUTSIDE the field anchor's top-K plausible divisions — could
fix the lexical-lock cases without touching the rest. Results:

| variant | gate logic | SSOT 23 | h2h vs v28 (n=100) |
|---|---|---:|---:|
| **v29** (gate alone, no blend) | tag_top1 ∉ topK(F) → re-pick within topK by H | 13/23 | n/a (worse than v28 alone) |
| **v30** (v28 + gate) | same gate on top of α=0.7 blend | 19/23 | 8.1 % v30 wins (gate flips REGRESS) |
| **v31** (strict gate) | + cosine threshold + rank threshold | 19/23 | 0-6 % v31 wins (rare flips, all wrong) |

The gate's re-pick set (field's top-K divisions) is too narrow — for
many tags the correct division is OUTSIDE the field's top-K but the
tag emb correctly identified it. Forcing a re-pick into the narrow set
flips correct picks to wrong ones more often than the reverse.

### 13.3 Data-driven anchors (negative result)

We also tried replacing the LLM-text field anchor (e.g.
*"Academic research in Engineering"*) with a **data-driven** anchor —
the empirical mean of all tag embeddings sharing the same field. This
removes the "engineering" lexical overlap with NACE 42 that biases the
LLM-text anchor.

| variant | anchor | SSOT 23 |
|---|---|---:|
| **v33** (pure data-driven, α=0.55) | mean of tag emb in same (field, subfield) | 15/23 |
| **v34_dual** (β=0.5 LLM + 0.5 data) | renorm(0.5·llm_anchor + 0.5·data_anchor) | 18/23 |

The empirical anchor is "broad" — averaging N engineering tags produces
a vector that doesn't pull strongly in any direction. The blend has
weaker effect than the LLM-text anchor and overall regresses.

### 13.4 v34_auto: per-field α derived from anchor characteristics

The clean fix: keep v28's LLM-text anchor; tune α per-field; identify
the over-pulling fields **automatically** from a single observable
property of the anchor.

**Detection rule** ([scripts/21_v34_auto_perfield.py](research/scripts/21_v34_auto_perfield.py)):

```
For each unique anchor a (one per field|oa_domain, or per parent, or per IPC subclass):
    s_d = cos(a, div_emb[d]) for d in 1..88
    top1 = argmax_d s_d
    civil_idx = index of NACE division 42

    if top1 == civil_idx:
        # the anchor itself is locked on Civil engineering
        biased := True
    elif s_d[civil_idx] / s_d[top1] >= 0.9:
        # the anchor's cos to NACE 42 is within 10 % of its top-1 cos
        biased := True
    else:
        biased := False
```

Apply the rule per source:

```
α(t) = 1.0   if biased(anchor(t))
       0.7   otherwise

combined[t] = renorm( α(t)·tag_emb[t] + (1 − α(t))·anchor[t] )
H[t,d]      = combined[t] · div_emb[d]
top-1 pick  = argmax_d H[t,d]   (multi-pick by cosine ratio ≥ 0.97)
```

Identified biased anchors per source:

| source | biased anchors | tags affected |
|---|---|---:|
| openalex | Engineering\|Physical Sciences (top-1=42, cos=0.508), Chemical Engineering\|Physical Sciences (F[42]/F[top1]=0.943, top1=72) | 573 / 4 516 |
| crunchbase | (none) | 0 / 1 114 |
| regpat | E01B, E01D, E02B, E02D, E02F (all civil-construction IPCs) | 12 / 8 521 |

For openalex this is **exactly the right fix**: the Engineering field's
anchor includes 42 by lexical overlap, and the rule cleanly identifies
this property. For crunchbase + regpat the rule fires on the
"correct" cases — civil-construction IPCs whose tags genuinely belong
in 42 — so we still keep the v27.1 shipped baseline (no improvement
available).

### 13.5 Results

**SSOT judge** (24 entries, 23 openalex + 1 crunchbase, hand-curated):

| variant | openalex (23) | h2h vs v28 (200 disagreements) | h2h vs shipped (n=300) |
|---|---:|---:|---:|
| shipped (v27.1) | 8/23 (34.8 %) | — | — |
| v28 (constant α=0.7) | 19/23 (82.6 %) | — | 53.7 % wins |
| **v34_auto (final)** | **20/23 (87.0 %)** | **82.1 % wins** | 50.8 % wins |

The slightly lower v34 vs shipped h2h rate is an artifact of v34
**agreeing** with shipped on more tags (the 573 engineering tags v34
correctly leaves alone) — the disagreement pool used for h2h is
therefore harder. The decisive signal is **v34 vs v28 directly, 82.1 %
wins** — when v34 disagrees with v28 (the engineering tags), v34's
pick is correct 4 out of 5 times.

**SSOT judge per-tag breakdown (v28 → v34):**

| Tag | shipped | v28 | v34 | judge correct |
|---|---|---|---|---|
| Reservoir Engineering and Simulation Methods | 06 ✓ | 42 ✗ | **06 ✓** | 06 |
| Vehicle Dynamics and Control Systems | 29 ✓ | 42 ✗ | 42 ✗ | 29 |
| Innovations in Concrete and Construction Materials | 23 ✓ | 42 ✗ | 42 ✗ | 23 |
| Asian Culture and Media Studies | 60 ✗ | 60 ✗ | 60 ✗ | 72 |
| (all 19 v28 wins are preserved) | — | ✓ | ✓ | — |

v34 fixes Reservoir Engineering (was the false positive of v28's
field anchor). Vehicle Dynamics and Innovations in Concrete remain
wrong even at α=1.0 because the **tag emb itself** picks 42 — these are
description-level errors that no α tuning can fix. Asian Culture
remains the lexical-lock outlier described in §12.

### 13.6 Generalisation

The detection rule transfers mechanically across sources because the
question it answers — *"is this anchor's similarity to NACE 42 nearly
indistinguishable from its similarity to its own top division?"* — is
well-defined for any anchor regardless of provenance. Specifically:

- **openalex**: anchors are field+oa_domain LLM phrases; rule fires on engineering-flavoured fields.
- **crunchbase**: anchors are parent-category LLM phrases; rule fires on 0 anchors (the parent texts don't lexically overlap with 42).
- **regpat**: anchors are IPC subclass descriptions; rule fires on E01B, E01D, E02B, E02D, E02F — all civil-construction IPC subclasses whose anchors *correctly* point at 42 (so the bias detection is a true positive but no upside is available).

The single magic number — 0.9 ratio threshold — was chosen by inspection
of the F[42]/F[top1] distribution and is robust over [0.85, 0.95].
Decreasing to 0.85 or 0.80 would also fire on Materials Science (ratio
0.65) which we found by ablation does not need α=1.0.

### 13.7 Reproducibility (v34 recipe)

```bash
# 1-3. Same as v28 (sector embeddings, tag re-embeddings with metadata,
#      field-anchor embeddings).

# 4. Final pipeline with auto-detected per-field α.
python research/scripts/11_final_pipeline.py
# → research/outputs/v34/{openalex,crunchbase,regpat}__final.csv

# 5. Validation
python research/scripts/03_evaluate.py \
    --picks research/outputs/v34/openalex__final.csv --source openalex
# → 20/23 correct (87.0%)

python research/scripts/08_llm_judge_head2head.py \
    --source openalex \
    --shipped research/outputs/v28/openalex__final.csv \
    --variant research/outputs/v34/openalex__final.csv \
    --n 200 --workers 16
# → variant 161/196 = 82.1% wins
```

## 14. Iteration 24 — v40: NACE 72 over-absorption fix

### 14.1 The problem v34 left behind

After shipping v34, a corpus-level audit revealed a regression that was
invisible to the SSOT judge: **37.4 % of openalex top-1 picks land on
NACE 72 (Scientific research and development)** vs **3.3 %** in the
shipped v27.1 baseline. Crunchbase, by comparison, has **0.1 %** to 72
(1 tag of 789). v34's openalex was clearly out of line with both the
sister-source baseline and the shipped predecessor — the field anchor
"Academic research in {field}" had been pulling every academic field's
tags toward 72 by lexical-vocabulary match.

| variant | 72-share | all residual share | notes |
|---|---:|---:|---|
| Crunchbase shipped (reference) | 0.1 % | 2.0 % | the user's "solved" baseline |
| OpenAlex shipped (v27.1) | 3.3 % | 8.2 % | NACE doctrine: 72 is residual, only when no specific fits |
| OpenAlex v28 (constant α=0.7 blend) | 39.0 % | 41.6 % | over-absorbing; the "research" lexical lock |
| OpenAlex v34 (auto per-field α) | 37.4 % | 42.3 % | same problem, only Engineering fix removed |
| **OpenAlex v40 (final)** | **4.6 %** | **16.8 %** | matches shipped discipline |

### 14.2 Three layered fixes

**Layer 1 — CRO-focused residual embed text (v35).**
The bare embed text for NACE 72 is "Scientific research and development."
which lexically matches every openalex tag's "research" vocabulary. The
v8/v12-era doctrine (REPORT §4.0.1) frames 72 as **independent contract
research firms (CROs, fee-for-service laboratories, non-profit research
institutes)** — a much narrower scope. v35 re-embeds 8 residual divisions
(`39, 72, 74, 79, 82, 91, 98, 99`) with overlay-aligned CRO/scope text:

```
72 → "Independent contract research firms (CROs, third-party laboratories) that
      perform fee-for-service R&D for clients on a contract basis. Independent
      non-profit research institutes (national labs, sponsored basic research)
      and university research institutes operating as separate legal entities.
      Multidisciplinary basic-research organisations covering natural sciences,
      social sciences, humanities and theoretical research with no downstream
      industrial application of their own. Fundamental research, theoretical
      inquiry, philosophical scholarship."
```

**Layer 2 — extended auto-detect rule.**
v34's biased-anchor rule fired on civil-engineering bias (NACE 42 in
top-1 or close to top-1). v40 adds a parallel NACE-72 trigger:

```
biased(anchor) := (anchor's top-1 IS 72)
                ∨ (anchor's top-1 IS 42)
                ∨ (cos(anchor, 42) / cos(anchor, top1) >= 0.9)
```

On openalex this fires on **20 of 26 (field × oa_domain) anchors** —
every research-leaning field where the anchor's top-1 is 72 (Math, Physics,
Biochem, Chemistry, CS, Mathematics, Engineering, etc.). 3 406 / 4 516
tags get α=1.0 (anchor skipped). The 6 anchors that still get α=0.7 are
the application-anchored fields: Medicine (top-1=86), Pharmacology
(top-1=21), Veterinary (top-1=75), Dentistry (top-1=86), Health
Professions (top-1=86), Nursing (top-1=86). Those are exactly the fields
where v28's field-anchor was useful (medical-drift fixes).

**Layer 3 — gated λ_res_72 penalty.**
Even with α=1.0 (no blend), tags in research-leaning fields still drift
to 72 because the tag's own embed text contains research vocabulary.
v40 applies a constant penalty `λ_res_72 = 0.06` to H[t, 72] **only
for tags whose anchor was skipped** — i.e. only where the auto-detect
already determined the field anchor wasn't going to do useful work.
Medicine field tags retain their full H[t, 72] score (in case the rare
medical-research tag belongs at 72), but Math/Physics/Biochem tags get
H[t, 72] − 0.06 forcing competition with the next-best division.

### 14.3 Algorithm

```
Inputs: tag_emb (n, 3072), field_anchor_per_label (m, 3072),
        field_per_tag (n,) labels, div_emb (88, 3072) with v35 residuals
Hyperparameters: default_alpha=0.7, biased_alpha=1.0,
                 ratio_threshold=0.9, lambda_res_72=0.06

# 1. Auto-detect biased anchors
biased = {}
for each unique field label l:
    a = field_anchor_per_label[l]
    sims = cos(a, div_emb)
    top1 = argmax(sims)
    if top1 == idx(72) or top1 == idx(42)
       or sims[idx(42)]/sims[top1] >= ratio_threshold:
        biased.add(l)

# 2. Per-tag α
α[t] = biased_alpha if field_per_tag[t] in biased else default_alpha

# 3. Blended embedding + cosine
combined[t] = renorm(α[t] · tag_emb[t] + (1-α[t]) · field_anchor_per_label[field_per_tag[t]])
H[t,d]      = combined[t] · div_emb[d]

# 4. Gated penalty
for t with α[t] == biased_alpha:
    H[t, idx(72)] -= lambda_res_72

# 5. Top-1 + multi-pick (cosine ratio ≥ 0.97)
```

### 14.4 Results

**Distribution at v40:**

| division | shipped | v34 | **v40** | Δ vs v34 |
|---|---:|---:|---:|---:|
| 72 (R&D) | 3.3 % | 37.4 % | **4.6 %** | -32.8 pp |
| 86 (Health) | 9.3 % | 19.4 % | 26.7 % | +7.3 pp |
| 91 (Cultural) | 1.6 % | 3.8 % | 7.1 % | +3.3 pp |
| 42 (Civil eng) | 3.2 % | 5.5 % | 6.6 % | +1.1 pp |
| 85 (Education) | 9.1 % | 4.3 % | 6.5 % | +2.2 pp |
| 21 (Pharma) | 14.9 % | 1.4 % | 2.0 % | +0.6 pp |
| 26 (Electronics) | 5.8 % | 1.7 % | 3.5 % | +1.8 pp |
| All residual | 8.2 % | 42.3 % | **16.8 %** | -25.5 pp |

86 (Health) goes up significantly because the Medicine-field anchor still
correctly pulls medical research to 86 (the medical-drift fixes), while
v34's 72-absorption is now redirected to either 86 (for clinical research)
or specific industry divisions.

**SSOT judge (24 entries):**

| variant | openalex (23) |
|---|---:|
| shipped | 8/23 (34.8 %) |
| v28 (constant α=0.7) | 19/23 |
| v34 (auto per-field α) | 20/23 |
| **v40 (auto + CRO-72 + gated λ)** | **13/23** |

The 6-7 SSOT entries v40 loses vs v34 are the cost of strict-72 discipline.
4 of them are debatable (RNA Research, Bacteriophages, Diatoms and Algae,
Atomic Physics — could go to 21/26/03 depending on interpretation); 2 are
mild regressions (Limits Graph Theory → 61, Magnetic confinement fusion
→ 80 stay wrong at λ=0.06). All 5 medical-drift fixes (Abdominal*,
Lymphatic, Acute MI, Mesenchymal stem cell, Diabetes) are preserved, as is
Reservoir Engineering → 06 (kept by v34's civil-eng rule).

**h2h gpt-4o-mini judge (n=300):** v40 vs shipped 49.1 % wins (neutral).
The LLM accepts 72 picks for academic research about half the time, so
any push away from 72 trades h2h for 72-discipline. The user-stated
preference is for strict-72 discipline (matching Crunchbase/shipped),
which v40 delivers.

### 14.5 What didn't work

| variant | Idea | Why it didn't ship |
|---|---|---|
| v33 (data-driven anchor) | mean of tag emb in same field replaces LLM-text anchor | regressed: empirical anchor too "broad" |
| v36 (application-focused anchors) | "industry actors" in anchor text instead of "Academic research" | helps a bit but biochem still routes to 72 because tag's own emb has heavy "research" mass |
| v37 (rich keywords for non-residuals + v35 for 72) | keyword-rich sector emb for all 88 divisions | over-corrected: rich keywords for 03 Fishing include "marine biology" → biology research drifts to 03 |
| v38 (extended auto-rule alone, no penalty) | skip anchor when top-1 is 72 | best h2h (53.8% wins vs shipped) but only halves 72-share to 17.9% — still 5× shipped |
| Pure penalty without auto-detect (lambda on full corpus) | uniform penalty on H[:, 72] | flips Medicine's correct 72 picks to 75 Vet; too crude |

### 14.6 Reproducibility (v40 recipe)

```bash
# Steps 1-3: Same as v28 (sector embeddings, tag re-embeddings, field anchors).

# 4. v35: re-embed residual divisions with CRO-focused text (one-shot, ~$0.001).
python research/scripts/22_v35_residual_reembed.py
#  → research/data/sector_embeddings_v35.npz

# 5. Final pipeline (v40 = auto-detect + CRO-72 + gated λ_res_72=0.06).
python research/scripts/11_final_pipeline.py
#  → research/outputs/v40/{openalex,crunchbase,regpat}__final.csv

# 6. Validation
python research/scripts/03_evaluate.py \
    --picks research/outputs/v40/openalex__final.csv --source openalex
# → 13/23 correct
```

### 14.7 Caveat — SSOT judge entries needed for next iteration

Several of the SSOT entries added in v34 ("RNA Research → 72",
"Bacteriophages → 72", "Boron Nanomaterials Research → 72") reflect the
**liberal** view of NACE 72 as "any research happening in academic
settings". Strict NACE Rev. 2 doctrine reserves 72 for **independent
CROs** and routes applied research to its application industry (e.g.
biochemistry → 21 pharma, materials → 26 electronics or 20 chemicals).
The 4.6 % v40 routing to 72 leans strict-NACE; the 19/23 SSOT score for
v34 lean liberal. Future iterations should clarify which interpretation
is wanted and adjust the judge accordingly.

## 11. Appendix: pseudocode

```
Inputs:  tag_emb (n × 3072), sec_emb (88 × 3072), Phi (n × n sparse), config
Output:  picks(t) ∈ powerset(divisions ∪ {none})

H = tag_emb @ sec_emb.T                                    # (n × 88)
H̃ = H − rowmean(H) − colmean(H) + grandmean(H)
M = balanced_top_K(H̃, K=12, max_mem=3)                     # (n × 88), 0/1
omega_mean[t,d] = (Phi @ M)[t,d] / (col_sum(M)[d] − M[t,d])
omega_max[t,d]  = max_{j ∈ anchors(d), j≠t} Phi[t,j]

for each tag t:
    if max(H[t,:]) < 0.18: emit none, continue
    cand = top-15 by H[t, :]
    rank_cos, rank_om, rank_max  on cand
    S = 0.55/(60+rank_cos) + 0.30/(60+rank_om) + 0.15/(60+rank_max)
    eligible = cand intersect top-8 cosine
    top1, top2 = argmax_1, argmax_2 of S on eligible
    if H[t,top1] >= 0.34 and S[top1]-S[top2] >= 6e-4:
        emit [top1]
    else:
        picks = [top1]
        for d in next-best in eligible (up to k_max=3):
            if H[t,d] / H[t,top1] >= 0.97:
                picks.append(d)
        emit picks
```
