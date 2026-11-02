# Citation Search: JaxMARL-HFT (arXiv:2511.02136)

**Original paper:** JaxMARL-HFT: GPU-Accelerated Large-Scale Multi-Agent Reinforcement Learning for High-Frequency Trading
- URL: https://arxiv.org/abs/2511.02136
- Authors: Valentin Mohl, Sascha Frey, Reuben Leyland, Kang Li, George Nigmatulin, Mihai Cucuringu, Stefan Zohren, Jakob Foerster, Anisoara Calinescu
- Submitted: 3 Nov 2025 (q-fin.TR; cs.MA)
- Venue: ICAIF '25 (Best Paper Honourable Mention), DOI: 10.1145/3768292.3770416
- Code: https://github.com/vmohl/JaxMARL-HFT

**Search date:** 2026-11-01
**Sources used:** Semantic Scholar Graph API, OpenAlex, arXiv full-text (HTML), web search, Google Scholar (partial/blocked), ACM DL "Cited By"

---

## All citing works (ordered by recency, newest first)

### arXiv documents (5)

| # | Date | arXiv ID | Title | Authors | How verified |
|---|------|----------|-------|---------|--------------|
| 1 | 2026-10-20 | [2610.28907](https://arxiv.org/abs/2610.28907) | JaxMARL-BC: Multi-Agent Reinforcement Learning for Macroeconomic Modelling, in Seconds | Daniele Montagnani, Andrea Coletta, Antonietta Mira, Aldo Glielmo | Full-text HTML contains "JaxMARL-HFT" (3x) |
| 2 | 2026-10-19 | [2610.27100](https://arxiv.org/abs/2610.27100) | LOB-Impact Benchmark: Emergent Market Impact in Generative Limit Order Book Models | George Nigmatulin, Valentin Mohl, Sascha Frey, Kang Li, Aramis Marti-Shahandeh, Mihai Cucuringu, Anisoara Calinescu, Jakob Foerster, Stefan Zohren | Full-text HTML contains "JaxMARL-HFT" / "Mohl et al., 2025" — **self-citation by original authors** |
| 3 | 2026-10-19 | [2610.26789](https://arxiv.org/abs/2610.26789) | Cooperative Multi-Agent Reinforcement Learning for Market Making on Prediction-Market Order Books | Skander Lejmi | Full-text HTML: "JaxMARL-HFT" (28x), "Mohl et al" (15x), cites arXiv:2511.02136 directly; **builds experiments on JaxMARL-HFT** |
| 4 | 2026-10-13 | [2610.18158](https://arxiv.org/abs/2610.18158) | When and How Order-Flow Interventions Work After Dealers Adapt | Guanghao Wu, Zhuo Cai, Zhidong Li, Shoujin Wang | Full-text HTML contains "JaxMARL-HFT" (2x), "Mohl et al" |
| 5 | 2026-05-04 | [2605.03125](https://arxiv.org/abs/2605.03125) | Taming the Curses of Multiagency in Robust Markov Games with Large State Space through Linear Function Approximation | Jingchu Gai, Laixi Shi | Full-text HTML contains "Jaxmarl-hft"; also listed in Semantic Scholar citations |

### Non-arXiv documents (3)

| # | Date | DOI | Title | Venue | Authors |
|---|------|-----|-------|-------|---------|
| 6 | 2026-06-01 | [10.11591/eei.v15i3.11171](https://doi.org/10.11591/eei.v15i3.11171) | Novel framework and reference architecture for artificial intelligence models for stock markets | Bulletin of Electrical Engineering and Informatics (Gold OA) | E. Cancho-Rodriguez, Miguel Angel Cano Lengua |
| 7 | 2026-04-29 | [10.1007/s44163-026-01072-y](https://doi.org/10.1007/s44163-026-01072-y) | Orchestrated intelligence: an adaptive multi-agent architecture for high-frequency options risk management | Discover Artificial Intelligence, vol. 6 (Springer, Gold OA) | Mohammadreza Ayatollahi, Seyed Mohammadbagher Jafari |
| 8 | 2025-11-15 | [10.1145/3768292.3793390](https://doi.org/10.1145/3768292.3793390) | Session Summary Podcast: Session 1: Agent-Based Financial Systems | Proc. 6th ACM ICAIF (AI-generated podcast companion; note: AI content may contain errors) | AI Generated |

**Total: 8 citing works (5 arXiv + 3 non-arXiv) as of 2026-11-01.**

---

## Search methodology & coverage notes

### Semantic Scholar Graph API
- Endpoint: `GET /graph/v1/paper/arXiv:2511.02136/citations`
- Also queried `DOI:10.1145/3768292.3770416` (ACM version) — same 5 results.
- Returned 5 citations total; only 2 of them have arXiv IDs (2610.26789, 2605.03125).
- **Missed** 3 arXiv citers (2610.28907, 2610.27100, 2610.18158) that were not yet indexed.

### OpenAlex
- Work ID: `W7104033318` (DOI: 10.48550/arxiv.2511.02136)
- `filter=cites:W710403318` returned **0 citations** (indexing lag / coverage gap).

### arXiv full-text verification
- Fetched `https://arxiv.org/html/{id}v1` for each candidate and grepped for `jaxmarl-hft|2511.02136|Mohl et al`.
- All 5 arXiv citers confirmed present in full text.

### Web search
- Queries used: `"JaxMARL-HFT" site:arxiv.org`, `"Mohl et al" "JaxMARL-HFT" arxiv 2026`, `"2511.02136"`, etc.
- Uncovered the 3 arXiv papers Semantic Scholar had not indexed yet.

### Google Scholar
- Author profile (Sascha Frey) shows **3 citations** for JaxMARL-HFT — lower count, likely lagging or deduplicating differently.
- Direct `scholar.google.com/scholar?cites=...` lookups blocked by CAPTCHA.
- ACM DL "Cited By" shows only the ICAIF podcast entry.

### Not cited by
- **ABIDES-MARL** (arXiv:2511.02016, Cheridito et al., submitted same day 2025-11-03) does **not** appear to cite JaxMARL-HFT (concurrent work; checked HTML — no match).

### Citation-index lag summary
| Source | Count | Notes |
|--------|-------|-------|
| Semantic Scholar | 5 | 2 arXiv + 3 non-arXiv |
| OpenAlex | 0 | Not yet indexed |
| Google Scholar | 3 | Conservative / lagging |
| ACM DL | 1 | Only podcast entry |
| Full-text web search | 5 arXiv | Most complete for arXiv |
| **Combined unique total** | **8** | 5 arXiv + 3 non-arXiv |
