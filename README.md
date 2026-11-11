# Lamba

Code and technical reports for **Lamba**, a family of fine-tuned models from Vixero Labs, built on a Qwen-family
hybrid (Gated DeltaNet + full attention) base.

| Version | What it is | Weights |
|---|---|---|
| 0.1, 0.2 | Doom-loop and answer-letter-bias fixes (LoRA adapters); 0.2 adds a DPO stage on tool-use transcripts | not released |
| 0.3, 0.4 | Identity, tool follow-through, Malay knowledge and tone | **0.4: open weights** (link below) |

- **Lamba 0.4 weights and model card:** https://huggingface.co/VixeroAI/Lamba-0.4
- **Technical report, Lamba 0.1 and 0.2:** [`reports/lamba-0.1-0.2/main.pdf`](reports/lamba-0.1-0.2/main.pdf)
- **Technical report, Lamba 0.3 and 0.4:** in preparation. It continues the 0.1/0.2 report.

## Lamba 0.4 at a glance

Lamba 0.4 against Lamba 0.3, stock Qwen3.8-27B and other models, on our own test sets (not independently reproduced; Lamba 0.4 is on a different base than 0.1 to 0.3). The charts are vector SVG with real text and adapt to light and dark mode; a plain-text version and the tables are below. They are generated from one data table by [`assets/build_charts.py`](assets/build_charts.py); an interactive version is in [`assets/lamba-0.4-metrics.html`](assets/lamba-0.4-metrics.html).

![Chart: Lamba 0.4 against Lamba 0.3 and stock Qwen3.8-27B. Lamba 0.4 leads on 7 of 13 tests, is level on 4 and trails on 2.](assets/lamba-0.4-vs-0.3.svg)

![Chart: Lamba 0.4 against Lamba 0.3, stock Qwen and seven other models on loops and tool maths, five panels.](assets/lamba-0.4-vs-other-models.svg)

<details>
<summary>Text version of the charts</summary>

**Lamba 0.4 leads Lamba 0.3 on 7 of 13 tests, is level on 4 and trails on 2.**

| Calls a search tool when it should (first turn) | MalayMMLU, generative 5k | Chat tone: friendly, blind judge | Refuses harmful prompts (with the constitution) |
|:-:|:-:|:-:|:-:|
| **99.2%** | **84.6%** | **8.0/10** | **93.3%** |
| +26.6 pts vs 0.3<br>0.3: 72.6% | +2.1 pts vs 0.3<br>0.3: 82.5% | about +2.3 vs 0.3<br>0.3: about 5.7 | +15.7 pts vs 0.3<br>0.3: 77.6% |

#### Lamba 0.4 against Lamba 0.3 and stock Qwen

Every bar is "higher is better" (rates where lower is better are shown as 100 minus the rate), on a 0 to 100 scale. Rows are sorted by the gap between 0.4 and 0.3, shown on the right. Stock Qwen3.8-27B appears only where we ran it. Lamba 0.4's base model is different from 0.3's, so the comparison mixes training and base.

```text
0.4 AHEAD (7)

Ibu negara / negeri questions correct                       +34.4
  Lamba 0.4   ███████████████████▎  96.1
  Lamba 0.3   ████████████▍░░░░░░░  61.7

Does not self-introduce on "X tu apa?" (100 minus rate)     +29.0
  Lamba 0.4   ███████████████████▌  97.4
  Lamba 0.3   █████████████▋░░░░░░  68.4

Calls a search tool when it should (first turn)             +26.6
  Lamba 0.4   ███████████████████▉  99.2
  Lamba 0.3   ██████████████▌░░░░░  72.6

Refuses harmful prompts, with constitution                  +15.7
  Lamba 0.4   ██████████████████▋░  93.3
  Lamba 0.3   ███████████████▌░░░░  77.6
  Stock Qwen  ████████████████▋░░░  83.1

Malaysian facts correct                                      +8.0
  Lamba 0.4   ████████████████▍░░░  81.9
  Lamba 0.3   ██████████████▊░░░░░  73.9
  Stock Qwen  ████████████████░░░░  80.2

Avoids needless searches (100 minus over-search)             +4.2
  Lamba 0.4   ███████████████████▊  98.8
  Lamba 0.3   ██████████████████▉░  94.6

MalayMMLU generative, 5k                                     +2.1
  Lamba 0.4   ████████████████▉░░░  84.6
  Lamba 0.3   ████████████████▌░░░  82.5

LEVEL, WITHIN NOISE (4)

GPQA-Diamond (160 vs 155 of 198)                             +2.5
  Lamba 0.4   ████████████████▏░░░  80.8
  Lamba 0.3   ███████████████▋░░░░  78.3

Finishes within token budget                                 +1.9
  Lamba 0.4   ███████████████░░░░░  74.8
  Lamba 0.3   ██████████████▋░░░░░  72.9
  Stock Qwen  ██████▊░░░░░░░░░░░░░  34.0

Refuses harmful prompts, no system prompt                    +0.7
  Lamba 0.4   ███████████████████▎  96.2
  Lamba 0.3   ███████████████████▏  95.5
  Stock Qwen  ███████████████████▏  95.5

Clean endings (100 minus tail repetition)                    −0.3
  Lamba 0.4   ███████████████████▍  97.0
  Lamba 0.3   ███████████████████▌  97.3
  Stock Qwen  ██████████████████░░  90.0

0.3 AHEAD (2)

Says "Lamba" on held-out identity items                      −3.5
  Lamba 0.4   ███████████████████▎  96.2
  Lamba 0.3   ████████████████████  99.7

Loop-free (100 minus Liquid-style loop rate)                 −2.5
  Lamba 0.4   ██████████████████▊░  93.5
  Lamba 0.3   ███████████████████▎  96.0
  Stock Qwen  ███████████████▋░░░░  78.0

```

#### Against other models

Higher is better on every chart. Lamba runs are at repetition penalty 1.05; the other models ran at provider defaults with no penalty. Models without a result are left out of that chart (Gemini 3.8 Flash hides its reasoning so repetition cannot be scored; no loop test was run for Claude Opus 5.5). Claude Sonnet 5.5 and Opus 5.5 ran without extended thinking.

```text
LOOP-FREE - Lamba 0.4 ranks 3rd of 8
(100 minus the share with a 60-character block repeated 4 or more times)
  Claude Sonnet 5.5   ████████████████████ 100.0
  Lamba 0.3           ███████████████████▎  96.0
▶ Lamba 0.4           ██████████████████▊░  93.5
  ThinkingCap         ████████████████▎░░░  81.3
  ILMU v3.1           ████████████████▏░░░  80.7
  DeepSeek v4.1 Flash ██████████████▊░░░░░  74.0
  GLM-5.3 Flash       ██████████████░░░░░░  70.0
  Stock Qwen3.8-27B   █████████████▎░░░░░░  66.0

CLEAN ENDINGS - Lamba 0.4 ranks 3rd of 8
(100 minus the share with any repetition in the last 2,500 characters)
  Claude Sonnet 5.5   ████████████████████ 100.0
  Lamba 0.3           ███████████████████▌  97.3
▶ Lamba 0.4           ███████████████████▍  97.0
  ThinkingCap         █████████████████▍░░  86.7
  DeepSeek v4.1 Flash ████████████████▌░░░  82.7
  ILMU v3.1           ████████████████▎░░░  81.3
  Stock Qwen3.8-27B   ███████████████▋░░░░  78.0
  GLM-5.3 Flash       ██████████████▊░░░░░  74.0

FINISHES WITHIN BUDGET - Lamba 0.4 ranks 4th of 9
(100 minus the share that hit the 4,000-token cap)
  Claude Sonnet 5.5   ███████████████████░  95.3
  Gemini 3.8 Flash    █████████████████▉░░  89.3
  ILMU v3.1           ███████████████▌░░░░  77.3
▶ Lamba 0.4           ███████████████░░░░░  74.8
  Lamba 0.3           ██████████████▋░░░░░  72.9
  ThinkingCap         ██████████▏░░░░░░░░░  50.7
  DeepSeek v4.1 Flash ████████░░░░░░░░░░░░  40.0
  GLM-5.3 Flash       ███████░░░░░░░░░░░░░  35.3
  Stock Qwen3.8-27B   █████▉░░░░░░░░░░░░░░  29.3

TOOL MATHS CORRECT - Lamba 0.4 ranks 6th of 10
(share of 240 eval30 runs that were correct)
  ThinkingCap         ████████████████████ 100.0
  DeepSeek v4.1 Flash ████████████████████ 100.0
  Gemini 3.8 Flash    ████████████████████ 100.0
  Claude Sonnet 5.5   ████████████████████ 100.0
  Claude Opus 5.5     ████████████████████ 100.0
▶ Lamba 0.4           ███████████████████▉  99.6
  Lamba 0.3           ███████████████████▉  99.6
  Stock Qwen3.8-27B   ███████████████████▎  96.2
  GLM-5.3 Flash       ██████████████████▎░  91.2
  ILMU v3.1           ███████████████░░░░░  75.0

STOPS AFTER VERIFYING - Lamba 0.4 ranks 5th of 10
(100 minus the share of solved runs that kept calling tools after verifying)
  Claude Sonnet 5.5   ████████████████████ 100.0
  Claude Opus 5.5     ███████████████████▋  98.3
  GLM-5.3 Flash       ███████████████████▍  96.8
  Stock Qwen3.8-27B   ███████████████████▎  96.1
▶ Lamba 0.4           ███████████████████▏  95.9
  Gemini 3.8 Flash    ███████████████████▏  95.9
  Lamba 0.3           ███████████████████░  95.1
  ThinkingCap         ██████████████████▉░  94.1
  DeepSeek v4.1 Flash ██████████████████░░  89.9
  ILMU v3.1           █████████████▉░░░░░░  69.6

```

</details>

<details>
<summary>The numbers behind the charts (tables)</summary>

| Metric (higher is better) | Lamba 0.4 | Lamba 0.3 | Stock Qwen3.8-27B | Gap, 0.4 vs 0.3 |
|---|---:|---:|---:|---:|
| **0.4 ahead** | | | | |
| Ibu negara / negeri questions correct | 96.1 | 61.7 | not run | +34.4 |
| Does not self-introduce on "X tu apa?" (100 minus rate) | 97.4 | 68.4 | not run | +29.0 |
| Calls a search tool when it should (first turn) | 99.2 | 72.6 | not run | +26.6 |
| Refuses harmful prompts, with constitution | 93.3 | 77.6 | 83.1 | +15.7 |
| Malaysian facts correct | 81.9 | 73.9 | 80.2 | +8.0 |
| Avoids needless searches (100 minus over-search) | 98.8 | 94.6 | not run | +4.2 |
| MalayMMLU generative, 5k | 84.6 | 82.5 | not run | +2.1 |
| **Level, within noise** | | | | |
| GPQA-Diamond (160 vs 155 of 198) | 80.8 | 78.3 | not run | +2.5 |
| Finishes within token budget | 74.8 | 72.9 | 34.0 | +1.9 |
| Refuses harmful prompts, no system prompt | 96.2 | 95.5 | 95.5 | +0.7 |
| Clean endings (100 minus tail repetition) | 97.0 | 97.3 | 90.0 | −0.3 |
| **0.3 ahead** | | | | |
| Says "Lamba" on held-out identity items | 96.2 | 99.7 | not run | −3.5 |
| Loop-free (100 minus Liquid-style loop rate) | 93.5 | 96.0 | 78.0 | −2.5 |

| Model | Loop-free | Clean endings | Finishes within budget | Tool maths correct | Stops after verifying |
|---|---:|---:|---:|---:|---:|
| **Lamba 0.4** | **93.5** | **97.0** | **74.8** | **99.6** | **95.9** |
| Lamba 0.3 | 96.0 | 97.3 | 72.9 | 99.6 | 95.1 |
| Qwen3.8-27B (stock) | 66.0 | 78.0 | 29.3 | 96.2 | 96.1 |
| ThinkingCap-Qwen3.8-27B | 81.3 | 86.7 | 50.7 | 100.0 | 94.1 |
| DeepSeek v4.1 Flash | 74.0 | 82.7 | 40.0 | 100.0 | 89.9 |
| GLM-5.3 Flash | 70.0 | 74.0 | 35.3 | 91.2 | 96.8 |
| ILMU v3.1 | 80.7 | 81.3 | 77.3 | 75.0 | 69.6 |
| Gemini 3.8 Flash | n/a | n/a | 89.3 | 100.0 | 95.9 |
| Claude Sonnet 5.5 | 100.0 | 100.0 | 95.3 | 100.0 | 100.0 |
| Claude Opus 5.5 | n/a | n/a | n/a | 100.0 | 98.3 |

| Model | GPQA-D /198 | Reasoning chars | MalayMMLU % | Tokens/answer |
|---|---:|---:|---:|---:|
| Lamba base | 151 | 8,900 | 83.60 | 369 |
| Coder390 base | 157 | n/a | 83.84 | n/a |
| Lamba 0.1 | 162 | 8,950 | 83.26 | 377 |
| Lamba 0.2 | 154 | 8,741 | 82.84 | 372 |
| Lamba 0.3 | 155 | 8,654 | 82.50 | 422 |
| **Lamba 0.4** | **160** | **7,260** | **84.62** | **278** |

</details>

## Layout

```
reports/lamba-0.1-0.2/   LaTeX source, figures and PDF of the 0.1/0.2 report
trainer/                 custom trainer (FTPO, DPO, soft-KL and KL-replay losses, datasets) and training configs
lamba-0.1-0.2/
  scripts/               loop mining, loop-test and tool-eval harnesses, frozen scoring scripts
  prompts/               g150 loop-test prompts and the eval30 tool tasks
lamba-0.4/
  training/              dataset builders, chain scripts and configs for the 0.4 rounds
  eval/                  0.4 evaluation scripts
  serving/               vLLM launchers
serving/                 chat server, hardened Python sandbox, constitution, vLLM launchers for 0.1/0.2
```

## Not included

Raw model outputs, training data, evaluation data from third-party benchmarks (for example GPQA question text),
model weights for 0.1 and 0.2, and internal working notes. Several scripts hard-code the original machine's paths
(`/home/ubuntu/...`) and expect those inputs, so they document the pipeline but will not run unmodified.

## Reproducing the evaluations

The metrics in the report are computed by the scripts in `lamba-0.1-0.2/scripts/` (see the Reproduction section of
the report): `loop_test_g.py` and `tail_stats_cmp.py` for loops, `loop_compare_v2.py` for the pre-registered
endpoint, `gen_sandboxb_eval.py` and `rumination_metric_v2_2_FROZEN.py` for the tool evaluation.

## Building the report

```
cd reports/lamba-0.1-0.2
tectonic -X compile main.tex
```

## License

Apache License 2.0 (see [`LICENSE`](LICENSE)). The trainer in `trainer/` is derived from Liquid AI's AntiDoom
(https://github.com/Liquid4All/antidoom, also Apache-2.0); see [`NOTICE`](NOTICE) for which files are unmodified,
modified or new.
