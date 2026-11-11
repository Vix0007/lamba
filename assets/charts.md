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
