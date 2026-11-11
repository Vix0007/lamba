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
