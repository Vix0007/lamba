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
