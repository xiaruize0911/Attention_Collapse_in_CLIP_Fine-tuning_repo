# ICDM 2026 Teen Research Track submission

Five-page, single-blind IEEE conference paper on what can be measured during
CLIP fine-tuning, and which of those measurements predicts zero-shot forgetting.

## Build

Run `latexmk -pdf main.tex` in this folder. The paper uses the standard
`IEEEtran` conference class distributed with TeX Live. To regenerate `main.tex`
from the experiment records first, see "Experiment provenance" below.

## Venue facts (verified against the official call on 2026-08-16)

- Paper submission deadline: **August 20, 2026** (tentative, per the call page)
- Notification: September 18, 2026; conference November 12-15, 2026, Shenyang
- Up to 5 pages including all figures, tables, and references
- IEEE Computer Society proceedings format; single-blind review
- `High School Student` must appear in the first author's affiliation
- Submit through the ICDM 2026 system under the Teen Research Track
- Source: <http://icdm2026.neu.edu.cn/11673/list.htm>

## Experiment provenance

**Study B (primary, fully reproducible).** Everything under `../study2/`:

```bash
bash study2/run_campaign.sh                 # core grid + reference configurations
bash study2/run_campaign4.sh                # backbone check and intervention
USE_TF=0 python3 -m study2.analyze_study2   # tables, statistics, figures
USE_TF=0 python3 -m study2.fill_paper       # renders main.tex from main_template.tex
```

`USE_TF=0` is required on this machine: `transformers` otherwise imports
TensorFlow, whose native library deadlocks and hangs the pipeline.

`main.tex` is generated. Edit `main_template.tex`, whose directives
(`@cell(...)@`, `@pred(...)@`, `@paired(...)@`, ...) are resolved against the run
records, so no number in the paper is transcribed by hand. A directive that
cannot be resolved is reported and left in place rather than silently filled.

Per-run records live in `study2/results/*.json` (one file per run, including the
per-epoch history) and the aggregate statistics in
`study2/analysis/summary.json`. Two run-level tables are written:
`run_level.csv` holds the primary backbone (ViT-B/32) only and is what every
number in the paper is computed from; `run_level_all_backbones.csv` adds the
ViT-B/16 generality runs. They are separate on purpose --- averaging the two
architectures together silently changes, for example, EuroSAT Full FT at
`1e-4` from a mean entropy drift of `-2.34%` to `-0.80%`. All runs were executed
on one Apple M4 laptop GPU.

**Study A (large grid).** The 80-run matched-learning-rate matrix on complete
training splits, summarised in `../controlled_heatmap_drift_report.md` on
`origin/master` and in `controlled_results.csv` here. It is reported as
cell means over five seeds; its raw per-run logs were produced on a different
machine and are not part of this repository, which is one reason Study B ships
its raw records.

## Files

- `main.tex`, `references.bib` — the submission
- `figures/` — figures used by the paper
- `independent_review.md` — the reviewer-style critique this revision responds to
- `revision_notes.md` — what changed and why, keyed to that critique
- `controlled_results.csv` — Study A cell means
- `submission_checklist.md` — pre-submission checks
- `main_v1_backup.tex` — the reviewed version, kept for diffing
- `smoke_test_results.md` — historical: short verification runs from 2026-08-13,
  superseded by Study B and not cited by the paper
