# State — 2026-08-17 (analysis complete)

Submission deadline: **August 20, 2026** (ICDM 2026 Teen Research Track).
Paper: `icdm2026_teen_submission/main.pdf` — **5 pages, no LaTeX errors, no
undefined references, zero overfull boxes, all data directives resolved, no
placeholders**. It is submittable as it stands.

## Runs on disk

84 records in `study2/results/`. Primary backbone (ViT-B/32): 48-run matched grid
(2 datasets x 2 methods x 4 LRs x 3 seeds), 6 linear probes, 6 last-block, and 6
frozen-projection runs. The ViT-B/16 extension contains 18 EuroSAT runs (2
methods x 3 learning rates x 3 seeds). Two additional single-seed interpolation
records are stored in `study2/intervention/`.

`analyze_study2.py` filters every primary statistic to ViT-B/32 and reports other
backbones in a separate section — without that filter the second backbone would
silently pool into the core grid and corrupt every number.

## Remaining before submission

1. Obtain a fresh independent human read of the final five-page build.
2. Recheck the official call for any deadline or formatting change.
3. Upload the verified PDF through the Teen Research Track submission system.

A replicated interpolation intervention would strengthen causal claims, but it
is correctly presented as future work rather than a submission blocker.

## Independent review outcome (first round)

Verdict: "significantly better than the track requires", recommend accept, high
confidence on soundness. It verified ~100 printed values against raw records and
found the arithmetic sound. Its findings, all of which I reproduced
independently and then fixed:

- bootstrap resampled runs not cells, inflating every interval — now resamples
  the 16 cell means, and the claim is softened to match (only the extremes
  separate at this sample size)
- "weight change is the most reliable within group" was false: signed dH is
  highest at 0.860, and log10(lr) alone reaches 0.810
- "does not separate the methods" was EuroSAT-only; on Pets p = 5.4e-5
- the decision test gated its candidate pool on *test* accuracy; now validation,
  which strengthened the result rather than weakening it
- an unreported 1k labelled CIFAR-100 probe at epoch 1 beats every label-free
  signal (rho = 0.967) — now reported as the upper bound
- retention percentage points were described as accuracy points
- printed `PLACEHOLDER-BACKBONE-NOTE` on page 5 — removed

## Thesis as it now stands

Attention drift is a **dose meter, not a damage meter**. Within one dataset,
method, and backbone, where learning rate is the principal varying factor,
signed entropy change strongly tracks eventual forgetting (mean |rho| 0.860 in
the primary grid). Across configurations, neither its strength nor polarity is
stable. Layerwise CKA and embedding drift provide the most consistent
cross-configuration diagnostics and remain informative after one epoch using a
few hundred unlabeled target images.

## Resume

```bash
USE_TF=0 python3 -m study2.analyze_study2   # tables, statistics, figures
USE_TF=0 python3 -m study2.fill_paper --audit
cd icdm2026_teen_submission && latexmk -pdf main.tex
```

`USE_TF=0` is mandatory: `transformers` otherwise imports TensorFlow, whose
native library deadlocks and hangs the pipeline with no output.

Nothing is committed to git; local `master` is 4 commits behind `origin/master`.
