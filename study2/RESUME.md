# State — 2026-08-17 (loop running)

Submission deadline: **August 20, 2026** (ICDM 2026 Teen Research Track).
Paper: `icdm2026_teen_submission/main.pdf` — **5 pages, no LaTeX errors, no
undefined references, zero overfull boxes, all data directives resolved, no
placeholders**. It is submittable as it stands.

## Runs on disk

69 records in `study2/results/`. Primary backbone (ViT-B/32): 48-run matched grid
(2 datasets x 2 methods x 4 LRs x 3 seeds), 6 linear probes, 6 last-block, 6
frozen-projection. Second backbone (ViT-B/16): 3 of 6 done, EuroSAT only, one
seed, 3 learning rates.

`analyze_study2.py` filters every primary statistic to ViT-B/32 and reports other
backbones in a separate section — without that filter the second backbone would
silently pool into the core grid and corrupt every number.

## Still open

1. **3 ViT-B/16 runs** (~1.5 h at 30-49 min each). When they land, replace the
   preliminary single-seed sentence in Limitations with a proper backbone
   paragraph. Needs ~40 words more than the current sentence, so trim to match.
2. **Cross-family predictor comparison** (reviewer item 12, no GPU): including the
   18 reference runs, |dH| rises 0.836 -> 0.867 while CKA falls 0.960 -> 0.955.
   State honestly that part of |dH|'s gain is the linear probe anchoring the
   correlation at exactly zero drift / 100% retention.
3. **Fresh independent review.** The review in hand was written against a version
   that has since changed materially (thesis reframed, statistics corrected).

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

Attention drift is a **dose meter, not a damage meter**. Within one dataset and
method, where only the learning rate varies, the signed entropy change is the
best predictor of eventual forgetting in the study (mean |rho| 0.860). Across
configurations its sign is not constant — it flips between methods, between
datasets, and between backbones — so it drops to mid-pack, and choosing a
configuration by it can be worse than choosing at random. Layerwise CKA and
embedding drift are the only signals that work in both regimes, after a single
epoch, from a few hundred unlabeled target images.

## Resume

```bash
bash study2/run_campaign4.sh                # finishes ViT-B/16, then intervention
USE_TF=0 python3 -m study2.analyze_study2   # tables, statistics, figures
USE_TF=0 python3 -m study2.fill_paper --audit
cd icdm2026_teen_submission && latexmk -pdf main.tex
```

`USE_TF=0` is mandatory: `transformers` otherwise imports TensorFlow, whose
native library deadlocks and hangs the pipeline with no output.

Nothing is committed to git; local `master` is 4 commits behind `origin/master`.
