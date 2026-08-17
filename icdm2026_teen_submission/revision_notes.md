# Revision notes

How this version answers each finding in `independent_review.md`. Status is
`done`, `partial`, or `open`.

| # | Finding | Response | Status |
|---|---------|----------|--------|
| 1.1 | Model selection and reporting both used the provided test split | Study B carves validation out of the training split (`study2/data_splits.py`), selects the epoch on validation accuracy, and reads the test split exactly once per run. Study A is now explicitly labelled as having the weaker protocol, and no accuracy claim rests on it alone. | done |
| 1.2 | Primary evidence not reproducible; no dispersion or tests | Study B writes one JSON record per run including the full per-epoch history; `study2/analyze_study2.py` regenerates every table, statistic and figure. All Study B numbers carry standard deviations over seeds, and the Full FT vs LoRA comparison uses paired tests on identical (dataset, LR, seed) cells with Holm correction. | done |
| 1.3 | Transfer comparison confounded by the trained visual projection | Added a `lora_r8_frozen_proj` configuration that trains adapters and classifier only. The result reverses the confound: freezing the projection makes both axes worse at 1e-4 (retention 86.4% to ~77%, encoder weight drift 0.0042 to 0.0065), and at 1e-5 it leaves drift and retention untouched while target accuracy falls 78.8% to 22.6%. The projection is a cheap place to put task adaptation; deny it and the encoder pays. LoRA's advantage is not an artifact of what it leaves alone. | done |
| 1.4 | Matched learning rate is not a sufficient control | Added a best-validation ("iso-accuracy") comparison that reports transfer at each method's own best operating point, and a decision-oriented selection analysis restricted to configurations within 2 points of the best target accuracy. | done |
| 1.5 | No reference points bracketing the trade-off | Added linear probe (frozen encoder and projection) and last-block fine-tuning at multiple learning rates. | done |
| 1.6 | The diagnostic claim was never tested | The paper's central analysis is now a head-to-head ranking of label-free signals (attention entropy magnitude, ERF, Gini, CKA, embedding drift, relative weight change) against target-task loss and accuracy, evaluated by Spearman correlation with CIFAR-100 retention, within-group correlation, an epoch-1 early-warning variant, and a selection-utility test. The test refuted the original hypothesis: CKA (0.960) and embedding drift (-0.919) lead, the attention magnitudes tie with a weight-norm measurement that needs no images (-0.836 against -0.837), and the signed entropy change collapses to +0.361. | done |
| 1.7 | Attention measured on test-split images | The probe set is carved out of the training split and is disjoint from both the training subset and validation, and only its pixels are used. | done |
| 2 | Statistical reporting: layer-averaged $\Delta H$ hides layer-12 behaviour; single-seed auxiliary claims; unreported ERF | The main table reports both the layer-averaged and the block-12 entropy change; the per-layer heatmap is generated from the new runs; ERF and Gini appear in the predictor table; single-seed auxiliary claims from the older exploratory suite are dropped. | done |
| 3 | Thin related work | Bibliography rebuilt around the closest work: attention-entropy collapse, attention sinks, ViT registers, FLYP, task arithmetic, prompt learning, the projection norm, and the corruption benchmark. 11 entries before, 19 now after pruning citations that supported no claim, all in compact IEEE form to fit the page limit. | done |
| 4 | Two of five pages on side experiments and a narrated past bug | Sections V and VII of the previous version are removed. The adapter-aware evaluation point survives as two sentences in the method section, where it belongs. | done |
| 5.4 | Only one transfer axis | Added CIFAR-10 zero-shot and the seven corrupted EuroSAT test splits. | done |
| 6 | README deadline wrong | Corrected to August 20, 2026, with the source URL and the other venue facts. | done |
| -- | Second backbone (ViT-B/16) | Six runs queued (EuroSAT, three learning rates). The analysis now records `model_name` per run and filters every primary statistic to ViT-B/32, with a separate digest section per additional backbone --- without that filter the second backbone would have been silently pooled into the core grid. | in progress |
| -- | Causal claim | `study2/run_intervention.py` sweeps encoder interpolation after fine-tuning to test whether the predicted trade-off can be acted on. Inclusion depends on space. | open |

## Claims corrected against the completed data

Two statements in an intermediate draft did not survive the full 48-run grid and
were rewritten rather than kept:

- "retention falls monotonically for both methods and both datasets" is false for
  LoRA on Pets (Spearman -0.32, p = 0.3), which forgets almost nothing anywhere
  in the grid. The paper now reports three of four groups at -0.97 and names the
  exception.
- "the attention statistics are a clear tier below" is false at n = 48: the
  magnitude-based attention signals tie with relative weight change. The paper
  now separates the leading pair (CKA, embedding drift) from a middle tier and
  reserves the strong claim for the *signed* statistic.
- The paired bootstrap originally resampled *runs*, which treats seeds within a
  configuration as independent evidence and shrinks every interval. It now
  resamples the 16 cell means, and the conclusion changed: CKA's |rho| advantage
  over embedding drift, |dH| and the weight norm no longer excludes zero
  (+0.144 [-0.018, +0.424] for the weight norm). Only its advantage over signed
  dH and |dH_12| survives. The paper states this rather than the earlier,
  stronger version.
- The selection test is dataset-dependent: attention drift picks a 24.0%
  retention run on EuroSAT where a random pick averages 46.4%, but on Pets every
  signal picks well. Stated as such, in the abstract and the conclusion too.

## Second independent review

Verdict: again "significantly better than the track requires", accept, high
confidence. Its findings, each reproduced independently before fixing:

- **"CKA is +1.00 in all four" was false.** The sentence used cell means while
  the paper's within-group convention is run level, where CKA is +0.944, +0.888,
  +1.000, +1.000. The abstract already stated the correct weaker version, so the
  paper contradicted itself on its headline claim. Now "positive in all four
  (+0.89 to +1.00)", computed by a directive rather than hard-coded --- being
  hard-coded is why the audit trail did not catch it.
- **The Discussion still carried the retracted weight-norm superlative** that the
  first review flagged. Fixing it in one section and not the other is how a
  correction leaves a twin behind.
- **Fig. 1's caption asserted two things its own figure refutes** (LoRA entropy
  rising, retention monotone for both --- neither true on Pets), a claim these
  notes had already recorded as corrected in the body.
- "almost perfectly" overstated: one of four groups sits at rho = 0.558, and it
  is the group with almost no forgetting to track (97.7-100.8% retention).
- The Conclusion's "only signals that work in both regimes" is contradicted by
  the weight norm, whose polarity is also portable; what disqualifies it is the
  decision test.
- The epoch-1 separation was reported over 17 vs 31 *runs* two paragraphs after
  arguing runs are not independent. Now 6 vs 10 *configurations*.
- The AUC is oriented with `max(raw, 1-raw)`, i.e. its direction is chosen using
  the outcome; now disclosed as an upper bound.
- "first four blocks within 1.5%" was false (1.79% max); the labelled probe is a
  *subset* of the evaluation set, so its rho is partly self-correlation;
  last-block also trains the output layer norm; Sec. III-A's R^49 does not
  describe the ViT-B/16 arm; Sec. III-D described the decision gate as target
  accuracy where the code uses validation.
- The ViT-B/16 arm was single-seed with n = 3 per group, where |rho| = 1 arises
  from 1 in 3 permutations under the null. Two further seeds are running.

To pay for these in five pages, the corrupted-EuroSAT axis was cut from the paper
(it remains in the repository). Shrinking the figures further would have put axis
labels below print legibility.

## Environment note

`import peft` deadlocks on this machine because `transformers` imports
TensorFlow, whose native library hangs during `preload_check`. Every entry point
sets `USE_TF=0`; without it the pipeline appears to hang indefinitely.
