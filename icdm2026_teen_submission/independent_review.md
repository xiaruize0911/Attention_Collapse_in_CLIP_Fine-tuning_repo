# Independent evaluation of the ICDM 2026 Teen Track submission

Reviewed artifact: `icdm2026_teen_submission/main.tex` (5 pages, IEEEtran, single-blind),
plus the code and reports it depends on (`run_all_experiments.py`, `src/`,
`analyze_icdm2026_results.py`, `controlled_heatmap_drift_report.md` on `origin/master`).

Venue facts that set the bar (from the official call, fetched 2026-08-16):
Teen Research Symposium, up to 5 pages including references, IEEE CS proceedings
format, single-blind, **paper deadline August 20, 2026** (the folder's README says
August 30 — that is wrong and is the single most time-critical error in the
submission package). Stated review criteria: *technical soundness, originality,
relevance, and clarity, with expectations appropriate for high school-level
research.*

Overall assessment: the paper is above the typical bar for this track — a real
controlled design, honest limitations, careful language about causality. The
risk to acceptance is not ambition, it is **verifiability and framing**: the
central table cannot be audited, the accuracy protocol has a known leak, the
"drift is a useful diagnostic" claim is asserted rather than tested, and a full
page is spent on heterogeneous side experiments and on narrating a previous bug.

Findings are ordered by how much they would move a reviewer score.

---

## 1. Blocking-level soundness issues

### 1.1 Model selection and reporting both use the provided test split
`run_all_experiments.py` passes the test split as `val_loader`
(`run_full_ft_experiment`, line 442: `val_loader = get_dataloader(test_dataset, ...)`),
saves `best_model.pth` on the best *test* accuracy, and reports `best_val_acc`.
The transfer number is then measured from that test-selected checkpoint. So

* every target-accuracy number in Table I is a maximum over 20-30 test
  evaluations, not a generalization estimate;
* the LoRA/Full-FT transfer comparison uses checkpoints chosen by a criterion
  that is correlated with the quantity under comparison.

The paper discloses this in one sentence in the limitations. Disclosure is not a
fix: a reviewer who reads the code will treat every accuracy in the paper as
optimistically biased by an unknown amount.

### 1.2 The primary evidence is not reproducible from the repository
The 80-run matrix exists only as *cell means* in
`controlled_heatmap_drift_report.md` and `controlled_results.csv`. The raw
artifacts the report cites (`outputs/controlled_heatmap_drift/run_summaries.csv`,
`run_summaries.json`, `analysis_summary.json`) are not in the repository
(`outputs/` is git-ignored) and are not on this machine —
`outputs/icdm2026/analysis_summary.json` currently reports
`"completed_runs": 0`. Consequences:

* no per-seed values, so no confidence intervals, no significance tests, and no
  way to recompute the quoted run-level correlations (0.61, 0.90);
* Table I reports differences as small as 0.12 percentage points of entropy with
  no error term;
* the text says "table entries summarize five runs per cell using their mean and
  across-seed standard deviation", but no standard deviation appears in the table.

### 1.3 The transfer comparison is confounded by the trained visual projection
Both methods train `visual_projection` (`src/model.py:74-76` for LoRA). Zero-shot
transfer is computed *through* that projection. A large share of the reported
transfer loss may therefore come from the projection, not from the visual
encoder — which is exactly the component the attention story is about. Untested,
and cheap to test.

### 1.4 "Matched learning rate" is a real control but not a sufficient one
LoRA at learning rate `x` and Full FT at learning rate `x` do not take
equally-sized steps in function space (zero-initialized `B`, `alpha/r` scaling,
0.05% of the parameters). The paper's central comparison therefore still mixes
*how much the model learned* with *how much damage it took*; the Pets
`1e-6` cell (19.13% target accuracy, ~0% drift) is the paper's own proof of the
problem. The design needs a second alignment — same target accuracy, or same
weight drift — before "LoRA is more conservative" is separable from "LoRA moved
less".

### 1.5 No reference points bracketing the trade-off
There is no linear probe (zero encoder drift, zero forgetting, some target
accuracy) and no partial fine-tuning (e.g. last block). Without them, the claim
that adequately-tuned LoRA "occupies the desirable region more often" has no
upper or lower bound to be compared against. A reviewer will ask whether a
linear probe simply dominates on this axis.

### 1.6 The diagnostic claim is never tested
The paper's most useful idea — attention drift as an early warning for
forgetting — is argued from two dataset-level correlations quoted from the
report. It is not tested against the obvious cheaper competitors:
relative weight change, embedding/feature drift, CKA, or simply the target-task
loss. Nor is it tested *predictively* (does the signal at epoch 1-2 predict the
final transfer loss?). As written, the "practical diagnostic" section is advice
rather than a result.

### 1.7 Attention is measured on test-split images
`create_fixed_eval_subset(test_dataset, ...)` draws the 200 structural-analysis
images from the target test split. Minor, but it means the structural
measurement is not something a practitioner could compute before deployment,
and it touches the same split used for selection.

---

## 2. Statistical reporting

* `\Delta H` is a percentage change of a mean over 12 layers x 12 heads x 200
  images. Section IV-C shows layer 12 moving `-20.29%` while the reported mean is
  `-3.63%`, i.e. the headline statistic is dominated by cancellation across
  layers. The paper should report the layer-resolved statistic as primary, or at
  least give both with dispersion.
* Auxiliary claims rest on `n=5` single-seed points (Spearman `rho = -1.0`,
  `p = 0.0167`). Fine as a trend note, but the paper leans on it as
  "dose-response".
* The regularizer paragraph reports effects that admittedly fail
  Holm-Bonferroni. Either drop it or state the correction and the resulting
  non-conclusion in one clause.
* ERF@0.95 and Gini are defined and claimed in the abstract but never tabulated
  for the primary matrix.

---

## 3. Positioning and related work

Nine references is thin for ICDM, and the closest work is missing:

* attention-entropy collapse as a *training-stability* phenomenon
  (Zhai et al., ICML 2023) — the paper's mechanism, in a different context;
* attention sinks / high-norm tokens and ViT registers
  (Xiao et al. 2024; Darcet et al. 2024) — these explain why CLS attention mass
  concentrates and why entropy is a fragile summary;
* robust fine-tuning beyond WiSE-FT: LP-FT is cited, but FLYP, model soups, and
  the CLIP-forgetting literature are not;
* CKA is used and cited, but no attention-analysis-specific critique beyond
  Jain & Wallace / Wiegreffe & Pinter.

Without these, "attention drift" reads as an unrecognized rediscovery, and a
reviewer familiar with Zhai et al. may say so.

---

## 4. Presentation

* Roughly two of five pages go to Section V (auxiliary experiments from an
  earlier suite) and Section VII (reproducibility notes). Section V explicitly
  narrates a previous evaluation bug ("silently leaving LoRA adapters inactive").
  Honesty is good; a 5-page venue paper is the wrong place to relitigate it.
  Compress to a two-sentence protocol note.
* Table I is a 12-column double panel with no dispersion; Table II packs two
  unrelated panels into one float.
* Figure 2 is 55% text width and carries little annotation for the space.
* The abstract promises ERF, the tables do not deliver it.
* `smoke_test_results.md` documents runs that "are not used as evidence" — good
  practice, but the submission package should not ship contradicting artifacts
  (the README points reviewers to a report file that is not in the tree).

---

## 5. What would most raise the score, in order

1. Re-run the grid with a leakage-free protocol (validation carved from train,
   test read once) and ship the per-run records, so every number in the paper is
   auditable and comes with dispersion.
2. Turn Section VI's advice into a result: test whether attention drift predicts
   forgetting, and whether it beats weight drift / embedding drift / target loss,
   including an epoch-1 early-warning test.
3. Add the two missing reference points (linear probe, last-block) and the
   frozen-projection LoRA ablation that isolates the encoder.
4. Add a second transfer axis (a second zero-shot benchmark and/or corrupted
   target-domain data) so "forgetting" is not one number on one dataset.
5. Rewrite Sections V and VII into a half page; spend the reclaimed space on
   related work and on the predictive analysis.
6. Fix the README deadline (August 20, not August 30).
