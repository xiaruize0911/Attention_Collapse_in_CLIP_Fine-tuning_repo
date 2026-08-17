# Submission checklist — ICDM 2026 Teen Research Track

Deadline **August 20, 2026** (per the official call, checked 2026-08-16).
Notification September 18, 2026. Conference November 12-15, Shenyang.

## Eligibility and authorship

- [x] First author is a high school student at submission time
- [x] `High School Student` appears in the first author's affiliation
- [x] Author names visible (single-blind review)
- [ ] Confirm the first author is the primary contributor and can present
- [ ] If accepted: registration and accompanying adult guardian

## Format

- [x] IEEE Computer Society proceedings template (`IEEEtran`, conference mode)
- [x] Five pages including all figures, tables and references
- [x] Compiles with no LaTeX errors, no undefined references, no overfull boxes
- [x] Figures legible in greyscale and at print size

## Content integrity

- [x] Every number in the paper is generated from run records by
      `study2/fill_paper.py`; no hand-copied values
- [x] Target-task accuracy comes from a test split read once, at an epoch
      selected on a validation split carved out of training
- [x] Zero-shot transfer measured through the adapter-active encoder
- [x] Per-cell dispersion reported; method comparison uses paired tests with
      Holm correction
- [x] Study A is labelled as the weaker protocol and carries no accuracy claim
- [ ] Final build reports zero unresolved directives (run without `--draft`)
- [ ] `PLACEHOLDER-BACKBONE-NOTE` replaced once the ViT-B/16 runs finish
- [ ] Read the compiled PDF end to end once more before uploading

## Build

```bash
USE_TF=0 python3 -m study2.analyze_study2   # tables, statistics, figures
USE_TF=0 python3 -m study2.fill_paper       # renders main.tex (must report all resolved)
cd icdm2026_teen_submission && latexmk -pdf main.tex
```

## Submit

- [ ] Upload through the ICDM 2026 system under the Teen Research Track
- [ ] Recheck the call page for date or format changes before uploading
