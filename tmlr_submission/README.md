# CLIP Attention Collapse Paper

This folder contains a standalone LaTeX paper built from the verified experiment artifacts in `../Attention_Collapse_in_CLIP_Fine-tuning_repo`.

## What is here

- `main.tex` — the manuscript
- `references.bib` — bibliography used by the manuscript
- `scripts/build_assets.py` — copies regenerated figures and builds LaTeX tables directly from JSON metrics
- `tables/` — generated LaTeX tables
- `figures/` — copied figures used by the manuscript
- `paper_assets.json` — machine-readable summary of the exact metrics included in the paper

## Rebuild the paper assets

Run this from this folder:

`python scripts/build_assets.py`

## Compile the PDF

If `pdflatex` and `bibtex` are available:

`pdflatex main.tex`

`bibtex main`

`pdflatex main.tex`

`pdflatex main.tex`

## Important caveat

The manuscript explicitly documents that the repository's current zero-shot evaluation path does not establish adapter-active LoRA zero-shot behavior. Please keep that caveat if you extend or submit the paper.
