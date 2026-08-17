ASSET_SCRIPT=python scripts/build_assets.py
LATEX=pdflatex
BIBTEX=bibtex

.PHONY: assets pdf clean

assets:
	$(ASSET_SCRIPT)

pdf: assets
	$(LATEX) main.tex
	$(BIBTEX) main
	$(LATEX) main.tex
	$(LATEX) main.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot
