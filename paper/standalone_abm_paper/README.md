# Paper 2: Are Contagion Hubs Causal? A Calibrated Counterfactual Test

**Compiled PDF: `main.pdf`** (built locally with TeX Live 2026 / `acmart`, 0 errors).

Self-contained: `main.tex` + `figures/` + `references.bib`. ACM `sigconf` (`nonacm` draft mode).

## Build
```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
or upload this folder to Overleaf.

Takes the GNN hub ranking (Paper 1) as input. Reproduce numbers with `../../reproduce.sh`
(run the GNN repo's `reproduce.sh` first).
