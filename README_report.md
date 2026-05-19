# HW2 Report Build Notes

The report files are:

- `report.tex`
- `references.bib`
- `outputs/summary_results.csv`
- `tables/*.tex`
- `figures/*.png`

The summary CSV and LaTeX tables were generated from the real experiment outputs with:

```bash
python scripts/summarize_results.py
```

Compile the report from the `hw2` directory:

```bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

If `latexmk` is available:

```bash
latexmk -pdf report.tex
```

The report uses the NeurIPS 2026 style file in:

```text
Formatting_Instructions_For_NeurIPS_2026/neurips_2026.sty
```

