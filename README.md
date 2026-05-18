# FusionTrack

FusionTrack collects the research materials for multimodal target fusion,
trajectory completion, and behavior analysis. The repository currently contains
the manuscript source, visualization outputs, and placeholders for project code.

## Repository Layout

- `article_content/`: LaTeX source for the thesis manuscript, including chapter
  files, bibliography, required figures, fonts, and the local `latexmkrc`.
- `.github/workflows/build-article.yml`: GitHub Actions workflow that compiles
  `article_content/main.tex` and uploads the generated PDF as an artifact.
- `visualization_results/`: exported figures and demonstration videos used to
  show system behavior and analysis results.
- `code/`: project source-code directory. It includes algorithm code organized
  by task, including VPR, registration, and anomaly-detection modules.

## Build the Article Locally

The manuscript is built with XeLaTeX through `latexmk`.

```bash
cd article_content
latexmk -xelatex -interaction=nonstopmode -file-line-error main.tex
```

The generated PDF is `article_content/main.pdf`. Build products such as
`main.pdf`, `.aux`, `.log`, `.xdv`, and SyncTeX files should remain uncommitted.
Use the following command to clean generated files:

```bash
cd article_content
latexmk -C
```

## GitHub Actions PDF

On pushes or pull requests that change the manuscript source, GitHub Actions
runs the `Build article PDF` workflow. The compiled PDF is uploaded as the
`article-pdf` artifact on the workflow run page instead of being stored in the
repository.

## Notes

- Keep manuscript figures under the matching chapter folder, for example
  `article_content/images/chapter3/` for figures used in Chapter 3.
- Commit only source files and required static assets. Generated build outputs
  should be reproduced locally or downloaded from GitHub Actions artifacts.
