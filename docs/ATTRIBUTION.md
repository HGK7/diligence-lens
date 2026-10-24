# Third-party attribution

Diligence Lens may index public research corpora. Keep this file when you
redistribute the project or publish demos that include those materials.

## CUAD — Contract Understanding Atticus Dataset

| Field | Value |
|-------|--------|
| Full name | Contract Understanding Atticus Dataset (CUAD) |
| Curator | The Atticus Project |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Homepage | https://www.atticusprojectai.org/cuad/ |
| GitHub (code / data.zip) | https://github.com/The-Atticus-Project/cuad |
| Hugging Face mirror | https://huggingface.co/datasets/theatticusproject/cuad |
| Paper | https://arxiv.org/abs/2103.06268 |

### Required citation

If you use CUAD (including a subset copied into `dataroom/`), cite:

```bibtex
@article{hendrycks2021cuad,
  title={CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author={Dan Hendrycks and Collin Burns and Anya Chen and Spencer Ball},
  journal={NeurIPS},
  year={2021}
}
```

### How this project uses CUAD

1. You download the full CUAD distribution into `datasets/cuad/` (git clone or `data.zip`).
2. `scripts/export_cuad.py` copies a **subset** of `.txt` contracts into `dataroom/`.
3. Diligence Lens embeds and retrieves those files locally. It does **not** claim ownership of CUAD.

Each export also writes `dataroom/MANIFEST.json` with source path, seed, limit, and citation metadata. Do not delete that file when sharing a frozen demo corpus.

### License obligations (CC BY 4.0 summary)

You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may use CUAD commercially under CC BY 4.0, but this project’s own code may use a different license — see the repository root `LICENSE` when published.

Full legal text: https://creativecommons.org/licenses/by/4.0/

## Other optional corpora

If you later add ContractNLI, AcmeTech, or other packs, append a section here with license, citation, and how the files enter `dataroom/`.

## Project-generated content

Synthetic Nextwork seed docs (if present under `vendor_docs/`), Streamlit UI copy, LangGraph prompts, and Docker/config files are part of this repository unless marked otherwise.
