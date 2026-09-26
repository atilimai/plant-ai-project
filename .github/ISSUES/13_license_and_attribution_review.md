---
title: "License and attribution review"
labels: ["release", "research"]
---

## Summary

Conduct a formal review of the PlantVillage dataset license and all other third-party licenses used in this project. Document findings in `LICENSE_PLACEHOLDER.md` and `CITATION.md`. This is a required release gate item.

## Why It Matters

Releasing model weights or dataset-derived artifacts without confirming licensing terms could constitute a license violation. This review is a non-negotiable prerequisite for any public GitHub or Hugging Face release.

## Acceptance Criteria

- [x] PlantVillage dataset license and terms of use are reviewed and documented
- [x] Determination made on whether public release of model weights is permitted
- [x] Determination made on whether preprocessed dataset splits can be shared or must be excluded
- [x] Determination made on whether Hugging Face Hub hosting is permitted
- [x] Required attribution text identified and added to `CITATION.md`
- [x] A `LICENSE` file is added to the repository with the chosen code license
- [x] `LICENSE_PLACEHOLDER.md` is updated to reflect the completed review
- [x] All findings are documented before any public artifacts are released

## Dependencies

- None — this can be started in parallel with early development tasks

## Notes

- Primary reference: [PlantVillage project homepage](https://plantvillage.psu.edu/) and the associated paper (Hughes & Salathé, 2015)
- Check Kaggle dataset page for additional license notes if downloading from Kaggle
- Check Hugging Face Datasets page if accessing via `datasets` library
- If redistribution is not permitted, model weights and data splits must be excluded from the public release
- This issue blocks the final release checklist

---

## Status

Done. `LICENSING.md`: dataset is CC BY-SA 3.0 and the authors state that algorithms trained on it fall under the same licence, quoted from the original paper; weights and manifests are released as CC BY-SA 3.0, code stays GPL-2.0. The earlier claim of CC BY-NC-SA was wrong and is corrected. The ImageNet-pretrained backbone caveat is documented too.
