---
title: "License and attribution review"
labels: ["release", "research"]
---

## Summary

Conduct a formal review of the PlantVillage dataset license and all other third-party licenses used in this project. Document findings in `LICENSE_PLACEHOLDER.md` and `CITATION.md`. This is a required release gate item.

## Why It Matters

Releasing model weights or dataset-derived artifacts without confirming licensing terms could constitute a license violation. This review is a non-negotiable prerequisite for any public GitHub or Hugging Face release.

## Acceptance Criteria

- [ ] PlantVillage dataset license and terms of use are reviewed and documented
- [ ] Determination made on whether public release of model weights is permitted
- [ ] Determination made on whether preprocessed dataset splits can be shared or must be excluded
- [ ] Determination made on whether Hugging Face Hub hosting is permitted
- [ ] Required attribution text identified and added to `CITATION.md`
- [ ] A `LICENSE` file is added to the repository with the chosen code license
- [ ] `LICENSE_PLACEHOLDER.md` is updated to reflect the completed review
- [ ] All findings are documented before any public artifacts are released

## Dependencies

- None — this can be started in parallel with early development tasks

## Notes

- Primary reference: [PlantVillage project homepage](https://plantvillage.psu.edu/) and the associated paper (Hughes & Salathé, 2015)
- Check Kaggle dataset page for additional license notes if downloading from Kaggle
- Check Hugging Face Datasets page if accessing via `datasets` library
- If redistribution is not permitted, model weights and data splits must be excluded from the public release
- This issue blocks the final release checklist
