---
title: "Draft and complete model card"
labels: ["documentation", "release"]
---

## Summary

Complete the model card in `MODEL_CARD_DRAFT.md` with actual training details, evaluation results, known limitations, and ethical caveats, once a trained model is ready for release. Replace all placeholder sections with verified content.

## Why It Matters

A complete and accurate model card is a standard requirement for responsible AI release. It communicates what the model does, how it was trained, what its limitations are, and when it should not be used. It is required for publishing on Hugging Face Hub and for any public GitHub release.

## Acceptance Criteria

- [ ] All placeholder sections in `MODEL_CARD_DRAFT.md` are replaced with real content
- [ ] Model overview table is filled with actual model name, architecture, and training status
- [ ] Training details (optimizer, learning rate, batch size, epochs, hardware) are documented
- [ ] Evaluation metrics table is filled with actual per-class results
- [ ] Limitations section is reviewed and updated based on failure case analysis
- [ ] Ethical and practical caveats section is confirmed accurate
- [ ] Dataset license and attribution section is completed (requires issue #14)
- [ ] Model card is reviewed before any public release

## Dependencies

- Issue #03 or #04: Training complete
- Issue #06: Evaluation metrics computed
- Issue #10: Failure case analysis complete
- Issue #14: License and attribution verified

## Notes

- Reference the [Hugging Face model card template](https://huggingface.co/docs/hub/model-cards) for formatting guidance
- The model card must not contain placeholder accuracy values or fake metrics
- Include both the binary and multiclass track results in the model card
