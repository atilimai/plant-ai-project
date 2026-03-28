---
title: "Failure case analysis report"
labels: ["research"]
---

## Summary

Conduct a systematic failure case analysis to identify and document patterns in model errors. Produce a written report and a notebook that categorizes failure modes and links them to potential root causes (class confusion, background artifacts, ambiguous images, underrepresented classes, etc.).

## Why It Matters

Understanding *how* the model fails is as important as knowing its aggregate accuracy. Failure case analysis reveals actionable improvement directions, identifies potential biases, and is a responsible practice required before any public release of model predictions.

## Acceptance Criteria

- [ ] All test set misclassifications are collected and analyzed
- [ ] Failure cases are grouped into meaningful categories (e.g., confused class pairs, low-confidence errors, ambiguous images)
- [ ] Root cause hypotheses are documented for each failure category
- [ ] Grad-CAM visualizations are included for representative failure cases
- [ ] Quantitative breakdown of failure rates by class or category is provided
- [ ] Report exported to `artifacts/reports/`
- [ ] Analysis is presented in a Colab-compatible notebook

## Dependencies

- Issue #06: Evaluation pipeline to collect predictions and errors
- Issue #07: Confusion matrix to identify top confused class pairs
- Issue #08: Grad-CAM to visualize failure cases

## Notes

- Focus on systematic patterns, not individual outliers
- Consider whether failures are concentrated in specific plant species or disease categories
- Document any failure modes that suggest the model is using background cues rather than leaf features
- This analysis informs the limitations section of the model card
