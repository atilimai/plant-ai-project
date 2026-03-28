---
title: "Final release checklist execution"
labels: ["release"]
---

## Summary

Work through every item in `RELEASE_CHECKLIST.md` and confirm completion before tagging a public release. This issue serves as the final gate that must be closed before any public GitHub or Hugging Face release.

## Why It Matters

A structured release gate prevents incomplete, inaccurate, or license-violating artifacts from being published. It ensures that all evaluation, documentation, and legal review tasks are complete before the project is made public.

## Acceptance Criteria

- [ ] Every item in `RELEASE_CHECKLIST.md` is checked off
- [ ] Dataset provenance and license are confirmed (requires issue #13)
- [ ] No train/test leakage confirmed via `leaf_id` audit (requires issue #02)
- [ ] Final confusion matrices and per-class metrics are exported (requires issues #06, #07)
- [ ] Sample predictions and failure cases are documented (requires issues #09, #10)
- [ ] Grad-CAM visualizations are exported (requires issue #08)
- [ ] Model card is complete and reviewed (requires issue #12)
- [ ] Hugging Face release is ready (requires issue #14)
- [ ] No broken links or placeholder text remaining in documentation
- [ ] Repository is tagged with a release version

## Dependencies

All previous issues — this is the final release gate.

Specifically:
- Issue #02: Leakage audit complete
- Issue #06: Evaluation artifacts complete
- Issue #07, #08, #09, #10: Visualization and analysis complete
- Issue #12: Model card complete
- Issue #13: License verified
- Issue #14: Hugging Face packaging complete

## Notes

- Tag the release only after this issue is closed
- Release tag format: `v1.0.0`
- Include a release changelog in the GitHub release notes
- Announce the Hugging Face link in the GitHub README after release
