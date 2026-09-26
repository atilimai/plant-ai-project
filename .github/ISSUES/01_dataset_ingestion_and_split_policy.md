---
title: "Dataset ingestion and split policy"
labels: ["task"]
---

## Summary

Download the PlantVillage dataset from its official source, verify integrity, and implement a reproducible ingestion pipeline that stores raw data in `data/raw/` and produces processed outputs in `data/processed/`. Define and document the official split policy for this project.

## Why It Matters

All downstream training and evaluation depends on having a correctly structured and consistently split dataset. Without a reliable ingestion pipeline, results will not be reproducible across runs or environments.

## Acceptance Criteria

- [x] Dataset downloaded from a documented, versioned source
- [x] Download URL and file checksum recorded in `DATASET_NOTES.md`
- [x] Raw images stored in `data/raw/` (not committed to git; path documented)
- [x] Class distribution documented (number of images per class, total count)
- [x] Processed dataset (resized, normalized as needed) stored in `data/processed/`
- [x] Official split policy documented in `data/splits/README.md`
- [x] Split generation code or instructions reference `leaf_id` constraint (see issue #02)

## Dependencies

- None (first task)

## Notes

- Do not commit raw images to the repository
- Consider using Hugging Face Datasets or Kaggle API for reproducible downloads
- Record the exact dataset version and any preprocessing steps applied
- See `DATASET_NOTES.md` for background on dataset limitations

---

## Status

Done. `python -m src.data.prepare` downloads `data.zip` at a pinned revision, verifies both checksums, extracts colour and segmented variants and writes `data/splits/`. Source, checksums and statistics: `DATASET_NOTES.md`. Images stay out of git; processed copies are not written at all, preprocessing happens at load time.
