# Contributing to Plant Disease Classification

Thank you for your interest in contributing to this project.
This document describes how to contribute effectively and respectfully.

---

## How to Contribute

### Reporting Issues
- Check existing issues before opening a new one to avoid duplicates
- Use the appropriate issue template from `.github/ISSUE_TEMPLATE/`
- Be specific: include steps to reproduce, expected behavior, and actual behavior for bugs

### Submitting Changes
1. Fork the repository
2. Create a feature branch from `main`: `git checkout -b feature/your-description`
3. Make your changes with clear, focused commits
4. Ensure your changes do not break existing tests
5. Open a pull request with a descriptive title and summary

### Pull Request Guidelines
- Keep pull requests focused on a single concern
- Write clear commit messages
- Reference related issues in the PR description (e.g., `Closes #12`)
- Do not include unrelated changes

---

## Code Style

- Python: follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints where practical
- Write docstrings for public functions and classes
- Keep functions small and single-purpose

---

## Research Integrity Constraints

- **Do not submit split files that violate `leaf_id` integrity.** All train/val/test splits must respect `leaf_id` boundaries to prevent data leakage. See `DATASET_NOTES.md` for details.
- **Do not report or commit model performance metrics unless they are computed on a properly leakage-free test set.**
- **Do not commit dataset files directly to the repository.** Store paths, download scripts, or manifests only.

---

## License and Attribution

Before contributing artifacts that include dataset-derived outputs (model weights, visualizations of dataset images, evaluation results):
- Verify that the PlantVillage dataset license permits the intended use and redistribution
- See `LICENSE_PLACEHOLDER.md` for the current license status

---

## Questions

Open a discussion issue or a `documentation` type issue using the issue templates.
