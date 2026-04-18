# License and Attribution Review (Completed - Updated)

> **Review Status: Verified and Approved**
> The formal license review (Issue #15) has been updated to reflect the `mohanty/PlantVillage` dataset license from Hugging Face.

---

## License Determinations

### 1. Code License
The repository source code is officially licensed under the **GNU General Public License v2.0 (GPLv2)**.

### 2. Dataset License (PlantVillage - Hugging Face version)
The specific `mohanty/PlantVillage` dataset is licensed under **CC-BY-SA 3.0 (Creative Commons Attribution-ShareAlike 3.0)**.
- **Status:** Cleared for use. Redistribution or derivative works must give appropriate credit and be distributed under the same CC-BY-SA 3.0 license.

### 3. Model Weights & Data Splits
Because the underlying dataset is CC-BY-SA 3.0:
- **Model Weights:** Public release is **permitted**, but weights MUST be released under the **CC-BY-SA 3.0** license (as they are a derivative of the data).
- **Preprocessed Splits:** Redistribution is **permitted**, but MUST also be licensed under **CC-BY-SA 3.0**.

---

## Verification: GPLv2 and CC-BY-SA 3.0 Compatibility
As requested, a compatibility check has been performed:
- **Result:** **Compatible.** - **Reasoning:** In an ML project, the training code (licensed under GPLv2) and the dataset/model weights (licensed under CC-BY-SA 3.0) operate as separate artifacts. The code processes the data but is not a derivative work of the dataset. Distributing both in the same repository constitutes an "aggregate" rather than a single merged derivative work. Therefore, neither license's ShareAlike/Copyleft restrictions violate the other.

---

## Final Release Gate Status

- [x] PlantVillage dataset license updated to `CC-BY-SA 3.0`
- [x] Code (GPLv2) vs. Data (CC-BY-SA 3.0) compatibility verified
- [x] Public release conditions for model weights determined
- [x] `CITATION.md` created with required attribution
