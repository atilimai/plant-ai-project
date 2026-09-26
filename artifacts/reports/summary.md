# Results summary

All numbers come from `metrics.json` files written by `src.evaluation.evaluate`.
Test is the leaf-level held-out split (see `data/splits/README.md`). The two extra rows
per run are the same test photos with the background changed: `segmented` is the
dataset's background-removed copy (black), `segmented_gray` puts the leaf on a flat
background in the colour of the lab table. See `docs/results.md`.

## Multiclass

| run | split | accuracy | balanced_accuracy | macro_f1 | weighted_f1 | top3_accuracy | ece |
|---|---|---|---|---|---|---|---|
| multiclass_efficientnet_b0 | test | 0.9952 | 0.9939 | 0.9938 | 0.9952 | 0.9992 | 0.1435 |
| multiclass_efficientnet_b0 | test (segmented) | 0.8028 | 0.7865 | 0.7879 | 0.8240 | 0.9532 | 0.0961 |
| multiclass_efficientnet_b0 | test (segmented_gray) | 0.8430 | 0.8358 | 0.8131 | 0.8491 | 0.9649 | 0.1132 |
| multiclass_efficientnet_b0 | val | 0.9959 | 0.9933 | 0.9898 | 0.9960 | 0.9991 | 0.1448 |
| multiclass_mobilenet_v2 | test | 0.9932 | 0.9922 | 0.9905 | 0.9932 | 0.9996 | 0.1421 |
| multiclass_mobilenet_v2 | test (segmented) | 0.7259 | 0.7358 | 0.7513 | 0.7709 | 0.9151 | 0.1323 |
| multiclass_mobilenet_v2 | test (segmented_gray) | 0.8649 | 0.8761 | 0.8335 | 0.8688 | 0.9576 | 0.1733 |
| multiclass_mobilenet_v2 | val | 0.9923 | 0.9893 | 0.9865 | 0.9924 | 0.9999 | 0.1409 |

Healthy-vs-diseased read off the multiclass models:

| run | split | derived_binary_accuracy | derived_binary_balanced_accuracy | derived_binary_sensitivity | derived_binary_specificity | derived_binary_roc_auc |
|---|---|---|---|---|---|---|
| multiclass_efficientnet_b0 | test | 0.9996 | 0.9994 | 0.9998 | 0.9991 | 0.9999 |
| multiclass_efficientnet_b0 | test (segmented) | 0.9675 | 0.9475 | 0.9936 | 0.9013 | 0.9807 |
| multiclass_efficientnet_b0 | test (segmented_gray) | 0.9511 | 0.9254 | 0.9844 | 0.8664 | 0.9812 |
| multiclass_efficientnet_b0 | val | 0.9995 | 0.9992 | 0.9998 | 0.9986 | 1.0000 |
| multiclass_mobilenet_v2 | test | 0.9988 | 0.9985 | 0.9993 | 0.9977 | 0.9999 |
| multiclass_mobilenet_v2 | test (segmented) | 0.9503 | 0.9150 | 0.9962 | 0.8339 | 0.9888 |
| multiclass_mobilenet_v2 | test (segmented_gray) | 0.9761 | 0.9598 | 0.9973 | 0.9223 | 0.9935 |
| multiclass_mobilenet_v2 | val | 0.9997 | 0.9995 | 1.0000 | 0.9990 | 0.9999 |

## Binary

| run | split | accuracy | balanced_accuracy | macro_f1 | sensitivity | specificity | roc_auc | ece |
|---|---|---|---|---|---|---|---|---|
| binary_efficientnet_b0 | test | 0.9985 | 0.9981 | 0.9981 | 0.9989 | 0.9972 | 1.0000 | 0.0660 |
| binary_efficientnet_b0 | test (segmented) | 0.9735 | 0.9578 | 0.9666 | 0.9938 | 0.9218 | 0.9915 | 0.0585 |
| binary_efficientnet_b0 | test (segmented_gray) | 0.9704 | 0.9536 | 0.9628 | 0.9923 | 0.9148 | 0.9954 | 0.0695 |
| binary_efficientnet_b0 | val | 0.9997 | 0.9997 | 0.9997 | 0.9998 | 0.9995 | 1.0000 | 0.0661 |
| binary_mobilenet_v2 | test | 0.9986 | 0.9982 | 0.9982 | 0.9991 | 0.9972 | 0.9999 | 0.0638 |
| binary_mobilenet_v2 | test (segmented) | 0.9762 | 0.9744 | 0.9709 | 0.9786 | 0.9702 | 0.9965 | 0.0599 |
| binary_mobilenet_v2 | test (segmented_gray) | 0.9656 | 0.9580 | 0.9576 | 0.9755 | 0.9404 | 0.9897 | 0.0607 |
| binary_mobilenet_v2 | val | 0.9991 | 0.9988 | 0.9988 | 0.9994 | 0.9981 | 1.0000 | 0.0641 |
