# Citation

If you use this repository, the released weights, or the split manifests, please cite the dataset
first — the data is what makes the work possible, and its licence requires attribution.

## Dataset

```bibtex
@article{mohanty2016using,
  title   = {Using Deep Learning for Image-Based Plant Disease Detection},
  author  = {Mohanty, Sharada P. and Hughes, David P. and Salath{\'e}, Marcel},
  journal = {Frontiers in Plant Science},
  volume  = {7},
  pages   = {1419},
  year    = {2016},
  doi     = {10.3389/fpls.2016.01419}
}

@article{hughes2015open,
  title   = {An open access repository of images on plant health to enable the development of mobile disease diagnostics},
  author  = {Hughes, David P. and Salath{\'e}, Marcel},
  journal = {arXiv preprint arXiv:1511.08060},
  year    = {2015}
}
```

PlantVillage images are licensed CC BY-SA 3.0, and so are models trained on them. See `LICENSING.md`.

## This project

```bibtex
@misc{plantai2026leafdisease,
  title        = {Leaf-level, leakage-free plant disease classification on PlantVillage},
  author       = {{Atılım AI Club}},
  year         = {2026},
  howpublished = {\url{https://github.com/atilimai/plant-ai-project}},
  note         = {Models at \url{https://huggingface.co/atilimai/plantvillage-leaf-disease-classifier}}
}
```

## Methods used

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title     = {MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  author    = {Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle = {CVPR},
  year      = {2018}
}

@inproceedings{tan2019efficientnet,
  title     = {EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author    = {Tan, Mingxing and Le, Quoc V.},
  booktitle = {ICML},
  year      = {2019}
}

@inproceedings{selvaraju2017gradcam,
  title     = {Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization},
  author    = {Selvaraju, Ramprasaath R. and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  booktitle = {ICCV},
  year      = {2017}
}
```

## Note on comparability

Accuracy figures from this repository are **not** comparable with the PlantVillage numbers usually
quoted in the literature (often 99%+). Those are almost always measured on image-level splits, where
photos of the same physical leaf appear in both train and test. Our split is made at the leaf level
and audited; see `data/splits/README.md`. Please keep that distinction if you cite the numbers.
