# app

A small Gradio demo: upload a leaf photo, get the top-5 predictions and a Grad-CAM overlay showing
which part of the image drove the answer.

```bash
python app/app.py                                                   # uses models/exported/*
PLANT_MODEL_REPO=atilimai/plantvillage-leaf-disease-classifier python app/app.py   # from the Hub
```

Both the disease model (38 classes) and the healthy/diseased model are selectable if they have been
exported.

`scripts/build_hf_release.py` packages this same `app.py` as a Hugging Face Space: it copies the
handful of `src/` modules inference needs, writes the Space metadata, and adds a few PlantVillage
test images as examples. Publishing is a manual step (`scripts/upload_to_hub.py --space ...`).

The interface repeats the caveat that matters: the models were trained on single leaves
photographed on a plain background in a lab, so field photos are out of distribution and the
predictions on them should not be trusted.
