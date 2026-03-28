# app/

This directory is a placeholder for a future demo application.

## Planned Purpose

A lightweight inference demo (e.g., Gradio or Streamlit) that allows interactive plant disease classification from uploaded leaf images.

## Planned Features (Post-Release)

- Upload a leaf image and receive a predicted disease category with confidence score
- Display a Grad-CAM heatmap overlay alongside the prediction
- Support both binary (healthy/unhealthy) and multiclass (per-disease) prediction modes

## Current Status

⚠️ **This directory is empty. No application code exists yet.**

A demo application is planned for Week 4 and may be deployed as a Hugging Face Space after the model card and license review are complete. See `ROADMAP.md` and Issue #14 (Hugging Face packaging).

## Notes

- Application development depends on a trained and evaluated model checkpoint
- Deployment requires dataset license verification (Issue #13)
- If deployed to Hugging Face Spaces, it will use the Gradio framework
