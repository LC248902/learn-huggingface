---
title: Trashify Demo V4 🚮
emoji: 🗑️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.34.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🚮 Trashify Object Detector V4 

Object detection demo to detect `trash`, `bin`, `hand`, `trash_arm`, `not_trash`, `not_bin`, `not_hand`. 

Used as example for encouraging people to cleanup their local area.

If `trash`, `hand`, `bin` all detected = +1 point.

## Dataset

All Trashify models are trained on a custom hand-labelled dataset of people picking up trash and placing it in a bin.

The dataset can be found on Hugging Face as [`mrdbourke/trashify_manual_labelled_images`](https://huggingface.co/datasets/mrdbourke/trashify_manual_labelled_images).

## Demos

* [V1](https://huggingface.co/spaces/mrdbourke/trashify_demo_v1) = Fine-tuned [Conditional DETR](https://huggingface.co/docs/transformers/en/model_doc/conditional_detr) model trained *without* data augmentation.
* [V2](https://huggingface.co/spaces/mrdbourke/trashify_demo_v2) = Fine-tuned Conditional DETR model trained *with* data augmentation.
* [V3](https://huggingface.co/spaces/mrdbourke/trashify_demo_v3) = Fine-tuned Conditional DETR model trained *with* data augmentation (same as V2) with an NMS (Non Maximum Suppression) post-processing step.
* [V4](https://huggingface.co/spaces/mrdbourke/trashify_demo_v4) = Fine-tuned [RT-DETRv2](https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr_v2) model trained *without* data augmentation or NMS post-processing (current best mAP).

## Learn more

See the full end-to-end code of how this demo was built at [learnhuggingface.com](https://www.learnhuggingface.com/notebooks/hugging_face_object_detection_tutorial). 
