# SIXray — AI X-Ray Baggage Detection

A Streamlit academic proof of concept for object detection in X-ray baggage images
using the public, fine-tuned YOLO11s checkpoint hosted at
[Fransiskus1990/sixray-yolo11s-xray-detection](https://huggingface.co/Fransiskus1990/sixray-yolo11s-xray-detection).

The app downloads `best.pt` from Hugging Face Hub on startup, verifies the checkpoint's
class mapping, and caches the loaded model for the lifetime of the Streamlit process.
Inference runs on CPU. If the requested checkpoint cannot be downloaded or loaded,
the app reports the error and does not substitute other weights.

## Local setup

This application supports Python 3.11. From the repository directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, then upload a PNG, JPG, or JPEG image.
The first launch downloads the model checkpoint and may take longer than subsequent
interactions.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub. Do not add credentials, Hugging Face tokens, model
   checkpoint files, training images, or the SIXray dataset to the repository.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), select **Create app**
   and connect the GitHub repository and branch.
3. Set the app's main file path to `app.py`.
4. In **Advanced settings**, select Python **3.11**, then deploy. The Hugging Face
   repository is public, so no token or Streamlit secret is required.
5. The deployed app downloads and caches the checkpoint when it starts. Keep outbound
   network access available to Hugging Face Hub. If download fails, the app displays
   an error and will not fall back to pretrained weights.

The hosted environment runs CPU inference. Its available memory, startup time, and
inference performance depend on the Streamlit Community Cloud instance.

## Interface

- Confidence threshold control (default 0.25)
- Original and annotated images side by side
- Class names and confidence scores on annotated detections
- Detection count, class counts, bounding-box coordinates, and timing
- Downloadable annotated PNG
- Previously measured held-out test metrics and per-class AP@50

Class mapping: **Gun**, **Knife**, **Pliers**, **Scissors**, and **Wrench** (IDs 0–4).
The application checks that the loaded checkpoint exposes this mapping.

## Previously measured evaluation results

The performance section shows the provided V5D held-out test results: 831 test images,
1,593 annotated objects, 91.16% precision, 84.77% recall, 87.85% F1-score, 91.28%
mAP@50, and 66.38% mAP@50–95. Per-class AP@50 is Gun 98.94%, Knife 88.57%, Pliers
93.11%, Scissors 90.29%, and Wrench 85.46%. These are prior dataset evaluation
results, not a score computed for an uploaded image. The approximately 14.7 ms GPU
inference benchmark was measured on a Tesla T4 and does not describe Streamlit hosting.

## Safety and limitations

This application is an academic proof of concept. Predictions may be incorrect or
incomplete. It is not certified for operational security screening and should not be
used as the sole basis for security decisions.
