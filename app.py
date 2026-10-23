from __future__ import annotations

import io
import time
from collections import Counter
from typing import Any

import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


MODEL_REPOSITORY = "Fransiskus1990/sixray-yolo11s-xray-detection"
MODEL_FILENAME = "best.pt"
IMAGE_SIZE = 800
EXPECTED_CLASSES = {
    0: "Gun",
    1: "Knife",
    2: "Pliers",
    3: "Scissors",
    4: "Wrench",
}

st.set_page_config(
    page_title="SIXray — AI X-Ray Baggage Detection",
    page_icon="🛄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #071426 0%, #0b1c33 52%, #102744 100%);
        color: #edf4ff;
    }
    [data-testid="stHeader"] { background: rgba(7, 20, 38, 0.92); }
    [data-testid="stSidebar"] { background-color: #0a192c; }
    [data-testid="stMetric"] {
        background: rgba(18, 43, 70, 0.78);
        border: 1px solid rgba(116, 169, 220, 0.22);
        padding: 14px;
        border-radius: 12px;
    }
    .eyebrow {
        color: #79c8ff;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .disclaimer {
        border-left: 4px solid #f0b44b;
        padding: 0.8rem 1rem;
        background: rgba(240, 180, 75, 0.10);
        border-radius: 0 8px 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model() -> YOLO:
    """Download and cache the requested public checkpoint; never substitute weights."""
    checkpoint_path = hf_hub_download(
        repo_id=MODEL_REPOSITORY,
        filename=MODEL_FILENAME,
    )
    model = YOLO(checkpoint_path)

    names: Any = model.names
    if isinstance(names, (list, tuple)):
        actual_classes = dict(enumerate(names))
    elif isinstance(names, dict):
        try:
            actual_classes = {int(class_id): name for class_id, name in names.items()}
        except (TypeError, ValueError) as exc:
            raise RuntimeError("The checkpoint contains invalid class IDs.") from exc
    else:
        raise RuntimeError("The checkpoint does not expose a supported class mapping.")

    if actual_classes != EXPECTED_CLASSES:
        raise RuntimeError(
            "Checkpoint class mapping does not match the expected SIXray classes. "
            f"Expected {EXPECTED_CLASSES}; found {actual_classes}."
        )
    return model


def read_uploaded_image(uploaded_file: Any) -> Image.Image:
    """Validate the upload and return a fully loaded RGB image."""
    try:
        with Image.open(uploaded_file) as image:
            image.verify()
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as image:
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("This file is not a readable PNG, JPG, or JPEG image.") from exc


def get_detections(result: Any) -> list[dict[str, float | int | str]]:
    detections: list[dict[str, float | int | str]] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    coordinates = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    for class_id, confidence, (x1, y1, x2, y2) in zip(
        class_ids, confidences, coordinates
    ):
        detections.append(
            {
                "Class": EXPECTED_CLASSES[int(class_id)],
                "Confidence": round(float(confidence), 4),
                "X1": round(float(x1), 1),
                "Y1": round(float(y1), 1),
                "X2": round(float(x2), 1),
                "Y2": round(float(y2), 1),
            }
        )
    return detections


def annotated_png(result: Any) -> bytes:
    # Ultralytics plot() returns an OpenCV-style BGR array; Streamlit expects RGB.
    annotated_rgb = result.plot()[..., ::-1]
    output = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(annotated_rgb)).save(output, format="PNG")
    return output.getvalue()


st.markdown('<p class="eyebrow">Academic computer vision demonstration</p>', unsafe_allow_html=True)
st.title("SIXray — AI X-Ray Baggage Detection")
st.subheader("YOLO11s Object Detection | Academic Proof of Concept")
st.write(
    "Upload an X-ray baggage image to view detections from the trained SIXray YOLO11s model."
)

with st.sidebar:
    st.header("Detection settings")
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Detections below this confidence are omitted.",
    )
    st.caption(f"Model input size: {IMAGE_SIZE} × {IMAGE_SIZE} pixels")
    st.caption("Inference device: CPU")

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, and JPEG.",
)

model: YOLO | None = None
try:
    with st.spinner("Downloading or loading the SIXray model…"):
        model = load_model()
except Exception as exc:
    st.error(
        "The SIXray checkpoint could not be downloaded or loaded. "
        "Check the deployment's network access and dependency logs, then reload the app. "
        "No replacement pretrained weights were loaded."
    )
    st.caption(f"Model error: {exc}")

if uploaded_file is not None:
    try:
        original_image = read_uploaded_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
    else:
        if model is None:
            st.warning("Detection is unavailable until the requested checkpoint loads.")
        else:
            try:
                start_time = time.perf_counter()
                with st.spinner("Running detection on CPU…"):
                    predictions = model.predict(
                        source=original_image,
                        imgsz=IMAGE_SIZE,
                        conf=confidence_threshold,
                        device="cpu",
                        verbose=False,
                    )
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if not predictions:
                    raise RuntimeError("The model returned no prediction result.")

                prediction = predictions[0]
                detections = get_detections(prediction)
                output_image = annotated_png(prediction)
                inference_ms = prediction.speed.get("inference")
                if inference_ms is None:
                    inference_ms = elapsed_ms
                    timing_label = "Prediction time (ms)"
                else:
                    timing_label = "Model inference (ms)"

                original_column, prediction_column = st.columns(2)
                with original_column:
                    st.markdown("#### Original image")
                    st.image(original_image, use_container_width=True)
                with prediction_column:
                    st.markdown("#### Annotated prediction")
                    st.image(output_image, use_container_width=True)

                st.markdown("### Detection summary")
                metric_columns = st.columns(2)
                metric_columns[0].metric("Detected objects", len(detections))
                metric_columns[1].metric(timing_label, f"{float(inference_ms):.2f}")

                st.markdown("#### Counts by class")
                class_counts = Counter(str(item["Class"]) for item in detections)
                count_columns = st.columns(len(EXPECTED_CLASSES))
                for column, class_name in zip(count_columns, EXPECTED_CLASSES.values()):
                    column.metric(class_name, class_counts.get(class_name, 0))

                if detections:
                    st.markdown("#### Detection details")
                    st.table(detections)
                else:
                    st.info(
                        "No objects were detected at the selected confidence threshold."
                    )

                st.download_button(
                    "Download annotated image",
                    data=output_image,
                    file_name="sixray_detection.png",
                    mime="image/png",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(
                    "Inference failed for this image. Try another supported image or "
                    "check the app logs for details."
                )
                st.caption(f"Inference error: {exc}")

st.divider()
st.header("Model Performance")
st.caption(
    "Previously measured held-out test results (V5D). These are dataset evaluation "
    "metrics, not metrics calculated from the current uploaded image."
)

performance_columns = st.columns(6)
for column, label, value in zip(
    performance_columns,
    ("Test images", "Annotated objects", "Precision", "Recall", "F1-score", "mAP@50"),
    ("831", "1,593", "91.16%", "84.77%", "87.85%", "91.28%"),
):
    column.metric(label, value)
st.metric("mAP@50–95", "66.38%")

st.markdown("#### Per-class AP@50")
st.table(
    [
        {"Class": class_name, "AP@50": score}
        for class_name, score in (
            ("Gun", "98.94%"),
            ("Knife", "88.57%"),
            ("Pliers", "93.11%"),
            ("Scissors", "90.29%"),
            ("Wrench", "85.46%"),
        )
    ]
)
st.caption(
    "The reported GPU benchmark of approximately 14.7 ms inference per image was "
    "measured on a Tesla T4, not on Streamlit hosting."
)

st.markdown("### Safety and limitations")
st.markdown(
    '<div class="disclaimer">This application is an academic proof of concept. '
    "Predictions may be incorrect or incomplete. It is not certified for operational "
    "security screening and should not be used as the sole basis for security decisions."
    "</div>",
    unsafe_allow_html=True,
)
