from pathlib import Path
from time import perf_counter

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "best.pt"

CLASS_DESCRIPTIONS = {
    "Longitudinal": "Crack running parallel to the direction of the road.",
    "Transverse": "Crack running across the width of the road.",
    "Alligator": "Interconnected cracks resembling alligator skin.",
    "Pothole": "A bowl-shaped depression or hole in the road surface.",
}


# ---------------------------------------------------------
# Load model only once
# ---------------------------------------------------------
@st.cache_resource
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🛣️ Road Damage Detection System")

st.write(
    """
    Upload a road image to detect four types of road damage:
    **Longitudinal cracks, Transverse cracks, Alligator cracks and Potholes.**
    """
)

st.divider()


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("Detection Settings")

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Predictions below this confidence value will be hidden.",
    )

    image_size = st.selectbox(
        "Inference image size",
        options=[416, 512, 640],
        index=2,
        help="Larger image sizes may detect smaller cracks but take longer.",
    )

    st.divider()

    st.subheader("Damage Classes")

    for class_name, description in CLASS_DESCRIPTIONS.items():
        st.markdown(f"**{class_name}**")
        st.caption(description)


# ---------------------------------------------------------
# File uploader
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
)


# ---------------------------------------------------------
# Main prediction flow
# ---------------------------------------------------------
if uploaded_file is None:
    st.info("Upload a JPG, JPEG or PNG road image to begin.")

else:
    try:
        input_image = Image.open(uploaded_file).convert("RGB")

    except Exception as error:
        st.error(f"Unable to open the uploaded image: {error}")
        st.stop()

    original_column, output_column = st.columns(2)

    with original_column:
        st.subheader("Original Image")
        st.image(
            input_image,
            use_container_width=True,
        )

    detect_button = st.button(
        "Detect Road Damage",
        type="primary",
        use_container_width=True,
    )

    if detect_button:
        try:
            with st.spinner("Analysing the road image..."):
                model = load_model()

                start_time = perf_counter()

                results = model.predict(
                    source=input_image,
                    conf=confidence_threshold,
                    imgsz=image_size,
                    verbose=False,
                )

                processing_time = perf_counter() - start_time

            result = results[0]

            # YOLO returns an annotated image in BGR format
            annotated_array = result.plot()[..., ::-1]
            annotated_image = Image.fromarray(annotated_array)

            with output_column:
                st.subheader("Detected Damage")
                st.image(
                    annotated_image,
                    use_container_width=True,
                )

            boxes = result.boxes
            detection_rows = []

            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    x1, y1, x2, y2 = [
                        round(value, 2)
                        for value in box.xyxy[0].tolist()
                    ]

                    class_name = model.names[class_id]

                    detection_rows.append(
                        {
                            "Damage Type": class_name,
                            "Confidence": round(confidence * 100, 2),
                            "X1": x1,
                            "Y1": y1,
                            "X2": x2,
                            "Y2": y2,
                        }
                    )

            st.divider()
            st.subheader("Detection Summary")

            total_detections = len(detection_rows)

            metric_1, metric_2, metric_3 = st.columns(3)

            metric_1.metric(
                "Total Detections",
                total_detections,
            )

            metric_2.metric(
                "Processing Time",
                f"{processing_time:.2f} seconds",
            )

            if detection_rows:
                highest_confidence = max(
                    row["Confidence"] for row in detection_rows
                )

                metric_3.metric(
                    "Highest Confidence",
                    f"{highest_confidence:.2f}%",
                )

                detections_df = pd.DataFrame(detection_rows)

                st.dataframe(
                    detections_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("Damage Count")

                damage_counts = (
                    detections_df["Damage Type"]
                    .value_counts()
                    .rename_axis("Damage Type")
                    .reset_index(name="Count")
                )

                st.dataframe(
                    damage_counts,
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                metric_3.metric(
                    "Highest Confidence",
                    "N/A",
                )

                st.warning(
                    """
                    No road damage was detected above the selected
                    confidence threshold.
                    """
                )

        except FileNotFoundError as error:
            st.error(str(error))
            st.info(
                "Place the trained best.pt file in the same folder as app.py."
            )

        except Exception as error:
            st.error("Prediction failed.")
            st.exception(error)