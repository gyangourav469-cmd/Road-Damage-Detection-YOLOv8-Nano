# Road Damage Detection using YOLOv8 Nano

## Overview

This project detects road damages such as:

- Longitudinal Crack
- Transverse Crack
- Alligator Crack
- Pothole

using a YOLOv8 Nano object detection model trained on the RDD2022 dataset.

A Streamlit web application allows users to upload road images and receive predictions with bounding boxes and confidence scores.

---

## Technologies Used

- Python
- YOLOv8 Nano
- Streamlit
- OpenCV
- Pillow
- Pandas
- Google Colab
- VS Code

---

## Dataset

RDD2022 Road Damage Dataset

Classes:
- Longitudinal
- Transverse
- Alligator
- Pothole

---

## Installation

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Output

The application displays:

- Detected damages
- Bounding boxes
- Confidence scores
- Detection summary
- Damage count

---

## Model

The trained model is stored in:

```
best.pt
```

---

## Author

Gayan Gourav
IIT Guwahati
B.Sc. (Hons.) Data Science and Artificial Intelligence
