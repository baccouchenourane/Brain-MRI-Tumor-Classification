
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "brain_tumor_cnn1.keras"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)


def classify_image(image):
    if image is None:
        return {}

    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    return {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }


interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload a Brain MRI Image"),
    outputs=gr.Label(num_top_classes=4, label="Classification Results"),
    title="Brain MRI Tumor Classification",
    description="Classify a brain MRI image into Glioma, Meningioma, No Tumor or Pituitary.",
    article=(
        "Academic demonstration only. Not intended for medical diagnosis "
        "or treatment decisions. Scores are model outputs, not clinical probabilities."
    ),
)

if __name__ == "__main__":
    interface.launch()
