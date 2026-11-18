import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Charger le modèle entraîné
model = tf.keras.models.load_model("../brain_tumor_cnn1.keras")

# Classes du dataset
class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


def classify_image(image):

    # Prétraitement
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prédiction
    predictions = model.predict(image_array, verbose=0)[0]

    # Résultats
    results = {
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    }

    return results


# Interface Gradio
interface = gr.Interface(
    fn=classify_image,

    inputs=gr.Image(
        type="pil",
        label="Upload a Brain MRI Image"
    ),

    outputs=gr.Label(
        num_top_classes=4,
        label="Classification Results"
    ),

    title="Brain MRI Tumor Classification",

    description=(
        "Upload a brain MRI image to classify it "
        "into one of four categories: Glioma, "
        "Meningioma, No Tumor or Pituitary."
    )
)

interface.launch()
