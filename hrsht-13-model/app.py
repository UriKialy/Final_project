import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import gradio as gr

from weights import download_weights
from model import download_model
from tensorflow.keras.optimizers import Adam


def set_working_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)


def load_model():
    # (re)build and save architecture if needed
    download_model()

    # load the saved architecture
    model = tf.keras.models.load_model("model/model.h5")

    # compile with the built-in Adam (no legacy decay arg)
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )

    # download & load your pretrained weights
    download_weights()
    model.load_weights("weights/modeldense1.h5")
    return model


def preprocess(image):
    # sharpening filter
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def predict_img(img, model):
    img = preprocess(img)
    img = img / 255.0
    batch = img.reshape(1, 224, 224, 3)
    preds = model.predict(batch)[0]

    class_names = [
        'Benign with Density=1', 'Malignant with Density=1',
        'Benign with Density=2', 'Malignant with Density=2',
        'Benign with Density=3', 'Malignant with Density=3',
        'Benign with Density=4', 'Malignant with Density=4'
    ]
    return {class_names[i]: float(preds[i]) for i in range(len(class_names))}


def main():
    set_working_dir()
    model = load_model()

    interface = gr.Interface(
        fn=lambda img: predict_img(img, model),
        inputs=gr.Image(shape=(224, 224)),
        outputs=gr.Label(num_top_classes=8),
        title="Breast Cancer Detection",
        description="Upload a mammogram image to predict benign vs malignant and density.",
        capture_session=True
    )
    interface.launch(debug=True, share=True)


if __name__ == "__main__":
    main()
