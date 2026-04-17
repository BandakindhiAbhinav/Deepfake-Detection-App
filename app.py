import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

from model import CNNModel

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("Deepfake Image Detection")

device = torch.device("cpu")


@st.cache_resource
def load_model():

    model = CNNModel()

    model.load_state_dict(
        torch.load(
            "deep_cnn_model_weights.pth",
            map_location=device
        )
    )

    model.eval()

    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def predict_image(image):

    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        probability = output.item()

    if probability >= 0.5:
        return "Real", probability
    else:
        return "Deepfake", probability


uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image)

    if st.button("Predict"):

        label, prob = predict_image(image)

        st.subheader(label)

        st.write("Confidence:", prob)
