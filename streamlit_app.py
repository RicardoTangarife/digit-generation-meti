import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from model import DigitGenerator

st.set_page_config(page_title="MNIST Digit Generator", layout="centered")
st.title("🖊️ Handwritten Digit Generator (0–9)")

digit = st.selectbox("Select a digit to generate:", list(range(10)))

model = DigitGenerator()
model.load_state_dict(torch.load("digit_generator.pth", map_location="cpu"))
model.eval()

if st.button("Generate Images"):
    z = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        images = model(z, labels).squeeze(1)
    grid = make_grid(images, nrow=5, normalize=True)

    npimg = grid.numpy()
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.axis('off')
    st.pyplot(fig)