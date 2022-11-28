import torch
import model
import solver
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
import streamlit as st


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

model = solver.Generator(64, 5, 6)
model.load_state_dict(torch.load("/home/huyduong/stargan/200000-G.ckpt", map_location=lambda storage, loc: storage))

transform = []
transform.append(T.CenterCrop(178))
transform.append(T.Resize(128))
transform.append(T.ToTensor())
transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform = T.Compose(transform)


st.header("Demo StarGAN")
#Wearing_Hat Eyeglasses Smiling Male Young

st.write("Hit checkboxes to choose the attributes for the output image")

wearing_hat = st.checkbox('Wearing hat')
eyeglasses = st.checkbox('Eyeglasses')
smiling = st.checkbox("Smiling")
male = st.checkbox("Male")
young = st.checkbox("Young")
st.write("Choose input image")
uploaded_file = st.file_uploader("Choose an image")


if uploaded_file is not None:

    device =  torch.device("cpu")
    
    Imag = Image.open(uploaded_file)

    Img = transform(Imag)
    Img = Img.float().unsqueeze(0)
    x_real = Img

    attrs = [wearing_hat, eyeglasses, smiling, male, young]
    attrs = [int(i) for i in attrs]

    c_trg = torch.FloatTensor([attrs])
    x_real = x_real.to(device)
    x_fake = model(x_real, c_trg)
    save_image(denorm(x_fake.data.cpu()), "out.jpg", nrow=1, padding=0)
    out_image = Image.open("out.jpg")

    col1, col2 = st.columns(2)
    with col1:
        st.image(Imag, caption='Original', use_column_width='auto')
    with col2:
        st.image(out_image, caption='Transformed', use_column_width='auto')


