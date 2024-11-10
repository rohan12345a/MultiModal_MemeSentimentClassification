import streamlit as st
import torch
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms
from early_fusion import EarlyFusionModel
from late_fusion import LateFusionModel
from hybrid_fusion import HybridFusionModel

# Load the models (Ensure the models are available in the same directory or adjust paths)
model_paths = {
    'early_fusion': "C:\\Users\\Lenovo\\Downloads\\model_early_fusion.pth",
    'late_fusion': "C:\\Users\\Lenovo\\Downloads\\model_late_fusion.pth",
    'hybrid_fusion': "C:\\Users\\Lenovo\\Downloads\\model_hybrid_fusion.pth"
}

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

# Preprocess image for ResNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit interface
st.title("Multimodal Meme Sentiment Classification")
st.write("This app classifies text and image inputs using either Early Fusion, Late Fusion, or Hybrid Fusion model.")

# User selects which model to use
model_choice = st.selectbox("Select a model", ['Early Fusion', 'Late Fusion', 'Hybrid Fusion'])

# Initialize the model based on user choice
if model_choice == 'Early Fusion':
    model = EarlyFusionModel()
    model.load_state_dict(torch.load(model_paths['early_fusion']))
elif model_choice == 'Late Fusion':
    model = LateFusionModel()
    model.load_state_dict(torch.load(model_paths['late_fusion']))
else:
    model = HybridFusionModel()
    model.load_state_dict(torch.load(model_paths['hybrid_fusion']))

model.eval()

# Text input
text_input = st.text_area("Enter Text", "Type your text here...")

# Image input
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Prediction button
if st.button("Predict"):
    if text_input and uploaded_image is not None:
        # Process the image
        image = Image.open(uploaded_image)
        image = preprocess(image).unsqueeze(0)

        # Tokenize the text input
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Prepare inputs for the model
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Run the model
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, images=image)

        # Convert output to prediction
        prediction = torch.argmax(output, dim=1).item()

        # Display the result
        if prediction == 0:
            st.write("The model predicts the combination of text and images is offensive (0)")
        else:
            st.write("The model predicts: the combination of text and images is Non-offensive (1)")

        # Additional text and image sentiment classification (Offensive or Non-offensive)
        if prediction == 0:
            st.write("Sentiment: Offensive")
        else:
            st.write("Sentiment: Non-offensive")
    else:
        st.write("Please provide both text and an image for prediction.")


with st.container():
    with st.sidebar:
        members = [
            {"name": "Saksham Jain", "email": "sakshamgr8online@gmail. com",
             "linkedin": "https://www.linkedin.com/in/saksham-jain-59b2241a4/"},
            {"name": "Rohan Saraswat", "email": "rohan.saraswat2003@gmail. com", "linkedin": "https://www.linkedin.com/in/rohan-saraswat-a70a2b225/"},

        ]

        # Define the page title and heading
        st.markdown("<h1 style='font-size:28px'>Authors</h1>", unsafe_allow_html=True)

        # Iterate over the list of members and display their details
        for member in members:
            st.write(f"Name: {member['name']}")
            st.write(f"Email: {member['email']}")
            st.write(f"LinkedIn: {member['linkedin']}")
            st.write("")
