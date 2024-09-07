#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[5]:


import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from mtcnn import MTCNN
import os
import pickle
import tempfile



# Initialize the face detector and feature extractor
detector = MTCNN()
model = resnet50(pretrained=True)
model.fc = nn.Identity()  # For feature extraction
model.eval()

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract face embeddings for the user image
def load_face_embeddings():
    # Load the precomputed celebrity embeddings from a .pkl file
    with open('celeb_embeddings.pkl', 'rb') as f:
        celeb_embeddings = pickle.load(f)
    
    return celeb_embeddings

# Function to extract face embeddings for the user image
def extract_face_embedding(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection = detector.detect_faces(image_rgb)
    
    if len(detection) > 0:
        x, y, w, h = detection[0]['box']
        face = image_rgb[y:y+h, x:x+w]
        face = Image.fromarray(face)
        face_tensor = preprocess(face).unsqueeze(0)
        
        with torch.no_grad():
            embedding = model(face_tensor).numpy()
        
        return embedding
    else:
        return None
        
# Compare user image with precomputed celebrity embeddings
def find_celebrity_look_alike(user_embedding, celeb_embeddings, top_n=3):
    similarities = []
    
    # Compare the user embedding with each celebrity embedding
    for celeb_name, celeb_embedding in celeb_embeddings.items():
        similarity = cosine_similarity(user_embedding, celeb_embedding)[0][0]
        similarities.append((celeb_name, similarity))
    
    # Sort by similarity (highest first)
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Return the top N most similar celebrities
    return similarities[:top_n] if similarities else "No face detected in any celebrity images."

# Streamlit app
def main():
    st.title("Celebrity Look-Alike Finder")
    st.write("Upload a photo and find your celebrity look-alike!")
    celeb_embeddings=load_face_embeddings()
    # File upload for user image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    
    if uploaded_file is not None:
        # Convert the uploaded file to an opencv image
        img = Image.open(uploaded_file)
        img = np.array(img)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_file.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Display the uploaded image
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("")

        # Run the model and find top celebrity look-alikes
        if st.button("Find Look-Alikes"):
            st.write("Processing the image...")
            
            user_embedding = extract_face_embedding(temp_file.name)
            if user_embedding is not None:
                top_matches = find_celebrity_look_alike(user_embedding, celeb_embeddings)
                
                # Display results
                st.write(f"Top {len(top_matches)} Celebrity Matches:")
                for i, (celeb_name, similarity) in enumerate(top_matches):
                    st.write(f"Match {i+1}: {celeb_name} with similarity score of {similarity}")
                    celeb_image_path = os.path.join('./celeb_dataset/img_align_celeba/img_align_celeba', celeb_name)
                    celeb_image = Image.open(celeb_image_path)
                    st.image(celeb_image, caption=f"Celebrity {i+1}", use_column_width=True)
            else:
                st.write("No face detected in the user image.")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




