import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from streamlit_lottie import st_lottie
import json
import requests


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Define a custom Streamlit component to wrap the Lottie animation
def lottie_animation(url, width, height):
    st_lottie(url, width=width, height=height)





lottie_hello = load_lottieurl("https://lottie.host/9b34ea41-9a2a-41e9-94dc-04a16e928e1d/PRVhyWamle.json")

lottie_end = load_lottieurl("https://lottie.host/c9589ae4-903e-4298-a312-e592ac69de4b/SvqN0wBjQL.json")

# https://lottie.host/c9589ae4-903e-4298-a312-e592ac69de4b/SvqN0wBjQL.json

st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    height=None,
    width=None,
    key=None,
)



# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image, width= 224)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # show
        # col1, col2, col3, col4, col5 = st.columns(5)
        # print(indices)
        # print(indices[0][0])
        # print(filenames[indices[0][0]])
        # display_image_ex = Image.open(filenames[indices[0][4]])
        # st.image(display_image_ex)

        # Number of columns you want to display
        num_columns = 4
        num_rows = 1

        # Create Streamlit columns
        columns = st.columns(num_columns)

        # Iterate over the image URLs and display them in columns
        for i in range(1,5):
            with columns[i-1]:
                display_image_ex = Image.open(filenames[indices[0][i]])
                st.image(display_image_ex, use_column_width=True)

        st_lottie(
            lottie_end,
            speed=1,
            reverse=False,
            loop=True,
            height=None,
            width=None,
            key=None,
        )



        # col1.image(filenames[indices[0][0]], width=224, use_column_width=True)
        # col2.image(filenames[indices[0][1]], width=224, use_column_width=True)
        # col3.image(filenames[indices[0][2]], width=224, use_column_width=True)
        # col4.image(filenames[indices[0][3]], width=224, use_column_width=True)
        # col5.image(filenames[indices[0][4]], width=224, use_column_width=True)


        # with col1:
        #     st.image(filenames[indices[0][0]], width= 224)
        # with col2:
        #     st.image(filenames[indices[0][1]], width= 224)
        # with col3:
        #     st.image(filenames[indices[0][2]], width= 224)
        # with col4:
        #     st.image(filenames[indices[0][3]], width= 224)
        # with col5:
        #     st.image(filenames[indices[0][4]], width= 224)
    else:
        st.header("Some error occured in file upload")
