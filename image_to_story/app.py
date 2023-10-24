from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import requests
import os
import streamlit as st
import re

load_dotenv(find_dotenv())

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# img2txt


def img2txt(url):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-large", max_new_tokens=50)
    text = image_to_text(url)[0]["generated_text"]
    print(text)
    return text


# llm
def new_story(senario):
    API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}",
               'Content-Type': 'application/json'}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": "<s>[INST] write short story about " + senario + "[/INST]",
        "parameters": {"max_new_tokens": 200},
    })
    print(output[0]["generated_text"])
    story = output[0]["generated_text"]
    # Define a regular expression pattern to match text between <s> and [/INST]
    pattern = r'<s>.*?\[/INST\]'

    # Use re.sub() to remove matched text
    story = re.sub(pattern, '', str(story))

    return story


# txt to speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}
    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio1.flac', 'wb') as file:
        file.write(response.content)


def main():
    st.set_page_config(page_title="Image to audio story", page_icon="🤖")

    st.header("turn image into audio story")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.',
                 use_column_width=True)
        senario = img2txt(uploaded_file.name)
        story = new_story(senario)
        text2speech(story)

        with st.expander("senario"):
            st.write(senario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio1.flac")


if __name__ == '__main__':
    main()
