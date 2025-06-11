import streamlit as st

def prediction_card():
    st.page_link("app.py", label="Prediction", icon=":material/insert_chart:")
    st.write("This is a prediction card.")
    st.button("Predict")

def media_card():
    st.page_link("media.py", label="Media", icon=":material/image:")
    st.video("https://s3-us-west-2.amazonaws.com/assets.streamlit.io/videos/hero-video.mp4", autoplay=True)

def chat_card():
    st.page_link("chat.py", label="Chat", icon=":material/chat:")
    st.chat_message("user").write("Halo, semua!")
    st.chat_message("assistant").write("Halo!")
    st.chat_input("Ketik sesuatu")
