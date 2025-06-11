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

def team_card():
    st.markdown("### 👥 Tentang Kami")
    team = [
        {
            "name": "Nida Annisa Sholeha",
            "role": "Frontend Developer",
            "photo": "https://ui-avatars.com/api/?name=Nida+Annisa+Sholeha&background=388e3c&color=fff"
        },
        {
            "name": "Numan ",
            "role": "Machine Learning Engineer",
            "photo": "https://ui-avatars.com/api/?name=Numan+Rahman&background=388e3c&color=fff"
        },
        {
            "name": "Budi Santoso",
            "role": "Backend Developer",
            "photo": "https://ui-avatars.com/api/?name=Budi+Santoso&background=388e3c&color=fff"
        },
        {
            "name": "Siti Aminah",
            "role": "UI/UX Designer",
            "photo": "https://ui-avatars.com/api/?name=Siti+Aminah&background=388e3c&color=fff"
        },
        {
            "name": "Rizky Pratama",
            "role": "Data Scientist",
            "photo": "https://ui-avatars.com/api/?name=Rizky+Pratama&background=388e3c&color=fff"
        },
    ]
    cols = st.columns(len(team))
    for idx, member in enumerate(team):
        with cols[idx]:
            st.image(member["photo"], width=90)
            st.markdown(f"**{member['name']}**")
            st.caption(member["role"])
