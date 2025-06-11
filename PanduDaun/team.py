import streamlit as st

st.markdown("## Tim LAI25-SM013 Pengembang Aplikasi PanduDaun")

team = [
    {
        "name": "Nida Annisa Sholeha",
        "role": "Project Manager",
        "photo": "https://ui-avatars.com/api/?name=Nida+Annisa+Sholeha&background=388e3c&color=fff"
    },
    {
        "name": "Dewi Yuliana",
        "role": "Backend Developer",
        "photo": "https://ui-avatars.com/api/?name=Dewi+Yuliana&background=388e3c&color=fff"
    },
    {
        "name": "Indira Aline",
        "role": "UI/UX Designer",
        "photo": "https://ui-avatars.com/api/?name=Indira+Aline&background=388e3c&color=fff"
    },
    {
        "name": "Numan Zainul Rahman",
        "role": "Machine Learning Engineer",
        "photo": "https://ui-avatars.com/api/?name=Numan+Zainul+Rahman&background=388e3c&color=fff"
    },
    {
        "name": "Kenneth Angelo",
        "role": "Advisor",
        "photo": "https://assets.cdn.dicoding.com/small/avatar/dos-6075361ed7e6035c937dc59e25896ad020241108122210.png"
    },
]

cols = st.columns(len(team))
for idx, member in enumerate(team):
    with cols[idx]:
        st.markdown(
            f"""
            <div style='text-align:center'>
                <img src="{member['photo']}" width="90" style="border-radius:50%; margin-bottom:8px;" />
                <div><strong>{member['name']}</strong></div>
                <div>{member['role']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )