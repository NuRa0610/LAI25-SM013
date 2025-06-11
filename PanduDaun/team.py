import streamlit as st

st.markdown("## Tim LAI25-SM013 Pengembang Aplikasi PanduDaun")

team = [
    {
        "name": "Nida Annisa Sholeha",
        "role": "Project Manager",
        "photo": "https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/team_1.png"
    },
    {
        "name": "Dewi Yuliana",
        "role": "Backend Developer",
        "photo": "https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/team_2.png"
    },
    {
        "name": "Indira Aline",
        "role": "UI/UX Designer",
        "photo": "https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/team_3.png"
    },
    {
        "name": "Numan Zainul Rahman",
        "role": "Machine Learning Engineer",
        "photo": "https://raw.githubusercontent.com/NuRa0610/LAI25-SM013/main/PanduDaun/team_4.png"
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