import streamlit as st

st.set_page_config(layout="wide")

pages = {
    "Select how you want to extract images:": [
        st.Page("pages/0_individual_match.py", title="Análise de um jogo"),
        st.Page("pages/1_clustering.py", title="Perfis de jogadores"),
        st.Page("pages/2_notes.py", title="Notas"),
    ]      
}

pg = st.navigation(pages)

pg.run()