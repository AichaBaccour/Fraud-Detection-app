import streamlit as st

st.set_page_config(page_title="Détection de fraude explicable", page_icon="🛡️", layout="wide")

pages = [
    st.Page("pages/1_Donnees.py", title="1 · Données & variables", icon="🗂️", default=True),
    st.Page("pages/2_Distributions.py", title="2 · Distributions & visualisations", icon="📊"),
    st.Page("pages/3_Modeles.py", title="3 · Modèles & entraînement", icon="🧠"),
    st.Page("pages/4_Resultats.py", title="4 · Résultats, seuil & coûts", icon="🎯"),
    st.Page("pages/5_XAI.py", title="5 · Explicabilité (XAI)", icon="🔍"),
]
st.navigation(pages).run()