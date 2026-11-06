import pandas as pd
import plotly.express as px
import streamlit as st
import core

st.title("🗂️ Données & compréhension des variables")
st.caption("Jeu : Credit Card Transactions Fraud Detection (Sparkov, synthétique) — cible `is_fraud`.")

with st.sidebar:
    st.header("1. Aperçu rapide")
    source = st.radio("Source", ["Kaggle (automatique)", "Fichiers locaux (data/)", "Importer un CSV"], index=0)
    up = st.file_uploader("CSV (fraudTrain/fraudTest concaténés)", type="csv") if source == "Importer un CSV" else None
    go_prev = st.button("👁️ Charger l'aperçu", type="primary", width="stretch")

    st.header("2. Préparation (pages 2 à 5)")
    frac = st.select_slider("Échantillon (part des cartes conservées)", [0.05, 0.1, 0.25, 0.5, 1.0], value=0.1,
                            format_func=lambda x: f"{int(x*100)} %",
                            help="On échantillonne des cartes entières : l'historique de chaque carte "
                                 "reste complet, donc les variables de vélocité restent exactes.")
    go_prep = st.button("⚙️ Préparer les données", width="stretch")

SRC = {"Kaggle (automatique)": "kaggle", "Fichiers locaux (data/)": "local", "Importer un CSV": "upload"}

if go_prev:
    try:
        st.session_state["overview"] = core.load_overview(SRC[source], up.getvalue() if up else None)
        st.session_state["src"] = (SRC[source], up.getvalue() if up else None)
    except Exception as e:
        st.error(f"Échec du chargement : {e}")
        st.info("Si Kaggle refuse l'accès, téléchargez les CSV, mettez-les dans data/ et choisissez « Fichiers locaux ».")

if go_prep:
    if "src" not in st.session_state:
        st.warning("Chargez d'abord l'aperçu.")
    else:
        with st.spinner("Préparation : échantillon, variables, séparation chronologique…"):
            s, b = st.session_state["src"]
            raw = core.sample_cards(core.load_raw(s, b), frac)
            df = core.build_features(raw)
            st.session_state.update(df=df, mats=core.make_matrices(df), frac=frac)
            st.session_state.pop("bundle", None)
        st.success(f"Données préparées : {len(df):,} transactions ({df['cc_num'].nunique()} cartes). "
                   "Vous pouvez ouvrir les pages 2 à 5.")

ov = st.session_state.get("overview")
if ov is None:
    st.info("👈 Cliquez sur **Charger l'aperçu** dans le menu de gauche.")
    st.stop()

# ---------- Vue d'ensemble ----------
st.subheader("Vue d'ensemble du jeu de données")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes (transactions)", f"{ov['n_rows']:,}")
c2.metric("Colonnes", ov["n_cols"])
c3.metric("Cartes distinctes", f"{ov['n_cards']:,}")
c4.metric("Catégories marchands", ov["n_cats"])

c5, c6, c7, c8 = st.columns(4)
c5.metric("Légitimes", f"{ov['n_legit']:,}", f"{100*ov['n_legit']/ov['n_rows']:.2f} %", delta_color="off")
c6.metric("Fraudes", f"{ov['n_fraud']:,}", f"{100*ov['n_fraud']/ov['n_rows']:.3f} %", delta_color="off")
c7.metric("Rapport", f"1 fraude / {ov['n_rows']//max(ov['n_fraud'],1)}")
c8.metric("Période", f"{ov['date_min']:%b %Y} → {ov['date_max']:%b %Y}")

col_a, col_b = st.columns([1, 1])
with col_a:
    pie = pd.DataFrame({"Classe": ["Légitime", "Fraude"], "Nombre": [ov["n_legit"], ov["n_fraud"]]})
    st.plotly_chart(px.pie(pie, names="Classe", values="Nombre", hole=0.5, title="Répartition des classes"), width="stretch")
with col_b:
    st.markdown("**Montants par classe ($)**")
    st.dataframe(ov["amt_by_class"], width="stretch")
    st.info("Jeu **extrêmement déséquilibré** : l'accuracy n'a aucun sens → PR-AUC, précision@k, rappel@k, coût.")

tab1, tab2, tab3 = st.tabs(["10 premières lignes", "Dictionnaire des variables", "Variables construites"])

with tab1:
    st.dataframe(ov["head"], width="stretch")

with tab2:
    cols = list(ov["head"].columns)
    d = pd.DataFrame({"Variable": cols, "Type": [ov["dtypes"][c] for c in cols],
                      "Description": [core.DESCRIPTIONS.get(c, "") for c in cols],
                      "Exemple": [str(ov["head"][c].iloc[0]) for c in cols]})
    st.dataframe(d, width="stretch", hide_index=True)

with tab3:
    st.markdown("""
| Variable | Définition | Pourquoi |
|---|---|---|
| `hour_sin`, `hour_cos` | Heure codée de façon **cyclique** | 23 h et 0 h sont proches |
| `dow` | Jour de la semaine | Rythme hebdomadaire |
| `age` | Âge à la date de la transaction | Profil client |
| `distance_km` | Distance haversine client ↔ marchand | Transaction « loin de chez soi » |
| `n_1h`, `n_24h` | Nb de transactions de la carte sur 1 h / 24 h **passées** | Vélocité |
| `amt_ratio_30d` | Montant / moyenne des 30 derniers jours de la carte | Écart au profil |
| `hours_since_prev` | Temps depuis la transaction précédente de la carte | Rythme d'usage |
| `cat_*` | Catégorie du marchand (one-hot) | Risque par secteur |

**Pas de fuite :** les variables comportementales n'utilisent que le passé de la carte.
""")

# ---------- Après préparation ----------
if "df" in st.session_state:
    df = st.session_state["df"]
    st.divider()
    st.subheader("Données préparées (échantillon)")
    resume = df.groupby("split").agg(début=("trans_date_trans_time", "min"), fin=("trans_date_trans_time", "max"),
                                     transactions=("is_fraud", "count"), fraudes=("is_fraud", "sum"),
                                     taux=("is_fraud", "mean")).loc[["train", "val", "test"]]
    resume["taux_%"] = (resume.pop("taux") * 100).round(3)
    st.dataframe(resume, width="stretch")
    st.caption("Séparation **chronologique** 60 % / 20 % / 20 %. Le déséquilibre est traité uniquement sur le train.")
    st.plotly_chart(px.histogram(df, x="trans_date_trans_time", color="split", nbins=80,
                                 title="Transactions par période et par ensemble"), width="stretch")
else:
    st.warning("Pour utiliser les pages 2 à 5, cliquez sur **Préparer les données** dans le menu de gauche.")