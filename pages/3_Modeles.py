import os
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import core

st.title("🧠 Modèles & entraînement")
df = core.need_data()
mats = st.session_state["mats"]
BUNDLE_PATH = "models/bundle.joblib"

ytr = mats["train"][1]
st.write(f"Train : **{len(ytr):,}** lignes dont **{int(ytr.sum())}** fraudes · "
         f"Val : {len(mats['val'][1]):,} · Test : {len(mats['test'][1]):,}")

st.subheader("1. Choix des modèles")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Supervisés** (stratégie anti-déséquilibre : sous-échantillonnage 10:1, comparé en validation)")
    sel_sup = [m for m in core.SUPERVISED if st.checkbox(m, value=True, key=f"s_{m}")]
with c2:
    st.markdown("**Non supervisés** (détection d'anomalies)")
    sel_uns = [m for m in core.UNSUPERVISED if st.checkbox(m, value=True, key=f"u_{m}")]
selected = sel_sup + sel_uns

st.subheader("2. Échantillons & paramètres (vitesse ↔ précision)")
p1, p2, p3 = st.columns(3)
frac_train = p1.select_slider("Part des légitimes du train utilisée", [0.1, 0.25, 0.5, 1.0], value=1.0,
                              format_func=lambda x: f"{int(x*100)} %", help="Toutes les fraudes sont conservées.")
ratio = p1.slider("Ratio sous-échantillonnage (légitimes pour 1 fraude)", 1, 50, 10)
n_est = p2.select_slider("Nombre d'arbres (XGB / RF / IF)", [50, 100, 200, 300], value=200)
lof_n = p2.select_slider("Taille échantillon LOF", [2000, 5000, 10000, 20000], value=10000)
ae_n = p3.select_slider("Légitimes pour l'autoencodeur", [10000, 50000, 100000], value=50000)
ae_it = p3.slider("Itérations autoencodeur", 5, 100, 40)

st.subheader("3. Lancer")
b1, b2, b3 = st.columns(3)
train_clicked = b1.button("🚀 Entraîner les modèles", type="primary", width="stretch", disabled=not selected)
load_clicked = b2.button("📂 Charger modèles sauvegardés", width="stretch", disabled=not os.path.exists(BUNDLE_PATH))
if "bundle" in st.session_state:
    b3.download_button("⬇️ Exporter les scores (CSV)", width="stretch", file_name="scores.csv", mime="text/csv",
                       data=pd.DataFrame({f"{n}_{e}": s[e] for n, s in st.session_state["bundle"]["scores"].items()
                                          for e in ("val",)}).to_csv(index=False))

if train_clicked:
    bar = st.progress(0.0, "Démarrage…")
    bundle = core.train_models(selected, mats, frac_train=frac_train, ratio_neg=ratio, lof_n=lof_n, ae_n=ae_n,
                               ae_iter=ae_it, n_est=n_est, progress=lambda p, t: bar.progress(min(p, 1.0), t))
    bundle["meta"] = {"n": len(df), "frac": st.session_state.get("frac")}
    st.session_state["bundle"] = bundle
    os.makedirs("models", exist_ok=True)
    try:
        joblib.dump(bundle, BUNDLE_PATH, compress=3)
        st.toast("Modèles sauvegardés dans models/bundle.joblib")
    except Exception as e:
        st.warning(f"Sauvegarde impossible : {e}")
    bar.progress(1.0, "Terminé ✅")

if load_clicked:
    b = joblib.load(BUNDLE_PATH)
    if b.get("meta", {}).get("n") != len(df):
        st.error("Ces modèles ont été entraînés sur un autre échantillon de données (taille différente). "
                 "Rechargez les mêmes données ou réentraînez.")
    else:
        st.session_state["bundle"] = b
        st.success("Modèles chargés.")

if "bundle" in st.session_state:
    b = st.session_state["bundle"]
    st.divider()
    st.subheader("Tableau comparatif")
    comp = core.comparatif(b["scores"], mats)
    comp["Type"] = comp["Méthode"].map(lambda m: "Supervisé" if m in core.SUPERVISED else "Non supervisé")
    comp["Temps (s)"] = comp["Méthode"].map(b["timings"]).round(1)
    st.dataframe(comp.round(4), width="stretch", hide_index=True)
    long = comp.melt(id_vars="Méthode", value_vars=["PR-AUC val", "PR-AUC test"], var_name="Ensemble", value_name="PR-AUC")
    st.plotly_chart(px.bar(long, x="PR-AUC", y="Méthode", color="Ensemble", barmode="group", orientation="h",
                           title="PR-AUC par méthode"), width="stretch")
    st.caption("PR-AUC = métrique principale (événements rares). Le hasard vaut ≈ le taux de fraude. "
               "Passez à la page **Résultats** pour le seuil, les coûts et les matrices de confusion.")