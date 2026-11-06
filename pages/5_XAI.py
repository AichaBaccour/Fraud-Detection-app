import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import streamlit as st
import core

st.title("🔍 Explicabilité (XAI) — SHAP")
bundle, mats = core.need_models()
df = st.session_state["df"]
cols = bundle["columns"]
X_test, y_test, _ = mats["test"]
names = list(bundle["models"])

with st.sidebar:
    st.header("Échantillon à expliquer")
    n_fr = st.slider("Fraudes (test)", 20, 300, 100, 10)
    n_no = st.slider("Légitimes (test)", 50, 600, 250, 50)
    max_evals = st.slider("Évaluations (modèles non supervisés)", 30, 200, 60, 10,
                          help="Plus c'est élevé, plus l'explication par permutation est précise… et lente.")

nom = st.selectbox("Modèle à expliquer", names)


@st.cache_data(show_spinner=False)
def pick_sample(n_fr, n_no, n_test):
    rng = np.random.RandomState(42)
    yv = y_test.values
    return np.r_[rng.choice(np.where(yv == 1)[0], min(n_fr, int((yv == 1).sum())), replace=False),
                 rng.choice(np.where(yv == 0)[0], n_no, replace=False)]


idx = pick_sample(n_fr, n_no, len(X_test))
X_ech, y_ech = X_test.iloc[idx], y_test.iloc[idx]


def classe1(sv):
    if isinstance(sv, list):
        return sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def compute(nom):
    m, sc, bg = bundle["models"][nom], bundle["scaler"], bundle["bg"]
    if nom in ("LightGBM", "XGBoost (sous-éch. 10:1)", "Random Forest"):
        ex = shap.TreeExplainer(m)
        v, base = classe1(ex.shap_values(X_ech, check_additivity=False)), float(np.ravel(ex.expected_value)[-1])
    elif nom == "Régression logistique pondérée":
        ex = shap.LinearExplainer(m, sc.transform(bg))
        v, base = ex.shap_values(sc.transform(X_ech)), float(np.ravel(ex.expected_value)[-1])
    else:
        f_ = core.scorer(bundle, nom)
        e = shap.PermutationExplainer(lambda A: f_(pd.DataFrame(A, columns=cols)), bg.values)(X_ech.values, max_evals=max_evals)
        v, base = e.values, e.base_values
    return shap.Explanation(values=v, base_values=np.full(len(v), base) if np.ndim(base) == 0 else base,
                            data=X_ech.values, feature_names=cols)


key = (nom, n_fr, n_no, max_evals)
store = st.session_state.setdefault("expl", {})
if key not in store:
    with st.spinner(f"Calcul des valeurs SHAP — {nom}…"):
        store[key] = compute(nom)
E = store[key]

tab_g, tab_l, tab_c = st.tabs(["🌍 Explication globale", "🔎 Explication d'une alerte", "⚖️ Comparaison des modèles"])

with tab_g:
    imp = pd.Series(np.abs(E.values).mean(0), index=cols).sort_values(ascending=False)
    imp_df = imp.head(15).rename("|SHAP| moyen").reset_index().rename(columns={"index": "variable"})
    imp_df["libellé"] = imp_df["variable"].map(core.libelle)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(imp_df.iloc[::-1], x="|SHAP| moyen", y="libellé", orientation="h",
                           title="Variables les plus influentes"), width="stretch")
    with c2:
        st.markdown("**Beeswarm** (chaque point = une transaction ; couleur = valeur de la variable)")
        plt.figure()
        shap.plots.beeswarm(E, max_display=12, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

with tab_l:
    sv_all = core.scorer(bundle, nom)(X_ech)
    which = st.radio("Transactions", ["Fraudes", "Légitimes", "Toutes"], horizontal=True)
    order = np.argsort(-sv_all)
    mask = {"Fraudes": y_ech.values == 1, "Légitimes": y_ech.values == 0, "Toutes": np.ones(len(y_ech), bool)}[which]
    cand = [i for i in order if mask[i]]
    amt_col = df.loc[X_ech.index, "amt"].values
    fmt = lambda i: f"#{i} · score {sv_all[i]:.4f} · {amt_col[i]:,.2f} $ · {'FRAUDE' if y_ech.values[i] else 'légitime'}"
    i = st.selectbox("Transaction (triées par score décroissant)", cand, format_func=fmt)
    c1, c2 = st.columns([3, 2])
    with c1:
        plt.figure()
        shap.plots.waterfall(E[i], max_display=10, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
    with c2:
        n_codes = st.slider("Nombre de codes motifs", 3, 5, 4)
        t = pd.Series(E.values[i], index=cols)
        t = t[t > 0].sort_values(ascending=False)
        motifs = list(dict.fromkeys(core.libelle(v) for v in t.index))[:n_codes]
        st.markdown("### Codes motifs")
        if motifs:
            for r, m in enumerate(motifs, 1):
                st.markdown(f"**{r}.** {m}")
        else:
            st.write("Aucune variable ne pousse vers l'alerte.")
        st.markdown("**Valeurs de la transaction**")
        row = X_ech.iloc[[i]].T.loc[core.NUM_COLS]
        row.columns = ["valeur"]
        st.dataframe(row.round(3), width="stretch")

with tab_c:
    st.caption("Part d'influence (|SHAP| moyen normalisé, %). Calcule les explications de tous les modèles (peut être long).")
    if st.button("Calculer pour tous les modèles"):
        for n_ in names:
            k_ = (n_, n_fr, n_no, max_evals)
            if k_ not in store:
                with st.spinner(n_):
                    store[k_] = compute(n_)
    done = {n_: store[(n_, n_fr, n_no, max_evals)] for n_ in names if (n_, n_fr, n_no, max_evals) in store}
    if len(done) > 1:
        tab = pd.DataFrame({n_: 100 * np.abs(e.values).mean(0) / np.abs(e.values).mean(0).sum() for n_, e in done.items()}, index=cols)
        top = tab.mean(axis=1).sort_values(ascending=False).head(12).index
        tab.index = [core.libelle(c) if c.startswith("cat_") else c for c in tab.index]
        top_lbl = [core.libelle(c) if c.startswith("cat_") else c for c in top]
        st.plotly_chart(px.imshow(tab.loc[top_lbl].round(1), text_auto=True, aspect="auto", color_continuous_scale="Blues",
                                  title="Influence des variables selon le modèle (%)"), width="stretch")
    else:
        st.info("Calculez au moins deux modèles pour comparer.")