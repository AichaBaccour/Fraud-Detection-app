import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import core

st.title("📊 Distributions & visualisations")
df = core.need_data()
df = df.assign(classe=df["is_fraud"].map({0: "Légitime", 1: "Fraude"}))


def taux(col, order=None):
    t = df.groupby(col, observed=True)["is_fraud"].agg(transactions="count", fraudes="sum", taux="mean").reset_index()
    t["taux_%"] = (t["taux"] * 100).round(3)
    return t.drop(columns="taux")


tabs = st.tabs(["Taux de fraude", "Montants", "Temps", "Heure × catégorie", "Variable au choix", "Variables comportementales", "Corrélations"])

with tabs[0]:
    dim = st.selectbox("Taux de fraude selon…", ["category", "hour", "dow", "month", "age_group", "amt_bin", "state", "gender"],
                       format_func=lambda x: {"category": "Catégorie", "hour": "Heure", "dow": "Jour de semaine (0 = lundi)",
                                              "month": "Mois", "age_group": "Tranche d'âge", "amt_bin": "Tranche de montant",
                                              "state": "État", "gender": "Genre"}[x])
    t = taux(dim)
    if dim == "state":
        topn = st.slider("Top N (par nb de fraudes)", 5, 50, 15)
        t = t.sort_values("fraudes", ascending=False).head(topn)
    elif dim in ("category",):
        t = t.sort_values("taux_%", ascending=False)
    fig = px.bar(t, x=dim, y="taux_%", hover_data=["transactions", "fraudes"], title=f"Taux de fraude (%) par {dim}")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(t, width="stretch", hide_index=True)

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df.assign(log_amt=np.log1p(df["amt"])), x="log_amt", color="classe", histnorm="probability density",
                           barmode="overlay", opacity=0.6, nbins=60, title="Distribution de log(1 + montant)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.box(df, x="classe", y="amt", points=False, title="Montant par classe (sans valeurs extrêmes)")
        st.plotly_chart(fig, width="stretch")
    st.plotly_chart(px.bar(taux("amt_bin"), x="amt_bin", y="taux_%", title="Taux de fraude par tranche de montant (%)"),
                    width="stretch")

with tabs[2]:
    m = df.set_index("trans_date_trans_time")["is_fraud"].resample("D").mean().mul(100).rolling(7, min_periods=1).mean()
    st.plotly_chart(px.line(m.reset_index(), x="trans_date_trans_time", y="is_fraud",
                            labels={"is_fraud": "Taux de fraude (%)", "trans_date_trans_time": ""},
                            title="Évolution du taux de fraude (moyenne glissante 7 jours)"), width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(taux("hour"), x="hour", y="taux_%", title="Par heure"), width="stretch")
    d = taux("dow"); d["jour"] = d["dow"].map(dict(enumerate(["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"])))
    c2.plotly_chart(px.bar(d, x="jour", y="taux_%", title="Par jour de semaine"), width="stretch")

with tabs[3]:
    pv = df.pivot_table(index="category", columns="hour", values="is_fraud", aggfunc="mean") * 100
    st.plotly_chart(px.imshow(pv, aspect="auto", color_continuous_scale="Reds",
                              labels=dict(color="Taux (%)", x="Heure"), title="Taux de fraude (%) par catégorie et par heure"),
                    width="stretch")

with tabs[4]:
    num = [c for c in core.NUM_COLS]
    v = st.selectbox("Variable numérique", num)
    kind = st.radio("Graphique", ["Histogramme", "Boîte à moustaches", "Violon"], horizontal=True)
    lg = st.checkbox("Échelle log(1+x)")
    d = df.assign(val=np.log1p(df[v].clip(lower=0)) if lg else df[v])
    if kind == "Histogramme":
        fig = px.histogram(d, x="val", color="classe", histnorm="probability density", barmode="overlay", opacity=.6, nbins=60)
    elif kind == "Boîte à moustaches":
        fig = px.box(d, x="classe", y="val", points=False)
    else:
        fig = px.violin(d.sample(min(len(d), 50000), random_state=0), x="classe", y="val", box=True)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df.groupby("classe")[v].describe().round(3))

with tabs[5]:
    beh = ["n_1h", "n_24h", "amt_ratio_30d", "hours_since_prev", "distance_km"]
    st.dataframe(df.groupby("classe")[beh].mean().round(3), width="stretch")
    c1, c2 = st.columns(2)
    for col, cc in zip(["amt_ratio_30d", "hours_since_prev"], (c1, c2)):
        cc.plotly_chart(px.histogram(df.sample(min(len(df), 200000), random_state=0), x=col, color="classe",
                                     histnorm="probability density", barmode="overlay", opacity=.6, nbins=60,
                                     log_y=True, title=col), width="stretch")
    st.plotly_chart(px.scatter(df[df.is_fraud == 1].sample(min(3000, int(df.is_fraud.sum())), random_state=0),
                               x="distance_km", y="amt", color="category", title="Fraudes : distance vs montant"),
                    width="stretch")

with tabs[6]:
    cols = core.NUM_COLS + ["is_fraud"]
    corr = df[cols].corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Corrélations (Pearson)"), width="stretch")