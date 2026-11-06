import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
import streamlit as st
import core

# ───────────────────────── Style ─────────────────────────
GOOD, WARN, BAD, BLUE, GREY = "#0F9D8A", "#F2A93B", "#E5484D", "#3E63DD", "#889096"
PAL = [BLUE, GOOD, WARN, BAD, "#8E4EC6", "#12A594", GREY]

st.markdown("""
<style>
.kpi{border-radius:14px;padding:14px 16px;border:1px solid rgba(128,128,128,.25);
     background:rgba(128,128,128,.06);border-left:6px solid var(--c,#3E63DD);height:100%}
.kpi .l{font-size:.72rem;opacity:.7;text-transform:uppercase;letter-spacing:.05em}
.kpi .v{font-size:1.65rem;font-weight:700;line-height:1.25}
.kpi .s{font-size:.8rem;opacity:.75}
.cm-title{font-weight:700;margin:2px 0 6px 0}
.cm{display:grid;grid-template-columns:78px 1fr 1fr;gap:6px}
.cm .h{font-size:.75rem;opacity:.7;text-align:center;align-self:end;padding-bottom:2px}
.cm .rl{font-size:.75rem;opacity:.7;display:flex;align-items:center}
.cm .c{border-radius:12px;padding:10px 8px;text-align:center}
.cm .c b{display:block;font-size:1.5rem;line-height:1.2}
.cm .c span{font-size:.72rem;opacity:.8}
.cm .tn{background:rgba(15,157,138,.16)}.cm .tp{background:rgba(15,157,138,.38)}
.cm .fp{background:rgba(242,169,59,.30)}.cm .fn{background:rgba(229,72,77,.30)}
.cm.small .c b{font-size:1.05rem}.cm.small .c{padding:6px 4px}
.story{border-radius:14px;padding:14px 18px;background:rgba(62,99,221,.08);border:1px solid rgba(62,99,221,.25)}
</style>""", unsafe_allow_html=True)


def fmt(x, d=0):
    return f"{x:,.{d}f}".replace(",", " ")


def kpi(label, value, sub="", color=BLUE):
    return f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def confusion_html(r, title="", small=False):
    pr = lambda a, b: f"{100 * a / max(a + b, 1):.1f} %"
    return f"""<div class="cm-title">{title}</div>
<div class="cm {'small' if small else ''}">
 <div></div><div class="h">Pas d'alerte</div><div class="h">Alerte</div>
 <div class="rl">Légitime</div>
 <div class="c tn"><b>{fmt(r['TN'])}</b><span>bien laissées passer · {pr(r['TN'], r['FP'])}</span></div>
 <div class="c fp"><b>{fmt(r['FP'])}</b><span>fausses alertes · {pr(r['FP'], r['TN'])}</span></div>
 <div class="rl">Fraude</div>
 <div class="c fn"><b>{fmt(r['FN'])}</b><span>fraudes manquées · {pr(r['FN'], r['TP'])}</span></div>
 <div class="c tp"><b>{fmt(r['TP'])}</b><span>fraudes détectées · {pr(r['TP'], r['FN'])}</span></div>
</div>"""


# ───────────────────────── Données ─────────────────────────
st.title("🎯 Résultats, seuil d'alerte & coûts")
bundle, mats = core.need_models()
scores = bundle["scores"]
names = list(scores)
y = {e: np.asarray(mats[e][1]) for e in ("val", "test")}
amt = {e: np.asarray(mats[e][2]) for e in ("val", "test")}

# ───────────────────────── Hypothèses de coût (barre latérale) ─────────────────────────
with st.sidebar:
    st.header("💰 Hypothèses de coût")
    st.caption("Réglez ici les coûts : tout le reste de la page se met à jour.")
    c_alert = st.number_input("Coût pour examiner UNE alerte ($)", 0.0, 1000.0, 10.0, 1.0,
                              help="Temps d'un analyste, appel au client, etc. Payé pour chaque alerte, qu'elle soit vraie ou fausse.")
    mode = st.radio("Si une fraude passe inaperçue, on perd…", ["Le montant de la transaction", "Une somme fixe"],
                    help="Le plus réaliste : on perd le montant volé.")
    if mode == "Le montant de la transaction":
        mode_fn, c_fn_fixe = "montant", 0.0
        facteur = st.slider("Multiplicateur de la perte", 0.5, 3.0, 1.0, 0.1,
                            help="1,0 = on perd exactement le montant. 1,5 = on perd en plus 50 % de frais annexes (litiges, image…).")
    else:
        mode_fn, facteur = "fixe", 1.0
        c_fn_fixe = st.number_input("Perte fixe par fraude manquée ($)", 0.0, 10000.0, 500.0, 50.0)
    st.info("**Règle du jeu :** on ajoute le coût d'examen des alertes et les pertes dues aux fraudes qu'on a laissées passer. "
            "Le meilleur seuil est celui qui rend ce total le plus petit.")
cost = dict(c_fp=c_alert, c_tp=c_alert, mode_fn=mode_fn, facteur=facteur, c_fn_fixe=c_fn_fixe)

tab_sim, tab_cost, tab_curves, tab_recap = st.tabs(
    ["🎚️ Seuil & matrice de confusion", "💰 Comprendre les coûts", "📈 Courbes & comparatif", "📑 Récapitulatif des modèles"])

# ───────────────────────── Onglet 1 : simulateur de seuil ─────────────────────────
with tab_sim:
    model = st.selectbox("Modèle", names, key="model_sim")
    sv, st_ = scores[model]["val"], scores[model]["test"]
    s_opt, k_opt, c_opt = core.seuil_optimal(y["val"], sv, amt["val"], **cost)
    k_opt = max(k_opt, 1)
    kmax = int(min(len(sv), max(2000, 4 * k_opt)))
    sv_sorted = np.sort(sv)[::-1]
    lo, hi = float(sv_sorted[kmax - 1]), float(sv_sorted[0])
    ss = st.session_state
    kkey, tkey = f"k_{model}", f"t_{model}"
    thr_of_k = lambda k: float(sv_sorted[int(np.clip(k, 1, len(sv)) - 1)])

    if kkey not in ss:
        ss[kkey] = int(min(k_opt, kmax)); ss[tkey] = thr_of_k(ss[kkey])

    def on_k():
        ss[tkey] = thr_of_k(ss[kkey])

    def on_t():
        k = int(np.clip((sv >= ss[tkey]).sum(), 1, kmax))
        ss[kkey] = k; ss[tkey] = thr_of_k(k)

    def reset():
        ss[kkey] = int(min(k_opt, kmax)); ss[tkey] = thr_of_k(ss[kkey])

    cA, cB, cC = st.columns([3, 2, 1.3])
    cA.slider("Nombre d'alertes sur la validation  (plus on monte, plus le seuil est bas)", 1, kmax, key=kkey, on_change=on_k,
              help="Faites glisser : le seuil, les métriques, la matrice de confusion et les coûts se mettent à jour.")
    cB.number_input("…ou saisir le seuil exact (score ≥ seuil)", lo, hi, key=tkey, format="%.6f",
                    step=(hi - lo) / 1000 if hi > lo else 0.001, on_change=on_t)
    cC.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    cC.button("↩️ Seuil optimal", on_click=reset, width="stretch")

    seuil = ss[tkey]
    ev = core.evaluer_seuil(y["val"], sv, amt["val"], seuil, **cost)
    et = core.evaluer_seuil(y["test"], st_, amt["test"], seuil, **cost)
    et_opt = core.evaluer_seuil(y["test"], st_, amt["test"], s_opt, **cost)
    eco_test = et["coût sans modèle ($)"] - et["coût total ($)"]

    st.caption(f"Seuil actuel : **{seuil:.6g}** · seuil optimal (coût minimal sur la validation) : **{s_opt:.6g}** ({k_opt} alertes).")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(kpi("Alertes (test)", fmt(et["alertes"]), f"sur {fmt(len(st_))} transactions", BLUE), unsafe_allow_html=True)
    k2.markdown(kpi("Précision", f"{et['précision']:.1%}", "alertes qui sont de vraies fraudes", GOOD), unsafe_allow_html=True)
    k3.markdown(kpi("Rappel", f"{et['rappel']:.1%}", "fraudes détectées", GOOD), unsafe_allow_html=True)
    k4.markdown(kpi("F2 · MCC", f"{et['F2']:.2f} · {et['MCC']:.2f}", "F2 favorise le rappel", WARN), unsafe_allow_html=True)
    k5.markdown(kpi("Coût total", f"{fmt(et['coût total ($)'])} $", f"sans modèle : {fmt(et['coût sans modèle ($)'])} $", BAD), unsafe_allow_html=True)
    k6.markdown(kpi("Économie nette", f"{fmt(eco_test)} $",
                    f"{100 * eco_test / max(et['coût sans modèle ($)'], 1e-9):.1f} % vs sans modèle", GOOD if eco_test > 0 else BAD),
                unsafe_allow_html=True)

    gap = et["coût total ($)"] - et_opt["coût total ($)"]
    if abs(gap) < 1:
        st.success("✅ Vous êtes au seuil optimal.")
    elif gap > 0:
        st.warning(f"⚠️ Par rapport au seuil optimal, ce réglage coûte **{fmt(gap)} $ de plus** sur le test.")
    else:
        st.info(f"Sur le test, ce réglage coûte {fmt(-gap)} $ de moins que le seuil optimal de la validation (c'est normal : l'optimum est choisi sur la validation).")

    st.markdown("#### Matrices de confusion (elles bougent avec le seuil)")
    m1, m2 = st.columns(2)
    m1.markdown(confusion_html(ev, "Validation (là où on choisit le seuil)"), unsafe_allow_html=True)
    m2.markdown(confusion_html(et, "Test (la vérité finale)"), unsafe_allow_html=True)

    st.markdown("#### Balayage du seuil")
    ens_b = st.radio("Ensemble", ["val", "test"], horizontal=True, format_func=lambda e: {"val": "Validation", "test": "Test"}[e])
    s_, y_, a_ = scores[model][ens_b], y[ens_b], amt[ens_b]
    ordre = np.argsort(-s_)
    ks = np.unique(np.linspace(1, kmax, 400).astype(int))
    tp = np.cumsum(y_[ordre])[ks - 1]
    prec, rec = tp / ks, tp / max(y_.sum(), 1)
    f2 = np.where(prec + rec > 0, 5 * prec * rec / np.maximum(4 * prec + rec, 1e-12), 0)
    c_all, _ = core.cout_complet(y_, s_, a_, **cost)
    sweep = pd.DataFrame({"Alertes (k)": ks, "Seuil": s_[ordre][ks - 1], "Précision": prec, "Rappel": rec, "F2": f2, "Coût ($)": c_all[ks]})
    k_cur = int((s_ >= seuil).sum())
    g1, g2 = st.columns(2)
    f = go.Figure()
    for c, col in (("Précision", BLUE), ("Rappel", GOOD), ("F2", WARN)):
        f.add_scatter(x=sweep["Alertes (k)"], y=sweep[c], name=c, line=dict(color=col, width=3))
    f.add_vline(x=k_cur, line_dash="dash", line_color=BAD, annotation_text="seuil choisi")
    f.update_layout(title="Précision, rappel et F2 selon le nombre d'alertes", xaxis_title="Alertes", yaxis_tickformat=".0%",
                    legend=dict(orientation="h", y=-0.25), margin=dict(t=50))
    g1.plotly_chart(f, width="stretch")
    f = go.Figure(go.Scatter(x=sweep["Alertes (k)"], y=sweep["Coût ($)"], line=dict(color=BLUE, width=3), name="Coût total",
                             fill="tozeroy", fillcolor="rgba(62,99,221,.08)"))
    f.add_hline(y=core.sans_modele(y_, a_, mode_fn, facteur, c_fn_fixe), line_dash="dot", line_color=GREY, annotation_text="sans modèle")
    f.add_vline(x=int(np.argmin(c_all)), line_dash="dash", line_color=GOOD, annotation_text="optimum")
    f.add_vline(x=k_cur, line_dash="dash", line_color=BAD, annotation_text="choisi", annotation_position="bottom right")
    f.update_layout(title="Coût total selon le nombre d'alertes", xaxis_title="Alertes", yaxis_title="$", margin=dict(t=50))
    g2.plotly_chart(f, width="stretch")
    with st.expander("Voir / télécharger le tableau du balayage"):
        st.dataframe(sweep.iloc[::10].round(5), width="stretch", hide_index=True)
        st.download_button("⬇️ Balayage complet (CSV)", sweep.to_csv(index=False), f"balayage_{model}_{ens_b}.csv", "text/csv")

# ───────────────────────── Onglet 2 : coûts expliqués ─────────────────────────
with tab_cost:
    st.markdown("Sur la **période de test**, voici ce que coûte la fraude selon la façon dont on réagit. "
                "Les hypothèses sont dans la barre de gauche.")
    sans = et["coût sans modèle ($)"]
    scen = pd.DataFrame([
        {"Scénario": "① Sans modèle\n(on ne vérifie rien)", "Fraudes non détectées": sans, "Examen des alertes": 0.0},
        {"Scénario": f"② Seuil choisi\n({fmt(et['alertes'])} alertes)", "Fraudes non détectées": et["fraudes manquées ($)"], "Examen des alertes": et["coût alertes ($)"]},
        {"Scénario": f"③ Seuil optimal\n({fmt(et_opt['alertes'])} alertes)", "Fraudes non détectées": et_opt["fraudes manquées ($)"], "Examen des alertes": et_opt["coût alertes ($)"]},
    ])
    scen["Total"] = scen["Fraudes non détectées"] + scen["Examen des alertes"]

    c1, c2, c3 = st.columns(3)
    for col, (_, row), colr in zip((c1, c2, c3), scen.iterrows(), (GREY, BLUE, GOOD)):
        col.markdown(kpi(row["Scénario"].replace("\n", " "), f"{fmt(row['Total'])} $",
                         f"pertes {fmt(row['Fraudes non détectées'])} $ + examen {fmt(row['Examen des alertes'])} $", colr),
                     unsafe_allow_html=True)

    st.write("")
    fig = go.Figure()
    fig.add_bar(y=scen["Scénario"], x=scen["Fraudes non détectées"], orientation="h", name="Fraudes non détectées (pertes)", marker_color=BAD)
    fig.add_bar(y=scen["Scénario"], x=scen["Examen des alertes"], orientation="h", name="Examen des alertes", marker_color=BLUE)
    fig.update_layout(barmode="stack", title="Décomposition du coût total (test)", xaxis_title="$",
                      yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.25), margin=dict(t=50), height=330)
    st.plotly_chart(fig, width="stretch")

    perte_evitee = sans - et["fraudes manquées ($)"]
    gaspille = c_alert * et["FP"]
    st.markdown(f"""<div class="story">
<b>En clair ({model}, seuil choisi) :</b><br>
• Sur le test, il y avait <b>{fmt(et['TP'] + et['FN'])} fraudes</b> pour <b>{fmt(sans)} $</b> de pertes potentielles.<br>
• Le modèle déclenche <b>{fmt(et['alertes'])} alertes</b> : <b>{fmt(et['TP'])}</b> sont de vraies fraudes, <b>{fmt(et['FP'])}</b> sont de fausses alertes.<br>
• Il évite <b>{fmt(perte_evitee)} $</b> de pertes, mais examiner les alertes coûte <b>{fmt(et['coût alertes ($)'])} $</b>
 (dont <b>{fmt(gaspille)} $</b> « gaspillés » sur de fausses alertes).<br>
• <b>Bilan : {fmt(eco_test)} $ économisés</b> ({100 * eco_test / max(sans, 1e-9):.1f} %) par rapport à ne rien faire.
</div>""", unsafe_allow_html=True)

    st.markdown("#### Où est le bon réglage ?")
    st.caption("Courbe du coût total en fonction du nombre d'alertes (validation). Le point vert est le minimum : "
               "avant lui on rate trop de fraudes, après lui on paye trop d'alertes pour peu de fraudes en plus.")
    cc, _ = core.cout_complet(y["val"], sv, amt["val"], **cost)
    xmax = min(kmax, len(cc))
    k_now = int((sv >= seuil).sum())
    f = go.Figure(go.Scatter(x=np.arange(xmax), y=cc[:xmax], line=dict(color=BLUE, width=3), name="Coût total",
                             fill="tozeroy", fillcolor="rgba(62,99,221,.08)"))
    f.add_scatter(x=[k_opt], y=[cc[k_opt]], mode="markers+text", text=["optimum"], textposition="top center",
                  marker=dict(size=14, color=GOOD), name="Optimum")
    f.add_scatter(x=[min(k_now, xmax - 1)], y=[cc[min(k_now, xmax - 1)]], mode="markers+text", text=["choisi"],
                  textposition="bottom center", marker=dict(size=14, color=BAD), name="Choisi")
    f.add_hline(y=core.sans_modele(y["val"], amt["val"], mode_fn, facteur, c_fn_fixe), line_dash="dot", line_color=GREY,
                annotation_text="sans modèle")
    f.update_layout(xaxis_title="Nombre d'alertes", yaxis_title="Coût total ($)", margin=dict(t=20), height=380,
                    legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(f, width="stretch")

# ───────────────────────── Onglet 3 : courbes ─────────────────────────
with tab_curves:
    st.dataframe(core.comparatif(scores, mats).round(4), width="stretch", hide_index=True)
    ens = st.radio("Ensemble", ["test", "val"], horizontal=True, key="ens_curves",
                   format_func=lambda e: {"val": "Validation", "test": "Test"}[e])
    sel = st.multiselect("Modèles à afficher", names, default=names, key="sel_curves")
    c1, c2 = st.columns(2)
    fp_, fr_ = go.Figure(), go.Figure()
    for i, n in enumerate(sel):
        col = PAL[names.index(n) % len(PAL)]
        p, r, _ = precision_recall_curve(y[ens], scores[n][ens])
        fp_.add_scatter(x=r, y=p, mode="lines", line=dict(color=col, width=2.5),
                        name=f"{n} ({average_precision_score(y[ens], scores[n][ens]):.3f})")
        f_, t_, _ = roc_curve(y[ens], scores[n][ens])
        fr_.add_scatter(x=f_, y=t_, mode="lines", line=dict(color=col, width=2.5),
                        name=f"{n} ({roc_auc_score(y[ens], scores[n][ens]):.3f})")
    fp_.add_hline(y=y[ens].mean(), line_dash="dash", line_color=GREY, annotation_text="hasard")
    fr_.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=GREY), name="hasard")
    fp_.update_layout(title="Courbes précision-rappel", xaxis_title="Rappel", yaxis_title="Précision", legend=dict(orientation="h", y=-0.3))
    fr_.update_layout(title="Courbes ROC", xaxis_title="Taux de faux positifs", yaxis_title="Rappel", legend=dict(orientation="h", y=-0.3))
    c1.plotly_chart(fp_, width="stretch"); c2.plotly_chart(fr_, width="stretch")

    st.markdown("#### Précision@k et rappel@k (budget d'alertes)")
    parts = st.multiselect("Budget (% des transactions)", [0.1, 0.2, 0.5, 1.0], default=[0.2, 0.5])
    rows = []
    for n in names:
        for e in ("val", "test"):
            for part in parts:
                k = int(part / 100 * len(y[e]))
                p, r = core.precision_rappel_at_k(y[e], scores[n][e], k)
                rows.append([n, e, f"{part} %", k, round(p, 4), round(r, 4)])
    st.dataframe(pd.DataFrame(rows, columns=["Méthode", "Ensemble", "Budget", "k", "Précision@k", "Rappel@k"]),
                 width="stretch", hide_index=True)

# ───────────────────────── Onglet 4 : récapitulatif ─────────────────────────
with tab_recap:
    rows = []
    for n_ in names:
        so, ko, cv = core.seuil_optimal(y["val"], scores[n_]["val"], amt["val"], **cost)
        e_ = core.evaluer_seuil(y["test"], scores[n_]["test"], amt["test"], so, **cost)
        rows.append({"Méthode": n_, "Seuil": so, "Alertes val": ko, "Coût val ($)": round(cv),
                     "Alertes test": e_["alertes"], "Précision test": e_["précision"], "Rappel test": e_["rappel"],
                     "F2 test": e_["F2"], "MCC test": e_["MCC"], "TP": e_["TP"], "FP": e_["FP"], "FN": e_["FN"], "TN": e_["TN"],
                     "Fraudes manquées ($)": round(e_["fraudes manquées ($)"]), "Coût des alertes ($)": round(e_["coût alertes ($)"]),
                     "Coût test ($)": round(e_["coût total ($)"]),
                     "Économie test ($)": round(e_["coût sans modèle ($)"] - e_["coût total ($)"]),
                     "Économie test (%)": round(100 * (1 - e_["coût total ($)"] / max(e_["coût sans modèle ($)"], 1e-9)), 1)})
    r = pd.DataFrame(rows).sort_values("Coût test ($)").reset_index(drop=True)
    best = r.iloc[0]
    st.markdown(kpi("Meilleur modèle (coût test, seuil optimal sur validation)", best["Méthode"],
                    f"coût {fmt(best['Coût test ($)'])} $ · économie {fmt(best['Économie test ($)'])} $ ({best['Économie test (%)']} %)", GOOD),
                unsafe_allow_html=True)
    st.write("")
    st.dataframe(r.drop(columns=["TP", "FP", "FN", "TN"]).round(4), width="stretch", hide_index=True)

    fig = go.Figure()
    fig.add_bar(x=r["Méthode"], y=r["Fraudes manquées ($)"], name="Fraudes non détectées", marker_color=BAD)
    fig.add_bar(x=r["Méthode"], y=r["Coût des alertes ($)"], name="Examen des alertes", marker_color=BLUE)
    fig.update_layout(barmode="stack", title="Coût total par modèle (test, seuil optimal)", yaxis_title="$",
                      legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Matrices de confusion — tous les modèles (test, seuil optimal)")
    cols = st.columns(3)
    for i, row in r.iterrows():
        cols[i % 3].markdown(confusion_html(row, row["Méthode"], small=True), unsafe_allow_html=True)
        cols[i % 3].write("")
    st.download_button("⬇️ Récapitulatif (CSV)", r.to_csv(index=False), "recapitulatif_modeles.csv", "text/csv")