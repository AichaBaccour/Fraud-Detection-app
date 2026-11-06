"""Fonctions métier partagées (reprises du notebook DataMining_Project_2)."""
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (average_precision_score, roc_auc_score, fbeta_score,
                             matthews_corrcoef, confusion_matrix)

KAGGLE_ID = "kartik2112/fraud-detection"

NUM_COLS = ["amt", "hour_sin", "hour_cos", "dow", "age", "city_pop", "distance_km",
            "n_1h", "n_24h", "amt_ratio_30d", "hours_since_prev"]

DESCRIPTIONS = {
    "trans_date_trans_time": "Date et heure de la transaction",
    "cc_num": "Numéro de carte (identifiant client/carte)",
    "merchant": "Nom du marchand",
    "category": "Catégorie du marchand",
    "amt": "Montant de la transaction ($)",
    "first": "Prénom du titulaire", "last": "Nom du titulaire", "gender": "Genre du titulaire",
    "street": "Adresse", "city": "Ville du titulaire", "state": "État du titulaire", "zip": "Code postal",
    "lat": "Latitude du titulaire", "long": "Longitude du titulaire",
    "city_pop": "Population de la ville du titulaire",
    "job": "Profession", "dob": "Date de naissance",
    "trans_num": "Identifiant unique de la transaction",
    "unix_time": "Horodatage Unix",
    "merch_lat": "Latitude du marchand", "merch_long": "Longitude du marchand",
    "is_fraud": "CIBLE : 1 = fraude, 0 = légitime",
}

LIBELLES = {
    "amt": "Montant de la transaction",
    "hour_sin": "Heure de la transaction", "hour_cos": "Heure de la transaction",
    "dow": "Jour de la semaine", "age": "Âge du client",
    "city_pop": "Population de la ville", "distance_km": "Distance client-marchand",
    "n_1h": "Nombre de transactions dans la dernière heure",
    "n_24h": "Nombre de transactions sur 24 h",
    "amt_ratio_30d": "Montant inhabituel pour cette carte (30 j)",
    "hours_since_prev": "Délai depuis la transaction précédente",
}


def libelle(v):
    return "Catégorie : " + v[4:] if v.startswith("cat_") else LIBELLES.get(v, v)


# ----------------------------------------------------------------------------
# 1. Chargement
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement du jeu de données…")
def load_raw(source: str, uploaded_bytes: bytes | None = None, uploaded_name: str = ""):
    """source = 'kaggle' | 'upload' | 'local'. Retourne le DataFrame brut (train+test concaténés)."""
    import io
    if source == "upload" and uploaded_bytes is not None:
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    elif source == "local":
        import os
        parts = [pd.read_csv(p, index_col=0) for p in ("data/fraudTrain.csv", "data/fraudTest.csv") if os.path.exists(p)]
        if not parts:
            raise FileNotFoundError("Placez fraudTrain.csv et fraudTest.csv dans le dossier data/.")
        df = pd.concat(parts, ignore_index=True)
    else:
        import kagglehub
        path = kagglehub.dataset_download(KAGGLE_ID)
        train = pd.read_csv(f"{path}/fraudTrain.csv", index_col=0)
        test = pd.read_csv(f"{path}/fraudTest.csv", index_col=0)
        df = pd.concat([train, test], ignore_index=True)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df.sort_values("trans_date_trans_time").reset_index(drop=True)

@st.cache_data(show_spinner="Lecture rapide du jeu de données…")
def load_overview(source: str, uploaded_bytes: bytes | None = None):
    """Aperçu léger : 10 premières lignes + statistiques calculées sur 5 colonnes seulement."""
    import io, os
    cols = ["trans_date_trans_time", "cc_num", "category", "amt", "is_fraud"]
    if source == "upload" and uploaded_bytes is not None:
        head = pd.read_csv(io.BytesIO(uploaded_bytes), nrows=10)
        light = pd.read_csv(io.BytesIO(uploaded_bytes), usecols=cols)
    else:
        if source == "local":
            paths = [p for p in ("data/fraudTrain.csv", "data/fraudTest.csv") if os.path.exists(p)]
            if not paths:
                raise FileNotFoundError("Placez fraudTrain.csv et fraudTest.csv dans le dossier data/.")
        else:
            import kagglehub
            d = kagglehub.dataset_download(KAGGLE_ID)
            paths = [f"{d}/fraudTrain.csv", f"{d}/fraudTest.csv"]
        head = pd.read_csv(paths[0], nrows=10)
        light = pd.concat([pd.read_csv(p, usecols=cols) for p in paths], ignore_index=True)
    head = head.drop(columns=[c for c in head.columns if c.startswith("Unnamed")])
    light["trans_date_trans_time"] = pd.to_datetime(light["trans_date_trans_time"])
    n, nf = len(light), int(light["is_fraud"].sum())
    desc = light.groupby("is_fraud")["amt"].describe().round(2)
    desc.index = desc.index.map({0: "Légitime", 1: "Fraude"})
    return {
        "head": head, "n_rows": n, "n_cols": int(head.shape[1]), "n_fraud": nf, "n_legit": n - nf,
        "n_cards": int(light["cc_num"].nunique()), "n_cats": int(light["category"].nunique()),
        "date_min": light["trans_date_trans_time"].min(), "date_max": light["trans_date_trans_time"].max(),
        "amt_by_class": desc, "dtypes": {c: str(t) for c, t in head.dtypes.items()},
        "n_missing": 0, "source": source,
    }
def sample_cards(df, frac, seed=42):
    """Échantillonne des CARTES entières (l'historique de chaque carte reste intact,
    donc les variables de vélocité restent correctes)."""
    if frac >= 1:
        return df
    cards = df["cc_num"].drop_duplicates()
    keep = cards.sample(frac=frac, random_state=seed)
    return df[df["cc_num"].isin(keep)].reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Variables
# ----------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner="Construction des variables…")
def build_features(df_in: pd.DataFrame):
    df = df_in.copy()
    t = df["trans_date_trans_time"]
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    df["month"] = t.dt.to_period("M").astype(str)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["distance_km"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["age"] = (t - pd.to_datetime(df["dob"])).dt.days // 365
    df["age_group"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 65, 120]).astype(str)
    df["amt_bin"] = pd.cut(df["amt"], bins=[0, 10, 50, 100, 250, 500, 1000, 30000]).astype(str)

    # variables comportementales (par carte, uniquement le PASSÉ -> pas de fuite)
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    ts = df["trans_date_trans_time"].astype("int64").values // 10 ** 9
    if ts.max() > 10 ** 12:  # sécurité si pandas renvoie des ns
        ts = ts // 10 ** 9
    amt = df["amt"].values
    cards = df["cc_num"].values
    starts = np.r_[0, np.flatnonzero(cards[1:] != cards[:-1]) + 1, len(df)]
    n_1h = np.zeros(len(df)); n_24h = np.zeros(len(df))
    mean_30d = np.full(len(df), np.nan); since_prev = np.full(len(df), np.nan)
    for a, b in zip(starts[:-1], starts[1:]):
        tt = ts[a:b]
        cs = np.r_[0, np.cumsum(amt[a:b])]
        idx = np.arange(b - a)
        lo1 = np.searchsorted(tt, tt - 3600, side="left")
        lo24 = np.searchsorted(tt, tt - 86400, side="left")
        lo30 = np.searchsorted(tt, tt - 30 * 86400, side="left")
        n_1h[a:b] = idx - lo1
        n_24h[a:b] = idx - lo24
        cnt30 = idx - lo30
        sum30 = cs[idx] - cs[lo30]
        mean_30d[a:b] = np.where(cnt30 > 0, sum30 / np.maximum(cnt30, 1), np.nan)
        since_prev[a + 1:b] = np.diff(tt)
    df["n_1h"], df["n_24h"] = n_1h, n_24h
    df["amt_ratio_30d"] = (df["amt"] / mean_30d)
    df["amt_ratio_30d"] = df["amt_ratio_30d"].fillna(1.0).clip(upper=50)
    df["hours_since_prev"] = (since_prev / 3600)
    df["hours_since_prev"] = df["hours_since_prev"].fillna(720).clip(upper=720)

    # séparation chronologique 60/20/20
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    n = len(df)
    i_train, i_val = int(0.6 * n), int(0.8 * n)
    df["split"] = np.where(df.index < i_train, "train", np.where(df.index < i_val, "val", "test"))
    return df


@st.cache_data(show_spinner=False)
def make_matrices(df: pd.DataFrame):
    X_all = pd.concat([df[NUM_COLS], pd.get_dummies(df["category"], prefix="cat", dtype=int)], axis=1)
    out = {}
    for s in ("train", "val", "test"):
        m = df["split"] == s
        out[s] = (X_all[m], df.loc[m, "is_fraud"], df.loc[m, "amt"])
    return out


# ----------------------------------------------------------------------------
# 3. Entraînement  (mêmes hyper-paramètres que le notebook)
# ----------------------------------------------------------------------------
MODEL_NAMES = ["XGBoost (sous-éch. 10:1)", "Régression logistique pondérée", "Random Forest",
               "LightGBM", "Isolation Forest", "LOF", "Autoencodeur"]
SUPERVISED = MODEL_NAMES[:4]
UNSUPERVISED = MODEL_NAMES[4:]


def train_models(selected, mats, frac_train=1.0, ratio_neg=10, lof_n=20000, ae_n=100000,
                 ae_iter=40, n_est=200, seed=42, progress=None):
    """Entraîne les modèles choisis. Retourne (models, scores, timings)."""
    from sklearn.preprocessing import StandardScaler
    X_train, y_train, _ = mats["train"]
    X_val, y_val, _ = mats["val"]
    X_test, y_test, _ = mats["test"]

    if frac_train < 1:  # accélération : sous-échantillon du train (toutes les fraudes gardées)
        rng0 = np.random.RandomState(seed)
        pos0 = y_train.index[y_train == 1]
        neg0 = y_train.index[y_train == 0]
        neg0 = rng0.choice(neg0, size=int(frac_train * len(neg0)), replace=False)
        keep = np.sort(np.r_[pos0, neg0])
        X_train, y_train = X_train.loc[keep], y_train.loc[keep]

    rng = np.random.RandomState(seed)
    pos = y_train.index[y_train == 1]
    neg = y_train.index[y_train == 0]
    idx = np.r_[pos, rng.choice(neg, size=min(ratio_neg * len(pos), len(neg)), replace=False)]
    scaler = StandardScaler().fit(X_train)

    models, scores, timings = {}, {}, {}
    todo = [m for m in MODEL_NAMES if m in selected]

    for i, nom in enumerate(todo):
        if progress:
            progress(i / len(todo), f"Entraînement : {nom}…")
        t0 = time.time()
        if nom == "XGBoost (sous-éch. 10:1)":
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=n_est, max_depth=6, learning_rate=0.1, tree_method="hist",
                              n_jobs=-1, random_state=seed).fit(X_train.loc[idx], y_train.loc[idx])
            f = lambda X, m=m: m.predict_proba(X)[:, 1]
        elif nom == "Régression logistique pondérée":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(
                scaler.transform(X_train.loc[idx]), y_train.loc[idx])
            f = lambda X, m=m: m.predict_proba(scaler.transform(X))[:, 1]
        elif nom == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(n_estimators=n_est, min_samples_leaf=3, n_jobs=-1,
                                       random_state=seed).fit(X_train.loc[idx], y_train.loc[idx])
            f = lambda X, m=m: m.predict_proba(X)[:, 1]
        elif nom == "LightGBM":
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(n_estimators=int(1.5 * n_est), learning_rate=0.05, num_leaves=31,
                               random_state=seed, verbose=-1).fit(X_train.loc[idx], y_train.loc[idx])
            f = lambda X, m=m: m.predict_proba(X)[:, 1]
        elif nom == "Isolation Forest":
            from sklearn.ensemble import IsolationForest
            m = IsolationForest(n_estimators=n_est, random_state=seed, n_jobs=-1).fit(X_train)
            f = lambda X, m=m: -m.score_samples(X)
        elif nom == "LOF":
            from sklearn.neighbors import LocalOutlierFactor
            ech = X_train.sample(min(lof_n, len(X_train)), random_state=seed)
            m = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(scaler.transform(ech))
            f = lambda X, m=m: -m.score_samples(scaler.transform(X))
        elif nom == "Autoencodeur":
            from sklearn.neural_network import MLPRegressor
            normales = X_train[y_train == 0]
            normales = normales.sample(min(ae_n, len(normales)), random_state=seed)
            Xn = scaler.transform(normales)
            m = MLPRegressor(hidden_layer_sizes=(16, 6, 16), max_iter=ae_iter, random_state=seed).fit(Xn, Xn)
            def f(X, m=m):
                Z = scaler.transform(X)
                return ((Z - m.predict(Z)) ** 2).mean(axis=1)
        models[nom] = m
        scores[nom] = {"val": np.asarray(f(X_val)), "test": np.asarray(f(X_test))}
        timings[nom] = time.time() - t0
        if progress:
            progress((i + 1) / len(todo), f"{nom} terminé ({timings[nom]:.1f} s)")
    return {"models": models, "scores": scores, "timings": timings, "scaler": scaler,
            "columns": list(X_train.columns), "bg": X_train.sample(min(100, len(X_train)), random_state=seed)}


def scorer(bundle, nom):
    """Retourne une fonction DataFrame -> score pour un modèle du bundle."""
    m, sc = bundle["models"][nom], bundle["scaler"]
    if nom in ("XGBoost (sous-éch. 10:1)", "Random Forest", "LightGBM"):
        return lambda X: m.predict_proba(X)[:, 1]
    if nom == "Régression logistique pondérée":
        return lambda X: m.predict_proba(sc.transform(X))[:, 1]
    if nom == "Isolation Forest":
        return lambda X: -m.score_samples(X)
    if nom == "LOF":
        return lambda X: -m.score_samples(sc.transform(X))
    return lambda X: ((sc.transform(X) - m.predict(sc.transform(X))) ** 2).mean(axis=1)


# ----------------------------------------------------------------------------
# 4. Évaluation / coûts
# ----------------------------------------------------------------------------
def precision_rappel_at_k(y, s, k):
    top = np.argsort(-s)[:k]
    tp = np.asarray(y)[top].sum()
    return tp / max(k, 1), tp / max(np.asarray(y).sum(), 1)


def cout_complet(y, s, amt, c_fp, c_tp, mode_fn="montant", facteur=1.0, c_fn_fixe=0.0):
    """Coût total pour k = 0..n alertes (alertes triées par score décroissant)."""
    y, amt, s = np.asarray(y), np.asarray(amt), np.asarray(s)
    ordre = np.argsort(-s)
    yy, aa = y[ordre], amt[ordre]
    k = np.arange(len(s) + 1)
    tp = np.r_[0, np.cumsum(yy)]
    fp = k - tp
    perte = facteur * aa * yy if mode_fn == "montant" else c_fn_fixe * yy
    fn_cout = perte.sum() - np.r_[0, np.cumsum(perte)]
    return fn_cout + c_fp * fp + c_tp * tp, s[ordre]


def sans_modele(y, amt, mode_fn="montant", facteur=1.0, c_fn_fixe=0.0):
    y, amt = np.asarray(y), np.asarray(amt)
    return facteur * (amt * y).sum() if mode_fn == "montant" else c_fn_fixe * y.sum()


def seuil_optimal(y, s, amt, **cost):
    c, tries = cout_complet(y, s, amt, **cost)
    k = int(c.argmin())
    return (np.inf if k == 0 else float(tries[k - 1])), k, float(c.min())


def evaluer_seuil(y, s, amt, seuil, c_fp, c_tp, mode_fn="montant", facteur=1.0, c_fn_fixe=0.0):
    y, s, amt = np.asarray(y), np.asarray(s), np.asarray(amt)
    a = s >= seuil
    fr = y == 1
    tp, fp = int((a & fr).sum()), int((a & ~fr).sum())
    fn_m, tn = fr & ~a, int((~a & ~fr).sum())
    fn = int(fn_m.sum())
    manques = float((facteur * amt[fn_m]).sum() if mode_fn == "montant" else c_fn_fixe * fn)
    cout_alertes = float(c_fp * fp + c_tp * tp)
    ya = a.astype(int)
    return {"alertes": int(a.sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "précision": tp / max(a.sum(), 1), "rappel": tp / max(fr.sum(), 1),
            "F2": fbeta_score(y, ya, beta=2, zero_division=0),
            "MCC": matthews_corrcoef(y, ya) if 0 < a.sum() < len(a) else 0.0,
            "fraudes manquées ($)": manques, "coût alertes ($)": cout_alertes,
            "coût total ($)": manques + cout_alertes,
            "coût sans modèle ($)": float(sans_modele(y, amt, mode_fn, facteur, c_fn_fixe))}


def comparatif(scores, mats):
    rows = []
    for nom, s in scores.items():
        rows.append([nom,
                     average_precision_score(mats["val"][1], s["val"]), roc_auc_score(mats["val"][1], s["val"]),
                     average_precision_score(mats["test"][1], s["test"]), roc_auc_score(mats["test"][1], s["test"])])
    return pd.DataFrame(rows, columns=["Méthode", "PR-AUC val", "ROC-AUC val", "PR-AUC test", "ROC-AUC test"]
                        ).sort_values("PR-AUC val", ascending=False)


# ----------------------------------------------------------------------------
# 5. Garde-fous pour les pages
# ----------------------------------------------------------------------------
def need_data():
    if "df" not in st.session_state:
        st.warning("Aucune donnée chargée : ouvrez d'abord la page **Données** puis cliquez sur « Charger ».")
        st.stop()
    return st.session_state["df"]


def need_models():
    if "bundle" not in st.session_state:
        st.warning("Aucun modèle disponible : allez dans la page **Modèles** et entraînez-les (ou chargez-les).")
        st.stop()
    return st.session_state["bundle"], st.session_state["mats"]