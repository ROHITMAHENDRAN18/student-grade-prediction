"""Statistical EDA: inspection, outliers, univariate / bivariate / multivariate statistics and
distribution fitting. Writes CSV tables and results_eda.json (used by the report)."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from config import EDA_DIR, TARGET, BINARY_COLS, NOMINAL_COLS, PASS_MARK
from data_preprocessing import load_data, clean_data, numeric_cols


def inspect(df):
    """Dtypes + descriptive statistics (mean/median/mode/std/quantiles/skew/kurtosis) of numeric columns."""
    num = numeric_cols(df) + [TARGET]
    pd.DataFrame({"dtype": df.dtypes.astype(str), "n_unique": df.nunique(),
                  "missing": df.isna().sum()}).to_csv(EDA_DIR / "table_dtypes.csv")
    rows = []
    for c in num:
        s = df[c]
        rows.append({"feature": c, "mean": s.mean(), "median": s.median(), "mode": s.mode().iloc[0],
                     "std": s.std(), "min": s.min(), "q25": s.quantile(.25), "q75": s.quantile(.75),
                     "max": s.max(), "skew": s.skew(), "kurtosis": s.kurt()})
    desc = pd.DataFrame(rows).set_index("feature"); desc.round(3).to_csv(EDA_DIR / "table_descriptive_numeric.csv")
    return desc


def outlier_table(df):
    """IQR (1.5x) and |z|>3 outlier counts for numeric columns with more than 5 distinct values."""
    rows = []
    for c in numeric_cols(df) + [TARGET]:
        s = df[c]
        if s.nunique() <= 5:
            continue
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1; lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((s < lo) | (s > hi)).sum())
        rows.append({"feature": c, "lower_fence": lo, "upper_fence": hi, "iqr_outliers": n_iqr,
                     "iqr_pct": 100 * n_iqr / len(s), "z_gt_3": int((np.abs((s - s.mean()) / s.std()) > 3).sum())})
    t = pd.DataFrame(rows).set_index("feature").sort_values("iqr_pct", ascending=False)
    t.round(3).to_csv(EDA_DIR / "table_outliers.csv"); return t


def bivariate_tests(df):
    """Association of every input feature with the numeric target G3.
    numeric/ordinal -> Spearman rho (+p); binary -> Mann-Whitney U with Cohen's d (mean diff / pooled SD);
    nominal (>2 levels) -> Kruskal-Wallis H with epsilon^2 effect size."""
    n = len(df); rows = []
    for c in numeric_cols(df):
        r, p = stats.spearmanr(df[c], df[TARGET])
        rows.append({"feature": c, "type": "numeric/ordinal", "test": "Spearman", "statistic": r, "p_value": p, "effect_size": abs(r)})
    for c in BINARY_COLS:
        g = [x[TARGET].values for _, x in df.groupby(c)]
        U, p = stats.mannwhitneyu(g[0], g[1]); sp = np.sqrt(((len(g[0]) - 1) * g[0].var(ddof=1) + (len(g[1]) - 1) * g[1].var(ddof=1)) / (n - 2))
        d = (g[1].mean() - g[0].mean()) / sp
        rows.append({"feature": c, "type": "binary", "test": "Mann-Whitney U", "statistic": U, "p_value": p, "effect_size": abs(d)})
    for c in NOMINAL_COLS:
        g = [x[TARGET].values for _, x in df.groupby(c)]; H, p = stats.kruskal(*g)
        rows.append({"feature": c, "type": "nominal", "test": "Kruskal-Wallis", "statistic": H, "p_value": p, "effect_size": (H - len(g) + 1) / (n - len(g))})
    t = pd.DataFrame(rows).set_index("feature").sort_values("effect_size", ascending=False)
    t.to_csv(EDA_DIR / "table_bivariate_tests.csv"); return t


def multivariate(df):
    """Spearman matrix (numeric + target), VIF on numeric features and PCA."""
    num = numeric_cols(df)
    corr = df[num + [TARGET]].corr(method="spearman"); corr.round(3).to_csv(EDA_DIR / "table_spearman_corr.csv")
    X = StandardScaler().fit_transform(df[num]); vif = {}
    for i, c in enumerate(num):
        o = np.delete(X, i, 1); r2 = LinearRegression().fit(o, X[:, i]).score(o, X[:, i]); vif[c] = 1 / (1 - r2)
    vif = pd.Series(vif, name="VIF").sort_values(ascending=False); vif.round(2).to_csv(EDA_DIR / "table_vif.csv")
    return corr, vif, PCA().fit(X).explained_variance_ratio_


def distribution_fits(df):
    """Continuous: normal fit of G3 (with and without the zero grades). Discrete: Poisson vs negative binomial
    for absences; Bernoulli/Binomial for pass/fail; frequency distribution of alcohol ratings."""
    out = {}; g3 = df[TARGET]; nz = g3[g3 > 0]
    mu, sd = stats.norm.fit(g3); mu2, sd2 = stats.norm.fit(nz)
    out["G3_all"] = {"mean": mu, "std": sd, "skew": g3.skew(), "excess_kurtosis": g3.kurt(),
                     "dagostino_p": stats.normaltest(g3)[1], "ks_D": stats.kstest(g3, stats.norm(mu, sd).cdf).statistic}
    out["G3_nonzero"] = {"n": int(len(nz)), "mean": mu2, "std": sd2, "skew": nz.skew(), "excess_kurtosis": nz.kurt(),
                         "dagostino_p": stats.normaltest(nz)[1], "ks_D": stats.kstest(nz, stats.norm(mu2, sd2).cdf).statistic}
    out["n_G3_zero"] = int((g3 == 0).sum())
    ab = df["absences"]; m, v = ab.mean(), ab.var()
    out["absences"] = {"mean": m, "variance": v, "dispersion_index": v / m, "zero_share": float((ab == 0).mean())}
    # negative binomial by method of moments (r, p): mean = r(1-p)/p, var = r(1-p)/p^2
    p_nb = m / v; r_nb = m * p_nb / (1 - p_nb); out["absences"].update(nb_r=r_nb, nb_p=p_nb)
    k = np.arange(0, int(ab.max()) + 1); obs = ab.value_counts(normalize=True).reindex(k, fill_value=0).values
    out["absences"]["tv_poisson"] = 0.5 * float(np.abs(obs - stats.poisson.pmf(k, m)).sum())
    out["absences"]["tv_negbin"] = 0.5 * float(np.abs(obs - stats.nbinom.pmf(k, r_nb, p_nb)).sum())
    ps = (g3 >= PASS_MARK).mean(); lo, hi = stats.binomtest(int((g3 >= PASS_MARK).sum()), len(g3)).proportion_ci(method="wilson")
    out["pass_rate"] = ps; out["pass_wilson_ci"] = [lo, hi]
    for c in ["Dalc", "Walc"]:
        out[c + "_pct"] = {int(k): round(100 * v, 1) for k, v in df[c].value_counts(normalize=True).sort_index().items()}
    return out


def run():
    raw = load_data(); df, rep = clean_data(raw)
    desc = inspect(df); ot = outlier_table(df); bt = bivariate_tests(df); corr, vif, evr = multivariate(df)
    out = {"shape": list(raw.shape), "clean_report": rep, "n_numeric": len(numeric_cols(df)), "n_binary": len(BINARY_COLS),
           "n_nominal": len(NOMINAL_COLS), "dtypes": pd.Series(df.dtypes.astype(str)).value_counts().to_dict(),
           "g3_desc": desc.loc[TARGET].round(3).to_dict(), "pass_count": int((df[TARGET] >= PASS_MARK).sum()),
           "outliers": ot.round(2).reset_index().to_dict("records"), "bivariate": bt.reset_index().round(5).to_dict("records"),
           "vif": vif.round(2).to_dict(), "pca_var_ratio": [round(float(x), 4) for x in evr],
           "distribution_fits": distribution_fits(df)}
    # group means used in the narrative
    gm = {}
    for c in ["school", "higher", "sex", "failures", "studytime", "schoolsup", "internet", "address", "romantic", "paid"]:
        gm[c] = df.groupby(c)[TARGET].agg(["mean", "size"]).round(2).reset_index().astype(object).values.tolist()
    out["group_means"] = gm
    out["g3_zero_profile"] = {"mean_G2_of_zero": float(df[df[TARGET] == 0]["G2"].mean()), "mean_absences_of_zero": float(df[df[TARGET] == 0]["absences"].mean()),
                              "mean_absences_others": float(df[df[TARGET] > 0]["absences"].mean()),
                              "zero_with_G2_zero": int(((df[TARGET] == 0) & (df["G2"] == 0)).sum())}
    c = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=abs, ascending=False)
    out["top_corr_pairs"] = [[a, b, round(float(v), 3)] for (a, b), v in c.head(8).items()]
    out["pearson_G2_G3"] = float(df["G2"].corr(df[TARGET])); out["pearson_G1_G3"] = float(df["G1"].corr(df[TARGET]))
    with open(EDA_DIR / "results_eda.json", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: out[k] for k in ["shape", "clean_report", "pass_count", "g3_zero_profile", "top_corr_pairs", "pearson_G2_G3"]}, indent=1, default=str))
    print(bt.head(14)[["type", "statistic", "p_value", "effect_size"]].round(4)); print(vif.head(5)); print(out["distribution_fits"])
    return df


if __name__ == "__main__":
    run()
