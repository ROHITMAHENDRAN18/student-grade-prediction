"""EDA visualisations (Figures 1-12) + two interactive Plotly pages (stage 7: Interact)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import (EDA_DIR, TARGET, PASS_MARK, BLUE, ORANGE, GREEN, PASS_COLORS, SEED,
                    BINARY_COLS, set_style, save_fig)
from data_preprocessing import load_data, clean_data, numeric_cols

NICE = {"higher": "Wants higher education", "school": "School (GP vs MS)", "address": "Home address (urban vs rural)",
        "internet": "Internet at home", "sex": "Sex (F vs M)", "paid": "Extra paid classes", "schoolsup": "School support",
        "romantic": "In a relationship", "famsup": "Family support", "activities": "Extra-curricular activities",
        "nursery": "Attended nursery", "famsize": "Family size (LE3 vs GT3)", "Pstatus": "Parents (A apart vs T together)"}


def _pf(df):
    return np.where(df[TARGET] >= PASS_MARK, "Pass", "Fail")


def fig01_target(df):
    """Distribution of the target G3 (histogram with the pass mark)."""
    fig, ax = plt.subplots(figsize=(7, 3.9)); s = df[TARGET]
    ax.bar(*np.unique(s, return_counts=True), color=[ORANGE if v < PASS_MARK else BLUE for v in np.unique(s)], width=0.8)
    ax.axvline(PASS_MARK - 0.5, color=INK2 if False else "#52514e", ls="--", lw=1.3); ax.set_ylim(0, 118); ax.text(PASS_MARK - 0.3, 112, "pass mark (10)", fontsize=8)
    ax.annotate(f"{(s == 0).sum()} students\nwith G3 = 0", xy=(0, (s == 0).sum()), xytext=(2.2, 70), fontsize=8, arrowprops=dict(arrowstyle="->", color="#52514e"))
    ax.set(title="Fig. 1  Distribution of the final grade (G3)", xlabel="Final grade G3 (0-20)", ylabel="Number of students")
    save_fig(fig, EDA_DIR / "fig01_target_distribution.png")


def fig02_hist(df):
    """Histograms with mean (solid) and median (dashed) for six numeric variables."""
    cols = ["G1", "G2", "age", "absences", "studytime", "failures"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, c in zip(axes.ravel(), cols):
        s = df[c]; bins = np.arange(s.min() - .5, s.max() + 1.5, 1) if s.nunique() < 25 else 30
        ax.hist(s, bins=bins, color=BLUE, edgecolor="white", linewidth=0.5)
        ax.axvline(s.mean(), color="#0b0b0b", lw=1.6, label=f"mean {s.mean():.1f}"); ax.axvline(s.median(), color=ORANGE, lw=1.6, ls="--", label=f"median {s.median():.1f}")
        ax.set(title=c, xlabel=c, ylabel="Frequency"); ax.legend(fontsize=8)
    fig.suptitle("Fig. 2  Histograms of key numeric variables", fontweight="bold", y=1.0); fig.tight_layout()
    save_fig(fig, EDA_DIR / "fig02_histograms.png")


def fig03_box(df):
    """Boxplots of G3 grouped by five factors."""
    cols = ["school", "failures", "studytime", "higher", "Dalc"]
    labs = {"school": "School", "failures": "Past failures", "studytime": "Weekly study time (1-4)", "higher": "Wants higher education", "Dalc": "Workday alcohol (1-5)"}
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.8), sharey=True)
    for ax, c in zip(axes, cols):
        sns.boxplot(data=df, x=c, y=TARGET, color=BLUE, width=0.6, linewidth=1, fliersize=2, ax=ax)
        ax.set_title(labs[c], fontsize=9); ax.set_xlabel(""); ax.set_ylabel("Final grade G3" if c == "school" else "")
    fig.suptitle("Fig. 3  Final grade (G3) by key factors", fontweight="bold", y=1.03); fig.tight_layout()
    save_fig(fig, EDA_DIR / "fig03_boxplots_g3_by_factor.png")


def fig04_outliers(df):
    """Share of IQR outliers per numeric feature."""
    t = pd.read_csv(EDA_DIR / "table_outliers.csv", index_col=0).sort_values("iqr_pct")
    fig, ax = plt.subplots(figsize=(7, 3.8)); ax.barh(t.index, t["iqr_pct"], color=BLUE, height=0.65)
    for y, v in enumerate(t["iqr_pct"]): ax.text(v + 0.2, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set(title="Fig. 4  Share of IQR outliers per numeric feature", xlabel="Rows outside Q1 - 1.5 IQR / Q3 + 1.5 IQR (%)"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, t["iqr_pct"].max() * 1.15); save_fig(fig, EDA_DIR / "fig04_outlier_share.png")


def fig05_normal(df):
    """Continuous distribution: normal fit and Q-Q plot of G3 for (a) all students, (b) students with G3 > 0."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8)); s = df[TARGET]; nz = s[s > 0]
    for ax, d, ttl in [(axes[0], s, "(a) G3, all students"), (axes[1], nz, "(b) G3 > 0 (zero grades removed)")]:
        mu, sd = stats.norm.fit(d); ax.hist(d, bins=np.arange(d.min() - .5, d.max() + 1.5, 1), density=True, color=BLUE, edgecolor="white", linewidth=.5, label="Observed")
        x = np.linspace(d.min(), d.max(), 300); ax.plot(x, stats.norm.pdf(x, mu, sd), color=ORANGE, lw=2, label=f"Normal (μ={mu:.1f}, σ={sd:.1f})")
        ax.set(title=ttl, xlabel="G3", ylabel="Density"); ax.legend(fontsize=8)
    stats.probplot(s, dist="norm", plot=axes[2]); axes[2].get_lines()[0].set(color=BLUE, markersize=3, label="all"); axes[2].get_lines()[1].set(color=ORANGE, lw=2)
    z = stats.probplot(nz, dist="norm", fit=False); axes[2].plot(z[0], z[1], ".", color=GREEN, markersize=3, label="G3 > 0")
    axes[2].set(title="(c) Normal Q-Q plot", xlabel="Theoretical quantiles", ylabel="Ordered values"); axes[2].legend(fontsize=8)
    fig.suptitle("Fig. 5  Continuous distribution: final grade G3", fontweight="bold", y=1.03); fig.tight_layout()
    save_fig(fig, EDA_DIR / "fig05_normal_fit_qq.png")


def fig06_discrete(df):
    """Discrete distributions: (a) Poisson vs negative binomial for absences, (b) Binomial(10, p) for passes."""
    ab = df["absences"]; m = ab.mean(); v = ab.var(); p_nb = m / v; r_nb = m * p_nb / (1 - p_nb)
    k = np.arange(0, 26); obs = ab.value_counts(normalize=True).reindex(k, fill_value=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9)); w = 0.28
    a.bar(k - w, obs.values, width=w, color=BLUE, label="Observed"); a.bar(k, stats.poisson.pmf(k, m), width=w, color=ORANGE, label=f"Poisson (λ={m:.2f})")
    a.bar(k + w, stats.nbinom.pmf(k, r_nb, p_nb), width=w, color=GREEN, label=f"Negative binomial (r={r_nb:.2f})")
    a.set(title="(a) Number of school absences (shown 0-25)", xlabel="Absences", ylabel="Probability"); a.legend(fontsize=8)
    p = (df[TARGET] >= PASS_MARK).mean(); n = 10; rng = np.random.default_rng(SEED)
    sims = np.array([(df[TARGET].sample(n, random_state=int(r)) >= PASS_MARK).sum() for r in rng.integers(0, 10**6, 5000)])
    kk = np.arange(0, n + 1); emp = np.bincount(sims, minlength=n + 1) / len(sims)
    b.bar(kk - .2, emp, width=.4, color=BLUE, label="Simulated (5000 samples of 10 students)"); b.bar(kk + .2, stats.binom.pmf(kk, n, p), width=.4, color=ORANGE, label=f"Binomial(10, p={p:.3f})")
    b.set(title="(b) Passing students among 10 random students", xlabel="Number who pass", ylabel="Probability"); b.legend(fontsize=8, loc="upper left")
    fig.suptitle("Fig. 6  Discrete distributions: Poisson, negative binomial and Binomial", fontweight="bold", y=1.03); fig.tight_layout()
    save_fig(fig, EDA_DIR / "fig06_discrete_distributions.png")


def fig07_corr(df):
    """Spearman correlation heat-map (numeric/ordinal features and G3)."""
    cols = numeric_cols(df) + [TARGET]; corr = df[cols].corr(method="spearman"); fig, ax = plt.subplots(figsize=(8.4, 7))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, bool), 1), cmap=sns.diverging_palette(40, 250, s=85, l=50, as_cmap=True), center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7}, linewidths=.4, linecolor="white", cbar_kws={"label": "Spearman ρ", "shrink": .7}, ax=ax)
    ax.grid(False); ax.set_title("Fig. 7  Spearman correlation heatmap"); save_fig(fig, EDA_DIR / "fig07_correlation_heatmap.png")


def fig08_group_means(df):
    """Mean G3 with 95% CI for the two levels of each binary factor, sorted by size of the gap."""
    rows = []
    for c in BINARY_COLS:
        for lv, g in df.groupby(c)[TARGET]:
            rows.append({"factor": c, "level": lv, "mean": g.mean(), "ci": 1.96 * g.std(ddof=1) / np.sqrt(len(g)), "n": len(g)})
    t = pd.DataFrame(rows); gap = t.groupby("factor")["mean"].agg(lambda s: abs(s.iloc[0] - s.iloc[1])).sort_values()
    fig, ax = plt.subplots(figsize=(7.6, 5.4)); y = np.arange(len(gap))
    for i, f in enumerate(gap.index):
        a, b = t[t.factor == f].iloc[0], t[t.factor == f].iloc[1]
        ax.plot([a["mean"], b["mean"]], [i, i], color="#b9b8b2", lw=2, zorder=1)
        ax.errorbar(a["mean"], i, xerr=a["ci"], fmt="o", color=ORANGE, ms=7, capsize=3, zorder=3); ax.errorbar(b["mean"], i, xerr=b["ci"], fmt="o", color=BLUE, ms=7, capsize=3, zorder=3)
        ax.text(max(a["mean"] + a["ci"], b["mean"] + b["ci"]) + .15, i, f"{a['level']} (n={a['n']})  vs  {b['level']} (n={b['n']})", va="center", fontsize=7, color="#52514e")
    ax.set_yticks(y); ax.set_yticklabels([NICE.get(f, f) for f in gap.index]); ax.set(xlabel="Mean final grade G3 (95% confidence interval)", xlim=(7.5, 17.5), title="Fig. 8  Mean G3 for each binary factor")
    ax.scatter([], [], color=ORANGE, label="first level (alphabetical)"); ax.scatter([], [], color=BLUE, label="second level"); ax.legend(loc="lower right", fontsize=8); ax.grid(axis="y", visible=False)
    save_fig(fig, EDA_DIR / "fig08_group_means.png")


def fig09_scatter(df):
    """Scatter plots of G1 and G2 against G3 with regression line (jittered, coloured by pass/fail)."""
    rng = np.random.default_rng(SEED); pf = _pf(df); fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, c in zip(axes, ["G1", "G2"]):
        for lab in ["Pass", "Fail"]:
            m = pf == lab; ax.scatter(df[c][m] + rng.normal(0, .12, m.sum()), df[TARGET][m] + rng.normal(0, .12, m.sum()), s=14, alpha=.5, color=PASS_COLORS[lab], edgecolor="none", label=lab)
        b, a = np.polyfit(df[c], df[TARGET], 1); xs = np.array([df[c].min(), df[c].max()]); ax.plot(xs, a + b * xs, color="#0b0b0b", lw=1.6, label=f"line: G3 = {a:.2f} + {b:.2f}·{c}")
        r = df[c].corr(df[TARGET]); ax.set(title=f"{c} vs G3 (Pearson r = {r:.3f})", xlabel=f"{c} (0-20)", ylabel="G3 (0-20)" if c == "G1" else ""); ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Fig. 9  Earlier grades against the final grade", fontweight="bold", y=1.02); fig.tight_layout(); save_fig(fig, EDA_DIR / "fig09_grade_scatter.png")


def fig10_pca(df):
    """PCA projection of standardised numeric features, coloured by G3 (sequential scale)."""
    num = numeric_cols(df); Z = StandardScaler().fit_transform(df[num]); pca = PCA(2, random_state=SEED).fit(Z); P = pca.transform(Z)
    fig, ax = plt.subplots(figsize=(6.6, 5.2)); sc = ax.scatter(P[:, 0], P[:, 1], c=df[TARGET], cmap="viridis", s=14, alpha=.8, edgecolor="none")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Final grade G3"); ev = pca.explained_variance_ratio_ * 100
    ax.set(title="Fig. 10  PCA projection of numeric features", xlabel=f"PC1 ({ev[0]:.1f}% variance)", ylabel=f"PC2 ({ev[1]:.1f}% variance)"); save_fig(fig, EDA_DIR / "fig10_pca.png")


def fig11_pair(df):
    """Pairplot of key variables coloured by pass/fail."""
    d = df[["G2", "failures", "studytime", "absences", "age", TARGET]].copy(); d["Outcome"] = _pf(df)
    g = sns.pairplot(d, hue="Outcome", hue_order=["Pass", "Fail"], palette=PASS_COLORS, corner=True, plot_kws={"s": 9, "alpha": .4, "edgecolor": "none"}, diag_kind="kde", height=1.8)
    g.figure.suptitle("Fig. 11  Pairplot of key variables", fontweight="bold", y=1.01); g.figure.savefig(EDA_DIR / "fig11_pairplot.png"); plt.close(g.figure)


def fig12_heat(df):
    """Mean G3 for each combination of study time and past failures (cell counts printed)."""
    m = df.pivot_table(index="studytime", columns="failures", values=TARGET, aggfunc="mean"); n = df.pivot_table(index="studytime", columns="failures", values=TARGET, aggfunc="size")
    fig, ax = plt.subplots(figsize=(6.2, 4.4)); sns.heatmap(m, annot=False, cmap="Blues", vmin=6, vmax=15, linewidths=1, linecolor="white", cbar_kws={"label": "Mean G3"}, ax=ax)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if not np.isnan(m.iloc[i, j]): ax.text(j + .5, i + .5, f"{m.iloc[i, j]:.1f}\n(n={int(n.iloc[i, j])})", ha="center", va="center", fontsize=8, color="white" if m.iloc[i, j] > 11.5 else "#0b0b0b")
    ax.grid(False); ax.set(title="Fig. 12  Mean G3 by study time and past failures", xlabel="Past class failures", ylabel="Weekly study time (1 = <2 h … 4 = >10 h)"); save_fig(fig, EDA_DIR / "fig12_studytime_failures.png")


def interactive(df):
    """Stage 7 (Interact): hover / zoom / legend-filter pages built with Plotly."""
    import plotly.express as px
    d = df.copy(); d["Outcome"] = _pf(df)
    f1 = px.scatter(d, x="G2", y=TARGET, color="Outcome", color_discrete_map=PASS_COLORS, opacity=.6, hover_data=["G1", "failures", "studytime", "absences", "school", "higher"], title="Interactive: second-period grade vs final grade (click legend to filter)")
    f1.write_html(EDA_DIR / "interactive_g2_vs_g3.html", include_plotlyjs="cdn")
    f2 = px.box(d, x="failures", y=TARGET, color="school", points="all", title="Interactive: final grade by past failures and school")
    f2.write_html(EDA_DIR / "interactive_g3_by_failures.html", include_plotlyjs="cdn")


def run():
    set_style(); df, _ = clean_data(load_data())
    for f in (fig01_target, fig02_hist, fig03_box, fig04_outliers, fig05_normal, fig06_discrete, fig07_corr, fig08_group_means, fig09_scatter, fig10_pca, fig11_pair, fig12_heat, interactive):
        f(df); print("done", f.__name__)


if __name__ == "__main__":
    run()
