"""Evaluate both MLP settings on the held-out TEST set (seed-42 split): metrics, error analysis,
permutation importance and the repeated-split comparison. Writes Figures 13-20."""
import json, os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from config import MODEL_DIR, EVAL_DIR, SEED, PASS_MARK, BLUE, ORANGE, GREEN, set_style, save_fig
from data_preprocessing import load_data, clean_data, feature_columns
from train_model import prepare, predict, reg_metrics

TITLES = {"A": "Setting A: with G1 and G2", "B": "Setting B: without G1 and G2"}


def run():
    set_style(); df, _ = clean_data(load_data()); S = json.load(open(EVAL_DIR / "training_summary.json"))["settings"]
    res = {"metrics": {}, "passfail": {}}; P = {}; H = {}; imp_all = {}
    for name, use_g in {"A": True, "B": False}.items():
        pre, Xtr, Xval, Xte, ytr, yval, yte = prepare(df, use_g); cols = feature_columns(df, use_g)
        mlp = tf.keras.models.load_model(MODEL_DIR / f"mlp_setting_{name}.keras"); sc = tuple(S[name]["scale"]); H[name] = json.load(open(EVAL_DIR / f"history_{name}.json"))
        p = predict(mlp, Xte, sc); P[name] = (yte, p)
        mm = {"MLP": reg_metrics(yte, p), "MLP (validation)": reg_metrics(yval, predict(mlp, Xval, sc)), "Mean baseline": reg_metrics(yte, np.full_like(yte, ytr.mean(), dtype=float))}
        r = Ridge(alpha=10.0).fit(Xtr, ytr); mm["Ridge regression"] = reg_metrics(yte, np.clip(r.predict(Xte), 0, 20))
        f = RandomForestRegressor(300, random_state=SEED, n_jobs=-1, min_samples_leaf=2).fit(Xtr, ytr); mm["Random forest"] = reg_metrics(yte, np.clip(f.predict(Xte), 0, 20))
        res["metrics"][name] = mm
        res["passfail"][name] = {"accuracy": float(((p >= PASS_MARK - 0.5) == (yte >= PASS_MARK)).mean()), "auc": float(roc_auc_score(yte >= PASS_MARK, p)), "n_fail_test": int((yte < PASS_MARK).sum()), "n_test": int(len(yte))}
        # permutation importance on RAW columns: increase in test RMSE when a feature is shuffled
        rng = np.random.default_rng(SEED); Xraw = pd.DataFrame(df.drop(columns=["G3"]).iloc[0:0]); base = mm["MLP"]["RMSE"]
        from data_preprocessing import split_data
        _, _, X_te_raw, _, _, y_te_raw = split_data(df); imp = {}
        for col in cols:
            d = []
            for _ in range(10):
                Xp = X_te_raw.copy(); Xp[col] = rng.permutation(Xp[col].values)
                d.append(reg_metrics(yte, predict(mlp, pre.transform(Xp[cols]).astype("float32"), sc))["RMSE"] - base)
            imp[col] = (np.mean(d), np.std(d))
        imp = pd.DataFrame(imp, index=["mean_increase", "std"]).T.sort_values("mean_increase", ascending=False); imp.round(4).to_csv(EVAL_DIR / f"table_permutation_importance_{name}.csv"); imp_all[name] = imp
        res["residual_stats_" + name] = {"mean_error": float((p - yte).mean()), "sd_error": float((p - yte).std(ddof=1)), "within_1": float((np.abs(p - yte) <= 1).mean()), "within_2": float((np.abs(p - yte) <= 2).mean()),
                                         "shapiro_p": float(stats.shapiro(p - yte)[1]), "max_abs_error": float(np.abs(p - yte).max()), "worst_actual": float(yte[np.argmax(np.abs(p - yte))]), "worst_pred": float(p[np.argmax(np.abs(p - yte))])}
    res["top_importance"] = {k: v.head(10).round(4).reset_index().rename(columns={"index": "feature"}).to_dict("records") for k, v in imp_all.items()}
    pd.DataFrame({f"{n}|{k}": v for n, d in res["metrics"].items() for k, v in d.items()}).T.round(4).to_csv(EVAL_DIR / "table_test_metrics.csv")
    # ---- Fig 13 / 14: learning curves -------------------------------------------------------------
    for key, ttl, yl, fn in [("loss", "Fig. 13  Training vs validation loss (MSE, standardised target)", "MSE loss", "fig13_loss_vs_epoch.png"), ("mae", "Fig. 14  Training vs validation MAE (standardised target)", "MAE", "fig14_mae_vs_epoch.png")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
        for ax, n in zip(axes, "AB"):
            h = H[n]; ep = np.arange(1, len(h[key]) + 1); be = int(np.argmin(h["val_loss"])) + 1
            ax.plot(ep, h[key], color=BLUE, lw=2, label="Training"); ax.plot(ep, h["val_" + key], color=ORANGE, lw=2, label="Validation"); ax.axvline(be, color="#52514e", ls=":", lw=1.2)
            ax.text(be + .5, ax.get_ylim()[1] * .55, f"best epoch {be}", fontsize=8); ax.set(title=TITLES[n], xlabel="Epoch", ylabel=yl); ax.legend()
            res.setdefault("best_epoch", {})[n] = be; res.setdefault("epochs_run", {})[n] = len(ep)
        fig.suptitle(ttl, fontweight="bold", y=1.03); fig.tight_layout(); save_fig(fig, EVAL_DIR / fn)
    # ---- Fig 15: predicted vs actual -----------------------------------------------------------------
    rng = np.random.default_rng(SEED); fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharex=True, sharey=True)
    for ax, n in zip(axes, "AB"):
        y, p = P[n]; ax.scatter(y + rng.normal(0, .12, len(y)), p, s=22, alpha=.7, color=BLUE, edgecolor="none"); ax.plot([0, 20], [0, 20], color=ORANGE, lw=1.8, label="Perfect prediction")
        m = res["metrics"][n]["MLP"]; ax.set(title=f"{TITLES[n]}\nRMSE {m['RMSE']:.2f}, MAE {m['MAE']:.2f}, R² {m['R2']:.2f}", xlabel="Actual G3", ylabel="Predicted G3" if n == "A" else "", xlim=(-1, 20.5), ylim=(-1, 20.5)); ax.legend(loc="upper left"); ax.set_aspect("equal")
    fig.suptitle("Fig. 15  Predicted vs actual final grade (test set)", fontweight="bold", y=1.02); fig.tight_layout(); save_fig(fig, EVAL_DIR / "fig15_predicted_vs_actual.png")
    # ---- Fig 16: error distribution ---------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharex=True)
    for ax, n in zip(axes, "AB"):
        y, p = P[n]; e = p - y; ax.hist(e, bins=np.arange(-12, 12.5, 1), color=BLUE, edgecolor="white", lw=.5); ax.axvline(0, color="#0b0b0b", lw=1.3); ax.axvline(e.mean(), color=ORANGE, ls="--", lw=1.6, label=f"mean error {e.mean():+.2f}")
        ax.set(title=f"{TITLES[n]}  (SD {e.std(ddof=1):.2f})", xlabel="Prediction error (predicted - actual)", ylabel="Students" if n == "A" else ""); ax.legend()
    fig.suptitle("Fig. 16  Error distribution (test set)", fontweight="bold", y=1.03); fig.tight_layout(); save_fig(fig, EVAL_DIR / "fig16_error_distribution.png")
    # ---- Fig 17: residual vs predicted ----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    for ax, n in zip(axes, "AB"):
        y, p = P[n]; ax.scatter(p, p - y, s=20, alpha=.7, color=BLUE, edgecolor="none"); ax.axhline(0, color=ORANGE, lw=1.6); ax.set(title=TITLES[n], xlabel="Predicted G3", ylabel="Error (predicted - actual)" if n == "A" else "")
    fig.suptitle("Fig. 17  Residual plot (test set)", fontweight="bold", y=1.03); fig.tight_layout(); save_fig(fig, EVAL_DIR / "fig17_residual_plot.png")
    # ---- Fig 18: permutation importance ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, n in zip(axes, "AB"):
        t = imp_all[n].head(10).iloc[::-1]; ax.barh(t.index, t.mean_increase, xerr=t["std"], color=BLUE, height=.65, error_kw={"lw": 1, "ecolor": "#52514e"})
        ax.set(title=TITLES[n], xlabel="Increase in test RMSE when shuffled"); ax.grid(axis="y", visible=False)
    fig.suptitle("Fig. 18  Permutation importance (top 10 features)", fontweight="bold", y=1.02); fig.tight_layout(); save_fig(fig, EVAL_DIR / "fig18_permutation_importance.png")
    # ---- Fig 19: tuning --------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, n in zip(axes, "AB"):
        t = pd.read_csv(EVAL_DIR / f"tuning_results_{n}.csv"); t["cfg"] = t.hidden + " d=" + t.dropout.astype(str) + " lr=" + t.lr.astype(str); t = t.sort_values("val_RMSE", ascending=False)
        ax.barh(t.cfg, t.val_RMSE, color=[ORANGE if v == t.val_RMSE.min() else BLUE for v in t.val_RMSE], height=.65)
        for y, v in enumerate(t.val_RMSE): ax.text(v + .01, y, f"{v:.2f}", va="center", fontsize=7)
        ax.set(title=TITLES[n], xlabel="Validation RMSE (lower is better)", xlim=(0, t.val_RMSE.max() * 1.12)); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Fig. 19  Hyperparameter search results", fontweight="bold", y=1.02); fig.tight_layout(); save_fig(fig, EVAL_DIR / "fig19_tuning_results.png")
    # ---- Fig 20: repeated random splits ----------------------------------------------------------------------------------
    rp = pd.read_csv(EVAL_DIR / "repeated_splits.csv"); order = ["Mean baseline", "Ridge regression", "Random forest", "MLP"]; cols = {"Mean baseline": "#9b9a93", "Ridge regression": GREEN, "Random forest": ORANGE, "MLP": BLUE}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, n in zip(axes, "AB"):
        d = [rp[(rp.setting == n) & (rp.model == m)].RMSE.values for m in order]; bp = ax.boxplot(d, tick_labels=order, patch_artist=True, widths=.55, medianprops=dict(color="#0b0b0b"))
        for b, m in zip(bp["boxes"], order): b.set(facecolor=cols[m], alpha=.85)
        ax.set(title=TITLES[n], ylabel="Test RMSE over 10 random splits" if n == "A" else ""); ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Fig. 20  Stability over 10 random train/validation/test splits", fontweight="bold", y=1.02); fig.tight_layout(); save_fig(fig, EVAL_DIR / "fig20_repeated_splits.png")
    sm = rp.groupby(["setting", "model"])[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).round(4); sm.to_csv(EVAL_DIR / "table_repeated_splits_summary.csv"); res["repeated"] = {f"{a}|{b}": {m: {"mean": float(sm.loc[(a, b), (m, "mean")]), "std": float(sm.loc[(a, b), (m, "std")])} for m in ["MAE", "RMSE", "R2"]} for a, b in sm.index}
    json.dump(res, open(EVAL_DIR / "results_model.json", "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ["metrics", "passfail", "residual_stats_A", "residual_stats_B", "best_epoch", "epochs_run"]}, indent=1, default=float)); print(sm); print(imp_all["A"].head(8)); print(imp_all["B"].head(8))


if __name__ == "__main__":
    run()
