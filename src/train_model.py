"""Train the regression MLP for two experimental settings and run the baselines.
  Setting A: all 32 features (includes the earlier grades G1 and G2)
  Setting B: 30 features, NO earlier grades (early-warning setting)
Hyperparameters are chosen by grid search on the VALIDATION set only (split seed 42).
Then a repeated-split experiment (10 random splits) measures how stable the results are."""
import itertools, json, os, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import MODEL_DIR, EVAL_DIR, SEED, PASS_MARK
from data_preprocessing import load_data, clean_data, split_data, build_preprocessor, feature_columns

BATCH_SIZE, MAX_EPOCHS, PATIENCE = 32, 300, 25
GRID = {"hidden": [(32, 16), (64, 32), (64, 32, 16)], "dropout": [0.1, 0.3], "lr": [1e-3, 3e-3]}
SETTINGS = {"A": True, "B": False}                      # name -> use_grades


def build_mlp(n_in, hidden, dropout, lr):
    """MLP regressor: [Dense(ReLU, L2) -> BatchNorm -> Dropout] x L -> Dense(1, linear)."""
    layers = [tf.keras.layers.Input((n_in,))]
    for h in hidden:
        layers += [tf.keras.layers.Dense(h, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                   tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(dropout)]
    layers.append(tf.keras.layers.Dense(1))
    m = tf.keras.Sequential(layers)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return m


def fit_mlp(Xtr, ytr, Xval, yval, hidden, dropout, lr, seed=SEED):
    """Train with early stopping on validation loss. The target is standardised (train mean/std) for stable training."""
    tf.keras.utils.set_random_seed(seed)
    mu, sd = ytr.mean(), ytr.std()
    m = build_mlp(Xtr.shape[1], hidden, dropout, lr)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    h = m.fit(Xtr, (ytr - mu) / sd, validation_data=(Xval, (yval - mu) / sd), epochs=MAX_EPOCHS,
              batch_size=BATCH_SIZE, callbacks=[es], verbose=0)
    return m, h.history, (mu, sd)


def predict(m, X, scale):
    """Back-transform to grade points and clip to the valid 0-20 range."""
    return np.clip(m.predict(X, verbose=0).ravel() * scale[1] + scale[0], 0, 20)


def reg_metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": r2_score(y, p)}


def prepare(df, use_grades, seed=SEED):
    cols = feature_columns(df, use_grades); Xtr, Xval, Xte, ytr, yval, yte = split_data(df, seed)
    pre = build_preprocessor(cols).fit(Xtr[cols])                    # fitted on TRAIN only (no leakage)
    f = lambda X: pre.transform(X[cols]).astype("float32")
    return pre, f(Xtr), f(Xval), f(Xte), ytr.values, yval.values, yte.values


def run():
    df, _ = clean_data(load_data()); summary = {"batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "settings": {}}
    for name, use_g in SETTINGS.items():
        pre, Xtr, Xval, Xte, ytr, yval, yte = prepare(df, use_g); joblib.dump(pre, MODEL_DIR / f"preprocessor_{name}.joblib")
        rows, best = [], None
        for hidden, dr, lr in itertools.product(GRID["hidden"], GRID["dropout"], GRID["lr"]):
            t0 = time.time(); m, h, sc = fit_mlp(Xtr, ytr, Xval, yval, hidden, dr, lr)
            vm = reg_metrics(yval, predict(m, Xval, sc))
            rec = {"hidden": str(hidden), "dropout": dr, "lr": lr, "epochs_run": len(h["loss"]), "val_RMSE": vm["RMSE"], "val_MAE": vm["MAE"], "val_R2": vm["R2"], "seconds": round(time.time() - t0, 1)}
            rows.append(rec); print(name, rec)
            if best is None or vm["RMSE"] < best["rmse"]:
                best = {"rmse": vm["RMSE"], "model": m, "hist": h, "scale": sc, "cfg": {"hidden": list(hidden), "dropout": dr, "lr": lr}}
        pd.DataFrame(rows).to_csv(EVAL_DIR / f"tuning_results_{name}.csv", index=False)
        best["model"].save(MODEL_DIR / f"mlp_setting_{name}.keras")
        json.dump({k: [float(x) for x in v] for k, v in best["hist"].items()}, open(EVAL_DIR / f"history_{name}.json", "w"))
        summary["settings"][name] = {"use_grades": use_g, "n_inputs": int(Xtr.shape[1]), "best_config": best["cfg"], "scale": [float(best["scale"][0]), float(best["scale"][1])],
                                     "epochs_run": len(best["hist"]["loss"]), "val_RMSE": best["rmse"], "n_params": int(best["model"].count_params()),
                                     "sizes": {"train": len(ytr), "val": len(yval), "test": len(yte)}}
    # ---- repeated random splits: stability of every model (config fixed from the seed-42 search) -------
    reps = []
    for seed in range(10):
        for name, use_g in SETTINGS.items():
            pre, Xtr, Xval, Xte, ytr, yval, yte = prepare(df, use_g, seed); c = summary["settings"][name]["best_config"]
            m, h, sc = fit_mlp(Xtr, ytr, Xval, yval, tuple(c["hidden"]), c["dropout"], c["lr"], seed=seed)
            models = {"MLP": predict(m, Xte, sc)}
            for nm, mdl in {"Mean baseline": DummyRegressor(), "Ridge regression": Ridge(alpha=10.0), "Random forest": RandomForestRegressor(300, random_state=seed, n_jobs=-1, min_samples_leaf=2)}.items():
                mdl.fit(Xtr, ytr); models[nm] = np.clip(mdl.predict(Xte), 0, 20)
            for nm, p in models.items():
                reps.append({"seed": seed, "setting": name, "model": nm, **reg_metrics(yte, p)})
            print("repeat", seed, name, round(reps[-4]["RMSE"], 3))
    pd.DataFrame(reps).to_csv(EVAL_DIR / "repeated_splits.csv", index=False)
    # ---- baselines on the main split (seed 42) saved for the evaluation script -------------------------
    for name, use_g in SETTINGS.items():
        pre, Xtr, Xval, Xte, ytr, yval, yte = prepare(df, use_g)
        for nm, mdl in {"ridge": Ridge(alpha=10.0), "forest": RandomForestRegressor(300, random_state=SEED, n_jobs=-1, min_samples_leaf=2)}.items():
            mdl.fit(Xtr, ytr)
            if nm == "ridge": joblib.dump(mdl, MODEL_DIR / f"baseline_ridge_{name}.joblib")
    json.dump(summary, open(EVAL_DIR / "training_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=1)[:1500])


if __name__ == "__main__":
    run()
