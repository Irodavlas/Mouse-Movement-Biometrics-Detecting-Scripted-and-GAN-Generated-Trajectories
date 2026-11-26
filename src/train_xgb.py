from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from xgboost import XGBClassifier

human_csv = Path("data/processed/balabit_train_steps.csv")
bot_csv = Path("data/processed/bot_train_steps.csv")


def aggregate_trajectories(df: pd.DataFrame, label: int, desc: str = "") -> pd.DataFrame:
    agg_rows = []
    for traj_id, g in tqdm(df.groupby("traj_id", sort=False), desc=desc, unit="traj"):
        net_disp = np.hypot(g["dx"].sum(), g["dy"].sum())
        path_len = np.hypot(g["dx"], g["dy"]).sum()
        straightness = net_disp / (path_len + 1e-6)

        agg_rows.append(
            {
                "traj_id": traj_id,
                "user_id": g["user_id"].iloc[0],
                "label": label,
                "n_steps": len(g),
                "duration": g["dt"].sum(),
                "net_disp": net_disp,
                "path_len": path_len,
                "straightness": straightness,
                "dt_mean": g["dt"].mean(),
                "dt_std": g["dt"].std(ddof=0),
                "speed_mean": g["speed"].mean(),
                "speed_std": g["speed"].std(ddof=0),
                "speed_max": g["speed"].max(),
                "acc_std": g["acceleration"].std(ddof=0),
                "jerk_std": g["jerk"].std(ddof=0),
                "angle_change_mean": g["angle_change"].abs().mean(),
                "angle_change_std": g["angle_change"].std(ddof=0),
            }
        )
    return pd.DataFrame(agg_rows)


def main() -> None:
    print("Loading CSVs...")
    human_raw = pd.read_csv(human_csv)
    bot_raw = pd.read_csv(bot_csv)

    human_df = aggregate_trajectories(human_raw, label=1, desc="Aggregating human")
    bot_df = aggregate_trajectories(bot_raw, label=0, desc="Aggregating bot")
    full_df = pd.concat([human_df, bot_df], ignore_index=True)

    feature_cols = [
        c for c in full_df.columns if c not in {"traj_id", "user_id", "label"}
    ]
    X = full_df[feature_cols]
    y = full_df["label"]

    human_mask = full_df["label"] == 1
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    h_train_idx, h_val_idx = next(
        gss.split(X[human_mask], y[human_mask], groups=full_df.loc[human_mask, "user_id"])
    )

    bot_indices = full_df.index[~human_mask].to_numpy(copy=True)
    rng = np.random.default_rng(42)
    rng.shuffle(bot_indices)
    split_pt = int(0.8 * len(bot_indices))

    train_idx = np.concatenate([full_df.index[human_mask][h_train_idx], bot_indices[:split_pt]])
    val_idx = np.concatenate([full_df.index[human_mask][h_val_idx], bot_indices[split_pt:]])

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    clf.fit(X.iloc[train_idx], y.iloc[train_idx])

    preds = clf.predict(X.iloc[val_idx])
    probs = clf.predict_proba(X.iloc[val_idx])[:, 1]

    print(f"\nValidation ROC-AUC: {roc_auc_score(y.iloc[val_idx], probs):.4f}")
    print(classification_report(y.iloc[val_idx], preds, target_names=["Bot", "Human"]))

    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop Feature Importances:")
    print(importances)


if __name__ == "__main__":
    main()