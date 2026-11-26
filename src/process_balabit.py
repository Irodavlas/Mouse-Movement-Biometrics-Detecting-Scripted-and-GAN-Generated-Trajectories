from pathlib import Path
import numpy as np
import pandas as pd

root_path = Path("data/raw/balabit/training_files")
out_csv = Path("data/processed/balabit_train_steps.csv")

MIN_SEQUENCE = 10
# for now we focus on training files but also test files are good for short trajectories, we would need to add them later on

def add_kinematic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby("traj_id", sort=False)

    step_dist = np.hypot(df["dx"], df["dy"])
    df["step_speed"] = step_dist / df["dt"]
    df["speed"] = df["step_speed"]

    df["direction"] = np.arctan2(df["dy"], df["dx"])
    raw_angle_diff = grp["direction"].diff().fillna(0.0)
    df["angle_change"] = np.arctan2(np.sin(raw_angle_diff), np.cos(raw_angle_diff))

    dv = grp["speed"].diff().fillna(0.0)
    df["acceleration"] = dv / df["dt"]

    da = grp["acceleration"].diff().fillna(0.0)
    df["jerk"] = da / df["dt"]

    return df

def parse_training_directory():
    session_files = sorted([p for p in root_path.rglob("session*")])
    all_steps = []

    for file_path in session_files:
        user_id = file_path.parent.name
        session_id = file_path.stem.split("_")[1]
        print(user_id, session_id)

        df = pd.read_csv(file_path)
        df_steps = process_csv(df)

        if not df_steps.empty:
            df_steps["user_id"] = user_id
            df_steps["session_id"] = session_id
            df_steps["traj_id"] = (
                user_id
                + "_"
                + session_id
                + "_"
                + df_steps["seq_id"].astype(str).str.zfill(5)
            )
            all_steps.append(df_steps)

        

    if all_steps:
        dataset = pd.concat(all_steps, ignore_index=True)
        dataset = add_kinematic_features(dataset)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(out_csv, index=False)
        return dataset

    return pd.DataFrame()


def process_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
        We need to construct the new dataset by dropping:
- non movement or noisy rows , we keep the state MOVE and segment strokes between pressed and released.
- drop identical rows 
- if time delta is 0 instead we clamp to 0.0001
- we can keep the dx and dy rows if we use dt to discrinate, as pauses are human'like 
         
        At each Move we create another sequence, if its too short we drop it (few points or tiny start and end distances) - post processing maybe 

    """
    df = df.drop_duplicates(subset=["client timestamp", "state", "x", "y"]).copy()
    df = df[["client timestamp", "state", "x", "y"]]
    # segmenting sequences
    is_move = df["state"] == "Move"
    # we grab the prev move by shifting the whole sequence of 1 step so we can check current and prev
    prev_move = is_move.shift(1, fill_value=False)
    dt_gap = df["client timestamp"].diff() > 0.5

    start_block = is_move & (~prev_move | dt_gap)
    # crazy: cumsum sums the values, in this case we only have 1 True at start, so each seq gets and index
    blocks_id = start_block.cumsum()

    moves = (
        df.loc[is_move, ["x", "y", "client timestamp"]]
        .rename(columns={"client timestamp": "t"})
        .copy()
    )

    moves["seq_idx"] = blocks_id[is_move]

    # has shape (N,3)
    seqs_arr = [
        group[["x", "y", "t"]].to_numpy()
        for _, group in moves.groupby("seq_idx", sort=False)
    ]
    print(f"[INFO] sequence lenght before filtering: {len(seqs_arr)}")
    seq = [seq for seq in seqs_arr if len(seq) > MIN_SEQUENCE]
    print(f"[INFO] sequence lenght of valid sequences: {len(seq)}")

    rows = []
    for seq_id, s in enumerate(seq, start=1):
        dx = np.diff(s[:, 0])
        dy = np.diff(s[:, 1])
        dt_raw = np.diff(s[:, 2])
        dt = np.maximum(dt_raw, 1e-4)

        rows.append(
            pd.DataFrame(
                {
                    "seq_id": seq_id,
                    "x": s[1:, 0],
                    "y": s[1:, 1],
                    "t": s[1:, 2],
                    "dx": dx,
                    "dy": dy,
                    "dt_raw": dt_raw,
                    "dt": dt,
                }
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=["seq_id", "x", "y", "t", "dx", "dy", "dt_raw", "dt"]
        )

    df_steps = pd.concat(rows, ignore_index=True)
    return df_steps


print(parse_training_directory())