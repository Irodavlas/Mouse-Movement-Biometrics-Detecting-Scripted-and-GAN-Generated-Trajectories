from pathlib import Path
import numpy as np
import pandas as pd

human_csv = Path("data/processed/balabit_train_steps.csv")
bot_csv = Path("data/processed/bot_train_steps.csv")


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


def _sample_dt(n_steps: int, total_dt: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "linear":
        dt = np.full(n_steps, total_dt / n_steps, dtype=float)
    elif mode == "bezier":
        raw_dt = (total_dt / n_steps) * rng.uniform(0.85, 1.15, size=n_steps)
        dt = raw_dt * (total_dt / raw_dt.sum())
    else:
        # Base cadence + occasional sub-bursts (near 0) and longer pauses
        base = rng.normal(loc=0.11, scale=0.03, size=n_steps).clip(min=0.02)
        if rng.random() < 0.7:
            burst_idx = rng.choice(n_steps, size=max(1, n_steps // 6), replace=False)
            base[burst_idx] = 0.0
        if rng.random() < 0.5:
            pause_idx = rng.integers(0, n_steps)
            base[pause_idx] += rng.uniform(0.25, 0.6)
        dt = base * (total_dt / max(base.sum(), 1e-6))

    return np.maximum(dt, 1e-4)


def generate_stroke(
    n_steps: int,
    total_dx: float,
    total_dy: float,
    total_dt: float,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, n_steps + 1)
    dt = _sample_dt(n_steps, total_dt, mode, rng)

    if mode == "linear":
        x = u * total_dx
        y = u * total_dy
    else:
        perp_x, perp_y = -total_dy, total_dx
        norm = np.hypot(perp_x, perp_y) + 1e-6
        ux, uy = perp_x / norm, perp_y / norm

        if mode == "bezier":
            arc_offset = rng.uniform(-0.25, 0.25) * norm
            cx = 0.5 * total_dx + ux * arc_offset
            cy = 0.5 * total_dy + uy * arc_offset
            s = 3 * u**2 - 2 * u**3
            x = 2 * (1 - s) * s * cx + (s**2) * total_dx
            y = 2 * (1 - s) * s * cy + (s**2) * total_dy
        else:
            # Asymmetric two-control-point cubic Bezier + smoothed jitter
            c1_off = rng.uniform(-0.35, 0.35) * norm
            c2_off = rng.uniform(-0.35, 0.35) * norm
            x1, y1 = total_dx / 3.0 + ux * c1_off, total_dy / 3.0 + uy * c1_off
            x2, y2 = 2.0 * total_dx / 3.0 + ux * c2_off, 2.0 * total_dy / 3.0 + uy * c2_off

            # Ease-in-out curve parameter
            s = u**2 * (3.0 - 2.0 * u)
            omt = 1.0 - s
            x = (
                (omt**3) * 0.0
                + 3 * (omt**2) * s * x1
                + 3 * omt * (s**2) * x2
                + (s**3) * total_dx
            )
            y = (
                (omt**3) * 0.0
                + 3 * (omt**2) * s * y1
                + 3 * omt * (s**2) * y2
                + (s**3) * total_dy
            )

            # Smooth random walk jitter, tapered to 0 at endpoints so net delta stays exact
            taper = np.sin(np.pi * u) ** 0.5
            jitter_scale = max(1.5, 0.02 * np.hypot(total_dx, total_dy) / max(n_steps**0.5, 1.0))
            jx = np.cumsum(rng.normal(0.0, jitter_scale, size=n_steps + 1))
            jy = np.cumsum(rng.normal(0.0, jitter_scale, size=n_steps + 1))
            jx = (jx - jx[0]) * taper
            jy = (jy - jy[0]) * taper
            x = x + jx
            y = y + jy

    # Enforce exact net displacement over the stroke
    x[0], y[0] = 0.0, 0.0
    x[-1], y[-1] = total_dx, total_dy

    dx = np.diff(x)
    dy = np.diff(y)
    return dx, dy, dt


def build_bot_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    human_df = pd.read_csv(human_csv)

    stroke_specs = []
    for traj_id, g in human_df.groupby("traj_id", sort=False):
        stroke_specs.append(
            {
                "n_steps": len(g),
                "total_dx": g["dx"].sum(),
                "total_dy": g["dy"].sum(),
                "total_dt": g["dt"].sum(),
                "start_x": g["x"].iloc[0] - g["dx"].iloc[0],
                "start_y": g["y"].iloc[0] - g["dy"].iloc[0],
            }
        )

    modes = ["linear", "bezier", "bezier_noisy", "burst_pause"]
    rows = []
    for idx, spec in enumerate(stroke_specs, start=1):
        mode = modes[idx % len(modes)]
        dx, dy, dt = generate_stroke(
            n_steps=spec["n_steps"],
            total_dx=spec["total_dx"],
            total_dy=spec["total_dy"],
            total_dt=spec["total_dt"],
            mode=mode,
            rng=rng,
        )
        x = spec["start_x"] + np.cumsum(dx)
        y = spec["start_y"] + np.cumsum(dy)
        t = np.cumsum(dt)

        rows.append(
            pd.DataFrame(
                {
                    "seq_id": idx,
                    "x": x,
                    "y": y,
                    "t": t,
                    "dx": dx,
                    "dy": dy,
                    "dt_raw": dt,
                    "dt": dt,
                    "user_id": f"bot_{mode}",
                    "session_id": "synthetic",
                    "traj_id": f"bot_{mode}_{idx:06d}",
                }
            )
        )

    bot_df = pd.concat(rows, ignore_index=True)
    bot_df = add_kinematic_features(bot_df)
    bot_csv.parent.mkdir(parents=True, exist_ok=True)
    bot_df.to_csv(bot_csv, index=False)
    return bot_df


if __name__ == "__main__":
    build_bot_dataset()