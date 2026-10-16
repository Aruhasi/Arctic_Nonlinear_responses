#!/usr/bin/env python3
"""
Figure 3 redesign: joint SAT-GSAT density + matched-GSAT conditional PDFs + SIA-GSAT.

Outputs one 3-row figure per scenario (SSP1-1.9 and SSP1-2.6):
  Row 1: Arctic SAT vs GSAT for CO2-increasing and CO2-decreasing phases,
         with pooled OLS fits, member-cluster-bootstrap R2 uncertainty,
         80% highest-density-region (HDR) contours, and the matched-GSAT band.
  Row 2: P(Arctic SAT | matched GSAT band), one mini-panel per model,
         comparing CO2-increasing vs CO2-decreasing phases and annotating
         Delta T_matched with a member-cluster-bootstrap 95% CI.
  Row 3: September Arctic SIA vs GSAT for the two phases, with pooled OLS fits
         and member-cluster-bootstrap R2 uncertainty.

The script first diagnoses GSAT overlap and chooses matched-GSAT windows.
By default it matches within each model and scenario, because the four models
can occupy substantially different absolute GSAT ranges. Optional global and
per-scenario modes are retained for sensitivity tests when a common overlap exists.

This is designed around the same data paths/member definitions used in your
current Fig3_member_cluster_bootstrap_significance.py workflow.
"""
# %%
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr
# ============================================================================
# CONFIG
# ============================================================================
ROOT = Path("/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic")
SAT_DIR = ROOT / "docs/data/Regional_diagnosis/regional_mean"
SIA_DIR = ROOT / "docs/data/Regional_diagnosis/SIA"
GSAT_DIR = ROOT / "scripts/Timeseries/deseasonalized_output"
FIG_DIR = ROOT / "docs/figs/Diagnosis_plots"

SCENARIOS = ("ssp119", "ssp126")
SCENARIO_LABELS = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
}

MODELS = ("MIROC6", "MIROC-ES2L", "MPI-ESM1-2-LR", "CanESM5")
MODEL_DISPLAY_NAMES = {
    "MIROC6": "MIROC6",
    "MIROC-ES2L": "MIROC-ES2L",
    "MPI-ESM1-2-LR": "MPI-ESM1.2-LR",
    "CanESM5": "CanESM5",
}
SOURCE_MODEL = {
    "MIROC6": "MIROC",
    "MIROC-ES2L": "MIROC-ES2L",
    "MPI-ESM1-2-LR": "MPI-ESM-LR",
    "CanESM5": "CanESM5",
}
MEMBERS = {
    "MIROC6": [f"r{i}i1p1f1" for i in range(1, 51)],
    "MIROC-ES2L": [f"r{i}i1p1f2" for i in range(1, 11)],
    "MPI-ESM1-2-LR": [f"r{i}i1p1f1" for i in range(1, 51)],
    "CanESM5": [f"r{i}i1p1f1" for i in range(1, 26)]
              + [f"r{i}i1p2f1" for i in range(1, 26)],
}

PERIODS = {
    "mid-term": (2015, 2049),
    "long-term": (2065, 2099),
}
PERIOD_LABELS = {
    "mid-term": r"CO$_2$ increasing (2015-2049)",
    "long-term": r"CO$_2$ decreasing (2065-2099)",
}
PERIOD_SHORT = {
    "mid-term": r"CO$_2$↑",
    "long-term": r"CO$_2$↓",
}
PERIOD_LINESTYLES = {
    "mid-term": "-",
    "long-term": "--",
}

# Keep your existing model/phase colors.
PERIOD_MODEL_COLORS = {
    "mid-term": {
        "MIROC6": np.array([150, 184, 155]) / 255.0,
        "MIROC-ES2L": np.array([78, 146, 128]) / 255.0,
        "MPI-ESM1-2-LR": np.array([49, 100, 108]) / 255.0,
        "CanESM5": np.array([32, 54, 79]) / 255.0,
    },
    "long-term": {
        "MIROC6": np.array([212, 156, 135]) / 255.0,
        "MIROC-ES2L": np.array([184, 98, 101]) / 255.0,
        "MPI-ESM1-2-LR": np.array([139, 52, 94]) / 255.0,
        "CanESM5": np.array([80, 24, 78]) / 255.0,
    },
}
MODEL_MARKERS = {
    "MIROC6": "o",
    "MIROC-ES2L": "s",
    "MPI-ESM1-2-LR": "^",
    "CanESM5": "D",
}

# Joint-density / matched-GSAT settings.
HDR_MASS = 0.80
KDE_BW_METHOD = "scott"     # scipy gaussian_kde bw_method
KDE_GRID_N = 150

PREFERRED_MATCH_CENTER = 1.50  # K above 1850-1900
MATCH_HALF_WIDTH = 0.10        # +/- 0.10 K => 0.20-K window
MATCH_SCAN_STEP = 0.01
MIN_POINTS_PER_CELL = 20
MIN_MEMBERS_PER_CELL = 5       # MIROC-ES2L only has 10 members

# "global" => same band for all models and both scenarios (usually too strict).
# "per_scenario" => one band per scenario, common to all four models.
# "per_model" => RECOMMENDED: one band for each model x scenario, matching CO2-up
#                and CO2-down within that model. This avoids forcing models with
#                different GSAT ranges/climate sensitivities onto one absolute level.
MATCH_SCOPE = "per_model"

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 42
MIN_VALID_YEARS_PER_MEMBER = 3

SCATTER_SIZE = 25
SCATTER_ALPHA_MT = 0.22
SCATTER_ALPHA_LT = 0.20

# Optional manual x-axis limits for matched-GSAT PDFs (None => auto).
PDF_XLIMITS = {
    "MIROC6": (2.0, 8.0),
    "MIROC-ES2L": (0.0, 8.0),
    "MPI-ESM1-2-LR": (2.0, 6.0),
    "CanESM5": (4.0, 12.0),
}

# Figure typography and tick sizes (points).
LEGEND_FONTSIZE = 12
PDF_LEGEND_FONTSIZE = 11
TICK_LABELSIZE = 14
TICK_LENGTH = 7
TICK_WIDTH = 1.5
PANEL_LETTER_FONTSIZE = 22

PDF_ANNOT_BBOX = dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.6)
# ============================================================================
# DATA HELPERS
# ============================================================================
def norm_member(value):
    if pd.isna(value):
        return pd.NA
    text = str(value)
    r_match = re.search(r"r(\d+)", text)
    p_match = re.search(r"p(\d+)", text)
    if r_match and p_match:
        return int(r_match.group(1)) * 1000 + int(p_match.group(1))
    if r_match:
        return int(r_match.group(1))
    try:
        return int(float(text))
    except Exception:
        return pd.NA

def extract_year(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and 1500 <= float(value) <= 2500:
        return int(value)
    if hasattr(value, "year"):
        try:
            return int(value.year)
        except Exception:
            pass
    parsed = pd.to_datetime(value, errors="coerce")
    if not pd.isna(parsed):
        return int(parsed.year)
    return pd.NA

def period_for_year(year):
    if pd.isna(year):
        return pd.NA
    year = int(year)
    for period, (start, end) in PERIODS.items():
        if start <= year <= end:
            return period
    return pd.NA

def finalize_frame(frame, value_name):
    frame = frame.copy()
    frame = frame.dropna(subset=[value_name])
    frame["member"] = frame["member"].astype(str)
    frame["member_id"] = frame["member"].map(norm_member).astype("Int64")
    frame["period"] = frame["year"].map(period_for_year)
    frame = frame.dropna(subset=["member_id", "period", value_name])
    frame = frame.rename(columns={value_name: "value"})
    return frame[
        ["model", "scenario", "member", "member_id", "year", "period", "value"]
    ].reset_index(drop=True)

def load_sat():
    frames = []
    variable = "ArcticOcean66.5N"
    for model in MODELS:
        source_model = SOURCE_MODEL[model]
        path = SAT_DIR / f"{source_model}_ArcticOcean66.5N_RegionalMean_1850_2100.nc"
        with xr.open_dataset(path) as ds:
            da = ds[variable].sel(
                scenario=list(SCENARIOS), year=slice(2015, 2099)
            ).load()

        frame = da.to_dataframe(name="SAT").reset_index()
        frame["model"] = model
        frame["scenario"] = frame["scenario"].astype(str)
        frame["year"] = frame["year"].map(extract_year).astype("Int64")
        frame = frame.dropna(subset=["year"])
        frame["year"] = frame["year"].astype(int)
        frames.append(finalize_frame(frame, "SAT"))

    return pd.concat(frames, ignore_index=True)

def load_sia():
    frames = []
    for model in MODELS:
        for scenario in SCENARIOS:
            path = SIA_DIR / f"{model}_{scenario}_ArcticOcean66p5N_SIA_09_ano_wrt_1850-1900.nc"
            with xr.open_dataset(path) as ds:
                if "SIA_anom" not in ds:
                    raise KeyError(f"SIA_anom missing in {path}")
                da = ds["SIA_anom"].sel(
                    time=slice("2015-01-01", "2099-12-31")
                ).load()

            if "member" not in da.dims:
                da = da.expand_dims(member=["r1i1p1f1"])

            # Preserve the behavior of your current script: remove members
            # containing missing values over the selected period.
            invalid_members = da.isnull().any("time")
            valid_members = da["member"].where(~invalid_members, drop=True)
            da = da.sel(member=valid_members)

            frame = da.to_dataframe(name="SIA").reset_index()
            frame["model"] = model
            frame["scenario"] = scenario
            frame["year"] = frame["time"].map(extract_year).astype("Int64")
            frame = frame.dropna(subset=["year"])
            frame["year"] = frame["year"].astype(int)
            frames.append(finalize_frame(frame, "SIA"))

    return pd.concat(frames, ignore_index=True)


def load_gsat():
    frames = []
    for model in MODELS:
        source_model = SOURCE_MODEL[model]
        for scenario in SCENARIOS:
            for member in MEMBERS[model]:
                path = (
                    GSAT_DIR
                    / scenario
                    / source_model
                    / f"{source_model}_{scenario}_{member}_1850-2100_GMSAT_anomalies_wrt_1850-1900.nc"
                )
                if not path.exists():
                    continue

                with xr.open_dataset(path) as ds:
                    var_name = "tas" if "tas" in ds.data_vars else "tas_ano"
                    da = ds[var_name].load()

                year_coord = "year" if "year" in da.coords else "time"
                values = np.asarray(da.values, dtype=float).reshape(-1)
                years = [extract_year(value) for value in da[year_coord].values]

                frame = pd.DataFrame({"year": years, "GSAT": values})
                frame = frame.dropna(subset=["year", "GSAT"])
                frame["year"] = frame["year"].astype(int)
                frame["model"] = model
                frame["scenario"] = scenario
                frame["member"] = member
                frames.append(finalize_frame(frame, "GSAT"))

    if not frames:
        raise RuntimeError("No GSAT data loaded")
    return pd.concat(frames, ignore_index=True)

def merge_xy(gsat, response, model, scenario, period):
    x = gsat[
        (gsat["model"] == model)
        & (gsat["scenario"] == scenario)
        & (gsat["period"] == period)
    ]
    y = response[
        (response["model"] == model)
        & (response["scenario"] == scenario)
        & (response["period"] == period)
    ]

    merged = pd.merge(
        x,
        y,
        on=["model", "scenario", "member_id", "year", "period"],
        how="inner",
        suffixes=("_x", "_y"),
    )
    return merged
# %%
# ============================================================================
# REGRESSION + MEMBER-CLUSTER BOOTSTRAP
# ============================================================================
def member_cluster_statistics(merged, min_years=MIN_VALID_YEARS_PER_MEMBER):
    rows = []
    for _, group in merged.groupby("member_id", dropna=True):
        x = group["value_x"].to_numpy(dtype=float)
        y = group["value_y"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size < min_years or np.allclose(x, x[0]):
            continue

        rows.append(
            (
                float(x.size),
                float(np.sum(x)),
                float(np.sum(y)),
                float(np.sum(x * x)),
                float(np.sum(y * y)),
                float(np.sum(x * y)),
            )
        )

    if not rows:
        return np.empty((0, 6), dtype=float)
    return np.asarray(rows, dtype=float)

def regression_from_sufficient(stats_array):
    s = np.asarray(stats_array, dtype=float)
    scalar_input = s.ndim == 1
    if scalar_input:
        s = s[None, :]

    n = s[:, 0]
    sx = s[:, 1]
    sy = s[:, 2]
    sxx = s[:, 3]
    syy = s[:, 4]
    sxy = s[:, 5]

    x_var = n * sxx - sx * sx
    y_var = n * syy - sy * sy
    xy_cov = n * sxy - sx * sy

    slope = np.full_like(n, np.nan, dtype=float)
    intercept = np.full_like(n, np.nan, dtype=float)
    r2 = np.full_like(n, np.nan, dtype=float)

    valid = (n >= 3) & np.isfinite(x_var) & (x_var > 0)
    slope[valid] = xy_cov[valid] / x_var[valid]
    intercept[valid] = (sy[valid] - slope[valid] * sx[valid]) / n[valid]

    valid_r2 = valid & np.isfinite(y_var) & (y_var > 0)
    r2[valid_r2] = (xy_cov[valid_r2] ** 2) / (x_var[valid_r2] * y_var[valid_r2])
    r2 = np.clip(r2, 0.0, 1.0)

    if scalar_input:
        return float(slope[0]), float(intercept[0]), float(r2[0]), int(n[0])
    return slope, intercept, r2, n.astype(int)

def stable_seed(*parts, base_seed=BOOTSTRAP_SEED):
    text = "|".join(str(p) for p in parts)
    value = base_seed
    for char in text:
        value = (value * 131 + ord(char)) % (2**32 - 1)
    return int(value)

def member_cluster_bootstrap_regression(merged, nboot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    stats_by_member = member_cluster_statistics(merged)
    n_members = stats_by_member.shape[0]

    if n_members < 3:
        return {
            "n_members": n_members,
            "n_points": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "slope_ci_low": np.nan,
            "slope_ci_high": np.nan,
            "r2_ci_low": np.nan,
            "r2_ci_high": np.nan,
        }

    observed = stats_by_member.sum(axis=0)
    slope, intercept, r2, n_points = regression_from_sufficient(observed)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_members, size=(nboot, n_members))
    boot_stats = stats_by_member[idx].sum(axis=1)
    boot_slope, _, boot_r2, _ = regression_from_sufficient(boot_stats)

    boot_slope = boot_slope[np.isfinite(boot_slope)]
    boot_r2 = boot_r2[np.isfinite(boot_r2)]

    slope_ci = np.nanpercentile(boot_slope, [2.5, 97.5]) if boot_slope.size else [np.nan, np.nan]
    r2_ci = np.nanpercentile(boot_r2, [2.5, 97.5]) if boot_r2.size else [np.nan, np.nan]

    return {
        "n_members": int(n_members),
        "n_points": int(n_points),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "slope_ci_low": float(slope_ci[0]),
        "slope_ci_high": float(slope_ci[1]),
        "r2_ci_low": float(r2_ci[0]),
        "r2_ci_high": float(r2_ci[1]),
    }
# %%
# ============================================================================
# KDE + HIGHEST DENSITY REGION
# ============================================================================
def _finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]

def hdr_density_threshold(z, mass=HDR_MASS):
    """Density threshold whose super-level set contains `mass` of grid density."""
    flat = np.asarray(z, dtype=float).ravel()
    flat = flat[np.isfinite(flat) & (flat >= 0)]
    if flat.size == 0 or np.sum(flat) <= 0:
        return np.nan

    ordered = np.sort(flat)[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    idx = np.searchsorted(cumulative, mass, side="left")
    idx = min(idx, ordered.size - 1)
    return float(ordered[idx])

def plot_hdr_contour(ax, x, y, color, mass=HDR_MASS, bw_method=KDE_BW_METHOD):
    x, y = _finite_xy(x, y)
    if x.size < 10 or np.unique(x).size < 3 or np.unique(y).size < 3:
        return False

    try:
        kde = stats.gaussian_kde(np.vstack([x, y]), bw_method=bw_method)
    except (np.linalg.LinAlgError, ValueError):
        return False

    xspan = np.ptp(x)
    yspan = np.ptp(y)
    xpad = 0.08 * (xspan if xspan > 0 else 1.0)
    ypad = 0.08 * (yspan if yspan > 0 else 1.0)

    gx = np.linspace(x.min() - xpad, x.max() + xpad, KDE_GRID_N)
    gy = np.linspace(y.min() - ypad, y.max() + ypad, KDE_GRID_N)
    xx, yy = np.meshgrid(gx, gy)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    threshold = hdr_density_threshold(zz, mass=mass)
    if not np.isfinite(threshold):
        return False

    ax.contour(
        xx,
        yy,
        zz,
        levels=[threshold],
        colors=[color],
        linewidths=2.0,
        alpha=0.95,
        zorder=5,
    )
    return True

def safe_1d_kde(values, grid, bw_method=KDE_BW_METHOD):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 5 or np.unique(values).size < 3:
        return None
    try:
        kde = stats.gaussian_kde(values, bw_method=bw_method)
        return kde(grid)
    except (np.linalg.LinAlgError, ValueError):
        return None
# ============================================================================
# MATCHED-GSAT WINDOW DIAGNOSTICS
# ============================================================================
def build_sat_merged_cache(gsat, sat):
    cache = {}
    for scenario in SCENARIOS:
        for model in MODELS:
            for period in PERIODS:
                cache[(scenario, model, period)] = merge_xy(
                    gsat, sat, model, scenario, period
                )
    return cache

def overlap_bounds(cache, scenarios):
    """Strict min/max GSAT overlap across every model/phase in `scenarios`."""
    rows = []
    lowers = []
    uppers = []

    for scenario in scenarios:
        for model in MODELS:
            for period in PERIODS:
                merged = cache[(scenario, model, period)]
                x = merged["value_x"].to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                if x.size == 0:
                    raise ValueError(f"No SAT-GSAT data for {scenario}, {model}, {period}")

                lo = float(np.min(x))
                hi = float(np.max(x))
                q025, q975 = np.nanpercentile(x, [2.5, 97.5])
                rows.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "period": period,
                        "gsat_min": lo,
                        "gsat_max": hi,
                        "gsat_q025": float(q025),
                        "gsat_q975": float(q975),
                        "n_points_total": int(x.size),
                        "n_members_total": int(merged["member_id"].nunique()),
                    }
                )
                lowers.append(lo)
                uppers.append(hi)

    common_low = max(lowers)
    common_high = min(uppers)
    return common_low, common_high, rows

def window_cell_counts(cache, scenarios, center, half_width=MATCH_HALF_WIDTH):
    low = center - half_width
    high = center + half_width
    rows = []

    for scenario in scenarios:
        for model in MODELS:
            for period in PERIODS:
                merged = cache[(scenario, model, period)]
                selected = merged[
                    merged["value_x"].between(low, high, inclusive="both")
                ]
                rows.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "period": period,
                        "match_center": float(center),
                        "match_low": float(low),
                        "match_high": float(high),
                        "n_points_matched": int(len(selected)),
                        "n_members_matched": int(selected["member_id"].nunique()),
                    }
                )
    return rows

def window_is_adequate(count_rows):
    if not count_rows:
        return False
    return all(
        row["n_points_matched"] >= MIN_POINTS_PER_CELL
        and row["n_members_matched"] >= MIN_MEMBERS_PER_CELL
        for row in count_rows
    )

def choose_match_window(cache, scenarios, preferred=PREFERRED_MATCH_CENTER):
    common_low, common_high, support_rows = overlap_bounds(cache, scenarios)

    if common_high - common_low < 2 * MATCH_HALF_WIDTH:
        raise ValueError(
            f"No common GSAT interval wide enough for +/-{MATCH_HALF_WIDTH:.2f} K: "
            f"overlap is [{common_low:.3f}, {common_high:.3f}] K"
        )

    # First try the reviewer-motivated/preferred center if it lies inside support.
    if preferred is not None:
        preferred = float(preferred)
        if (
            preferred - MATCH_HALF_WIDTH >= common_low
            and preferred + MATCH_HALF_WIDTH <= common_high
        ):
            counts = window_cell_counts(cache, scenarios, preferred)
            if window_is_adequate(counts):
                return {
                    "center": preferred,
                    "low": preferred - MATCH_HALF_WIDTH,
                    "high": preferred + MATCH_HALF_WIDTH,
                    "source": "preferred",
                    "common_low": common_low,
                    "common_high": common_high,
                    "support_rows": support_rows,
                    "count_rows": counts,
                }

    # Otherwise scan the common support and maximize the weakest model/phase cell.
    start = common_low + MATCH_HALF_WIDTH
    stop = common_high - MATCH_HALF_WIDTH
    centers = np.arange(start, stop + 0.5 * MATCH_SCAN_STEP, MATCH_SCAN_STEP)

    best = None
    for center in centers:
        counts = window_cell_counts(cache, scenarios, float(center))
        min_members = min(row["n_members_matched"] for row in counts)
        min_points = min(row["n_points_matched"] for row in counts)
        total_points = sum(row["n_points_matched"] for row in counts)

        # Lexicographic objective: protect weakest cell first, then total support.
        score = (min_members, min_points, total_points)
        if best is None or score > best[0]:
            best = (score, float(center), counts)

    if best is None:
        raise RuntimeError("Unable to identify a matched-GSAT window")

    _, center, counts = best
    if not window_is_adequate(counts):
        warnings.warn(
            "The best available matched-GSAT window does not meet the configured "
            "MIN_POINTS_PER_CELL / MIN_MEMBERS_PER_CELL thresholds. Inspect the "
            "diagnostic CSV before using the result in the manuscript."
        )

    return {
        "center": center,
        "low": center - MATCH_HALF_WIDTH,
        "high": center + MATCH_HALF_WIDTH,
        "source": "optimized",
        "common_low": common_low,
        "common_high": common_high,
        "support_rows": support_rows,
        "count_rows": counts,
    }

def overlap_bounds_model(cache, scenario, model):
    """Strict GSAT overlap between CO2-up and CO2-down for one model/scenario."""
    rows = []
    lowers, uppers = [], []
    for period in PERIODS:
        merged = cache[(scenario, model, period)]
        x = merged["value_x"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError(f"No SAT-GSAT data for {scenario}, {model}, {period}")
        lo = float(np.min(x))
        hi = float(np.max(x))
        q025, q975 = np.nanpercentile(x, [2.5, 97.5])
        rows.append({
            "scenario": scenario,
            "model": model,
            "period": period,
            "gsat_min": lo,
            "gsat_max": hi,
            "gsat_q025": float(q025),
            "gsat_q975": float(q975),
            "n_points_total": int(x.size),
            "n_members_total": int(merged["member_id"].nunique()),
        })
        lowers.append(lo)
        uppers.append(hi)
    return max(lowers), min(uppers), rows

def window_cell_counts_model(cache, scenario, model, center, half_width=MATCH_HALF_WIDTH):
    """Matched-window support for the two phases of one model/scenario."""
    low, high = center - half_width, center + half_width
    rows = []
    for period in PERIODS:
        merged = cache[(scenario, model, period)]
        selected = merged[merged["value_x"].between(low, high, inclusive="both")]
        rows.append({
            "scenario": scenario,
            "model": model,
            "period": period,
            "match_center": float(center),
            "match_low": float(low),
            "match_high": float(high),
            "n_points_matched": int(len(selected)),
            "n_members_matched": int(selected["member_id"].nunique()),
        })
    return rows

def choose_match_window_model(cache, scenario, model, preferred=PREFERRED_MATCH_CENTER):
    """
    Choose a matched-GSAT window within one model/scenario.

    The two phases must overlap in GSAT for this model. The preferred target is
    used when possible; otherwise the code scans the model-specific overlap and
    maximizes the weakest phase's member support, then point support.
    """
    common_low, common_high, support_rows = overlap_bounds_model(cache, scenario, model)

    if common_high <= common_low:
        warnings.warn(
            f"No CO2-up/CO2-down GSAT overlap for {SCENARIO_LABELS[scenario]}, {model}: "
            f"[{common_low:.3f}, {common_high:.3f}] K. Matched-GSAT PDF will be unavailable."
        )
        return {
            "center": np.nan, "low": np.nan, "high": np.nan, "source": "no_overlap",
            "common_low": common_low, "common_high": common_high,
            "support_rows": support_rows, "count_rows": [],
        }

    # If the strict overlap is narrower than the requested 0.20-K window, use
    # 90% of the available overlap instead of aborting. This keeps the analysis
    # within genuine overlapping climate states.
    half_width = MATCH_HALF_WIDTH
    if common_high - common_low < 2 * half_width:
        half_width = 0.45 * (common_high - common_low)
        warnings.warn(
            f"Narrow GSAT overlap for {SCENARIO_LABELS[scenario]}, {model}; "
            f"using adaptive half-width {half_width:.3f} K instead of "
            f"{MATCH_HALF_WIDTH:.3f} K."
        )

    if preferred is not None:
        preferred = float(preferred)
        if preferred - half_width >= common_low and preferred + half_width <= common_high:
            counts = window_cell_counts_model(cache, scenario, model, preferred, half_width)
            if window_is_adequate(counts):
                return {
                    "center": preferred, "low": preferred-half_width, "high": preferred+half_width,
                    "half_width": half_width, "source": "preferred",
                    "common_low": common_low, "common_high": common_high,
                    "support_rows": support_rows, "count_rows": counts,
                }

    start, stop = common_low + half_width, common_high - half_width
    if stop < start:
        center = 0.5 * (common_low + common_high)
        centers = np.array([center])
    else:
        centers = np.arange(start, stop + 0.5 * MATCH_SCAN_STEP, MATCH_SCAN_STEP)

    best = None
    for center in centers:
        counts = window_cell_counts_model(cache, scenario, model, float(center), half_width)
        min_members = min(r["n_members_matched"] for r in counts)
        min_points = min(r["n_points_matched"] for r in counts)
        # Penalize imbalance between phases after protecting weakest support.
        points = [r["n_points_matched"] for r in counts]
        imbalance = abs(points[0] - points[1])
        total_points = sum(points)
        score = (min_members, min_points, -imbalance, total_points)
        if best is None or score > best[0]:
            best = (score, float(center), counts)

    _, center, counts = best
    if not window_is_adequate(counts):
        warnings.warn(
            f"Best matched-GSAT window for {SCENARIO_LABELS[scenario]}, {model} "
            "does not meet MIN_POINTS_PER_CELL / MIN_MEMBERS_PER_CELL. "
            "Inspect the diagnostic CSV before interpreting the PDF."
        )

    return {
        "center": center, "low": center-half_width, "high": center+half_width,
        "half_width": half_width, "source": "optimized",
        "common_low": common_low, "common_high": common_high,
        "support_rows": support_rows, "count_rows": counts,
    }

def select_matched(merged, low, high):
    selected = merged[
        merged["value_x"].between(low, high, inclusive="both")
    ].copy()
    return selected[np.isfinite(selected["value_x"]) & np.isfinite(selected["value_y"])]
# ============================================================================
# MATCHED-GSAT CLUSTER BOOTSTRAP
# ============================================================================
def matched_mean_cluster_bootstrap(
    matched_up,
    matched_down,
    nboot=BOOTSTRAP_N,
    seed=BOOTSTRAP_SEED,
):
    """
    Bootstrap Delta T_matched = mean(T_down) - mean(T_up).

    Preferred mode is PAIRED member-cluster resampling: the same ensemble-member
    IDs are sampled for both phases, and every selected member keeps all of its
    matched-GSAT years. If fewer than 3 members contribute matched observations
    in both phases, the function falls back to independent cluster resampling.

    The estimand is the pooled member-year conditional mean, consistent with the
    displayed member-year conditional PDF; cluster resampling supplies uncertainty.
    """
    up = matched_up.dropna(subset=["member_id", "value_y"]).copy()
    down = matched_down.dropna(subset=["member_id", "value_y"]).copy()

    up_groups = {
        int(mid): g["value_y"].to_numpy(dtype=float)
        for mid, g in up.groupby("member_id")
    }
    down_groups = {
        int(mid): g["value_y"].to_numpy(dtype=float)
        for mid, g in down.groupby("member_id")
    }

    up_groups = {k: v[np.isfinite(v)] for k, v in up_groups.items() if np.isfinite(v).any()}
    down_groups = {k: v[np.isfinite(v)] for k, v in down_groups.items() if np.isfinite(v).any()}

    up_values = np.concatenate(list(up_groups.values())) if up_groups else np.array([])
    down_values = np.concatenate(list(down_groups.values())) if down_groups else np.array([])

    if up_values.size == 0 or down_values.size == 0:
        return {
            "mean_up": np.nan,
            "mean_down": np.nan,
            "delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_points_up": int(up_values.size),
            "n_points_down": int(down_values.size),
            "n_members_up": len(up_groups),
            "n_members_down": len(down_groups),
            "n_common_members": 0,
            "bootstrap_mode": "none",
        }

    observed_up = float(np.mean(up_values))
    observed_down = float(np.mean(down_values))
    observed_delta = observed_down - observed_up

    rng = np.random.default_rng(seed)
    common = sorted(set(up_groups).intersection(down_groups))
    boot_delta = np.full(nboot, np.nan, dtype=float)

    if len(common) >= 3:
        mode = "paired_member_cluster"
        common = np.asarray(common, dtype=int)
        for b in range(nboot):
            sampled = rng.choice(common, size=len(common), replace=True)
            vals_up = np.concatenate([up_groups[int(mid)] for mid in sampled])
            vals_down = np.concatenate([down_groups[int(mid)] for mid in sampled])
            boot_delta[b] = np.mean(vals_down) - np.mean(vals_up)
    else:
        mode = "independent_member_cluster"
        ids_up = np.asarray(sorted(up_groups), dtype=int)
        ids_down = np.asarray(sorted(down_groups), dtype=int)
        if len(ids_up) < 2 or len(ids_down) < 2:
            ci = [np.nan, np.nan]
            return {
                "mean_up": observed_up,
                "mean_down": observed_down,
                "delta": observed_delta,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_points_up": int(up_values.size),
                "n_points_down": int(down_values.size),
                "n_members_up": len(up_groups),
                "n_members_down": len(down_groups),
                "n_common_members": len(common),
                "bootstrap_mode": mode,
            }

        for b in range(nboot):
            samp_up = rng.choice(ids_up, size=len(ids_up), replace=True)
            samp_down = rng.choice(ids_down, size=len(ids_down), replace=True)
            vals_up = np.concatenate([up_groups[int(mid)] for mid in samp_up])
            vals_down = np.concatenate([down_groups[int(mid)] for mid in samp_down])
            boot_delta[b] = np.mean(vals_down) - np.mean(vals_up)

    valid = boot_delta[np.isfinite(boot_delta)]
    ci = np.nanpercentile(valid, [2.5, 97.5]) if valid.size else [np.nan, np.nan]

    return {
        "mean_up": observed_up,
        "mean_down": observed_down,
        "delta": observed_delta,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "n_points_up": int(up_values.size),
        "n_points_down": int(down_values.size),
        "n_members_up": len(up_groups),
        "n_members_down": len(down_groups),
        "n_common_members": len(common),
        "bootstrap_mode": mode,
    }
# %%
# PLOTTING
def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(
        axis="both", which="major", top=False, right=False, direction="out",
        labelsize=TICK_LABELSIZE, length=TICK_LENGTH, width=TICK_WIDTH,
    )

def set_xy_limits(ax, x_arrays, y_arrays, top_extra=0.12):
    if not x_arrays or not y_arrays:
        return
    x = np.concatenate(x_arrays)
    y = np.concatenate(y_arrays)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return

    xspan = np.ptp(x) if np.ptp(x) > 0 else 1.0
    yspan = np.ptp(y) if np.ptp(y) > 0 else 1.0
    ax.set_xlim(x.min() - 0.04 * xspan, x.max() + 0.04 * xspan)
    ax.set_ylim(y.min() - 0.08 * yspan, y.max() + top_extra * yspan)

def model_handles(period):
    return [
        Line2D(
            [0], [0],
            marker=MODEL_MARKERS[model],
            linestyle=PERIOD_LINESTYLES[period],
            color=PERIOD_MODEL_COLORS[period][model],
            markerfacecolor="none",
            markeredgewidth=1.2,
            markersize=6.5,
            linewidth=2.0,
            label=model,
        )
        for model in MODELS
    ]

R2_FONTSIZE = 11.5
R2_LINE_SPACING = 0.065


def plot_relationship_panel(
    ax,
    gsat,
    response,
    response_name,
    scenario,
    period,
    y_label,
    match_windows=None,
    add_hdr=False,
    regression_rows=None,
    legend_loc="lower right",
    r2_loc="upper left",
):
    style_axis(ax)
    x_arrays, y_arrays = [], []
    text_rows = []

    # Model-specific matched-GSAT windows. These are deliberately subtle:
    # the main visual object remains the joint SAT-GSAT distribution.
    if match_windows:
        for _model in MODELS:
            info = match_windows.get(_model)
            if not info or not np.isfinite(info.get("low", np.nan)):
                continue
            color_band = PERIOD_MODEL_COLORS[period][_model]
            ax.axvspan(info["low"], info["high"], color=color_band, alpha=0.055, zorder=0)
            ax.axvline(info["center"], color=color_band, alpha=0.35, lw=0.8, zorder=1)

    for model in MODELS:
        merged = merge_xy(gsat, response, model, scenario, period)
        if merged.empty:
            continue

        x = merged["value_x"].to_numpy(dtype=float)
        y = merged["value_y"].to_numpy(dtype=float)
        x, y = _finite_xy(x, y)
        if x.size < 3:
            continue

        color = PERIOD_MODEL_COLORS[period][model]
        alpha = SCATTER_ALPHA_MT if period == "mid-term" else SCATTER_ALPHA_LT

        ax.scatter(
            x,
            y,
            s=SCATTER_SIZE,
            marker=MODEL_MARKERS[model],
            facecolors="none",
            edgecolors=color,
            linewidths=0.8,
            alpha=alpha,
            zorder=2,
        )

        boot = member_cluster_bootstrap_regression(
            merged,
            nboot=BOOTSTRAP_N,
            seed=stable_seed("regression", response_name, scenario, period, model),
        )

        if np.isfinite(boot["slope"]):
            xfit = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            yfit = boot["intercept"] + boot["slope"] * xfit
            ax.plot(
                xfit,
                yfit,
                color=color,
                linestyle=PERIOD_LINESTYLES[period],
                linewidth=2.2,
                zorder=4,
            )

        if add_hdr:
            plot_hdr_contour(ax, x, y, color=color, mass=HDR_MASS)

        text_rows.append((model, boot["r2"], color))
        if regression_rows is not None:
            regression_rows.append(
                {
                    "response": response_name,
                    "scenario": scenario,
                    "model": model,
                    "period": period,
                    **boot,
                }
            )

        x_arrays.append(x)
        y_arrays.append(y)

    set_xy_limits(ax, x_arrays, y_arrays, top_extra=0.22)
    ax.axhline(0, color="0.72", linestyle="--", linewidth=0.9, zorder=1)
    ax.axvline(0, color="0.72", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xlabel(r"GSAT anomaly ($^\circ$C)")
    ax.set_ylabel(y_label)
    ax.set_title(PERIOD_LABELS[period], fontweight="bold")

    r2_rows = [row for row in text_rows if np.isfinite(row[1])]
    for i, (model, r2, color) in enumerate(r2_rows):
        if r2_loc == "lower left":
            y_pos, va = 0.02 + (len(r2_rows) - 1 - i) * R2_LINE_SPACING, "bottom"
        else:
            y_pos, va = 0.98 - i * R2_LINE_SPACING, "top"
        ax.text(
            0.02,
            y_pos,
            f"{MODEL_DISPLAY_NAMES[model]}: $R^2$={r2:.2f}",
            transform=ax.transAxes,
            color=color,
            fontsize=R2_FONTSIZE,
            ha="left",
            va=va,
        )

    ax.legend(
        handles=model_handles(period),
        loc=legend_loc,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )

def plot_matched_pdf_panel(
    ax,
    model,
    scenario,
    sat_cache,
    match_low,
    match_high,
    summary_rows,
    show_ylabel=False,
):
    style_axis(ax)

    up = select_matched(sat_cache[(scenario, model, "mid-term")], match_low, match_high)
    down = select_matched(sat_cache[(scenario, model, "long-term")], match_low, match_high)

    vals_up = up["value_y"].to_numpy(dtype=float)
    vals_down = down["value_y"].to_numpy(dtype=float)
    vals_up = vals_up[np.isfinite(vals_up)]
    vals_down = vals_down[np.isfinite(vals_down)]

    all_vals = np.concatenate([vals_up, vals_down]) if vals_up.size and vals_down.size else np.array([])
    if all_vals.size >= 5:
        span = np.ptp(all_vals) if np.ptp(all_vals) > 0 else 1.0
        grid = np.linspace(all_vals.min() - 0.15 * span, all_vals.max() + 0.15 * span, 300)

        for period, values in (("mid-term", vals_up), ("long-term", vals_down)):
            density = safe_1d_kde(values, grid)
            if density is None:
                continue
            color = PERIOD_MODEL_COLORS[period][model]
            ax.plot(
                grid,
                density,
                color=color,
                linestyle=PERIOD_LINESTYLES[period],
                linewidth=2.2,
                label=PERIOD_SHORT[period],
            )
            ax.fill_between(grid, 0, density, color=color, alpha=0.10)
            ax.axvline(
                np.mean(values),
                color=color,
                linestyle=PERIOD_LINESTYLES[period],
                linewidth=1.2,
                alpha=0.9,
            )

    boot = matched_mean_cluster_bootstrap(
        up,
        down,
        nboot=BOOTSTRAP_N,
        seed=stable_seed("matched", scenario, model, match_low, match_high),
    )

    summary_rows.append(
        {
            "scenario": scenario,
            "model": model,
            "match_low": match_low,
            "match_high": match_high,
            **boot,
        }
    )

    ax.set_title(MODEL_DISPLAY_NAMES[model], fontsize=10.5, fontweight="bold")
    ax.set_xlabel(r"Arctic SAT anomaly ($^\circ$C)")
    if show_ylabel:
        ax.set_ylabel("Conditional density")
    else:
        ax.set_ylabel("")

    if np.isfinite(boot["delta"]):
        ax.text(
            0.03,
            0.97,
            (
                rf"$\Delta T_{{matched}}$={boot['delta']:+.2f} $^\circ$C"
                "\n"
                rf"95% CI [{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}]"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            bbox=PDF_ANNOT_BBOX,
        )
        ax.text(
            0.03,
            0.78,
            (
                f"n↑={boot['n_points_up']} ({boot['n_members_up']} mem.)\n"
                f"n↓={boot['n_points_down']} ({boot['n_members_down']} mem.)"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.3,
            color="0.20",
            bbox=PDF_ANNOT_BBOX,
        )

    if PDF_XLIMITS.get(model) is not None:
        ax.set_xlim(*PDF_XLIMITS[model])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=PDF_LEGEND_FONTSIZE, loc="upper right")

def add_panel_letter(ax, letter, x=-0.09, y=1.04):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=PANEL_LETTER_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

def plot_scenario_figure(
    scenario,
    gsat,
    sat,
    sia,
    sat_cache,
    match_windows,
):
    regression_rows = []
    matched_rows = []

    fig = plt.figure(figsize=(16, 15), constrained_layout=True)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.15, 0.78, 1.15])

    # Row 1: SAT-GSAT joint relationship, split by phase.
    gs1 = outer[0].subgridspec(1, 2, wspace=0.15)
    ax_a = fig.add_subplot(gs1[0, 0])
    ax_b = fig.add_subplot(gs1[0, 1], sharey=ax_a)

    plot_relationship_panel(
        ax_a,
        gsat,
        sat,
        "SAT",
        scenario,
        "mid-term",
        r"Arctic SAT anomaly ($^\circ$C)",
        match_windows=match_windows,
        add_hdr=True,
        regression_rows=regression_rows,
        legend_loc="lower right",
    )
    plot_relationship_panel(
        ax_b,
        gsat,
        sat,
        "SAT",
        scenario,
        "long-term",
        r"Arctic SAT anomaly ($^\circ$C)",
        match_windows=match_windows,
        add_hdr=True,
        regression_rows=regression_rows,
        legend_loc="lower right",
    )
    ax_b.set_ylabel("")

    # Clarify what the contour/band mean without crowding each model legend.
    # ax_a.text(
    #     0.50,
    #     0.02,
    #     "Colored bands: model-specific matched-GSAT windows\n"
    #     f"Contours: {int(HDR_MASS * 100)}% joint HDR",
    #     transform=ax_a.transAxes,
    #     ha="center",
    #     va="bottom",
    #     fontsize=8.5,
    #     color="0.28",
    # )

    # Row 2: conditional PDFs, four model mini-panels.
    gs2 = outer[1].subgridspec(1, 4, wspace=0.12)
    pdf_axes = []
    for i, model in enumerate(MODELS):
        ax = fig.add_subplot(gs2[0, i])
        info = match_windows[model]
        if np.isfinite(info.get("low", np.nan)):
            plot_matched_pdf_panel(
                ax,
                model,
                scenario,
                sat_cache,
                info["low"],
                info["high"],
                matched_rows,
                show_ylabel=(i == 0),
            )
            ax.set_title(
                f"{MODEL_DISPLAY_NAMES[model]}\nGSAT {info['low']:.2f}-{info['high']:.2f} K",
                fontsize=9.7, fontweight="bold"
            )
        else:
            style_axis(ax)
            ax.set_title(MODEL_DISPLAY_NAMES[model], fontsize=10.5, fontweight="bold")
            ax.text(0.5, 0.5, "No overlapping GSAT\nstate between phases",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9)
            if i == 0:
                ax.set_ylabel("Conditional density")
            ax.set_xlabel(r"Arctic SAT anomaly ($^\circ$C)")
            matched_rows.append({
                "scenario": scenario, "model": model,
                "match_low": np.nan, "match_high": np.nan,
                "mean_up": np.nan, "mean_down": np.nan, "delta": np.nan,
                "ci_low": np.nan, "ci_high": np.nan,
                "n_points_up": 0, "n_points_down": 0,
                "n_members_up": 0, "n_members_down": 0,
                "n_common_members": 0, "bootstrap_mode": "unavailable",
            })
        # Keep configured ranges even when matched data are unavailable.
        if PDF_XLIMITS.get(model) is not None:
            ax.set_xlim(*PDF_XLIMITS[model])
        pdf_axes.append(ax)

    # Row 3: SIA-GSAT relationship, split by phase.
    gs3 = outer[2].subgridspec(1, 2, wspace=0.15)
    ax_g = fig.add_subplot(gs3[0, 0])
    ax_h = fig.add_subplot(gs3[0, 1], sharey=ax_g)

    plot_relationship_panel(
        ax_g,
        gsat,
        sia,
        "September SIA",
        scenario,
        "mid-term",
        r"September Arctic SIA anomaly\n(million km$^2$)",
        add_hdr=False,
        regression_rows=regression_rows,
        legend_loc="upper right",
        r2_loc="lower left",
    )
    plot_relationship_panel(
        ax_h,
        gsat,
        sia,
        "September SIA",
        scenario,
        "long-term",
        r"September Arctic SIA anomaly\n(million km$^2$)",
        add_hdr=False,
        regression_rows=regression_rows,
        legend_loc="upper right",
        r2_loc="lower left",
    )
    ax_h.set_ylabel("")

    # Panel letters.
    add_panel_letter(ax_a, "a")
    add_panel_letter(ax_b, "b")
    for letter, ax in zip(("c", "d", "e", "f"), pdf_axes):
        add_panel_letter(ax, letter, x=-0.11, y=1.03)
    add_panel_letter(ax_g, "g")
    add_panel_letter(ax_h, "h")

    return fig, pd.DataFrame(regression_rows), pd.DataFrame(matched_rows)
# %%
def main():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 12,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "savefig.transparent": True,
        }
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SAT...")
    sat = load_sat()
    print("Loading SIA...")
    sia = load_sia()
    print("Loading GSAT...")
    gsat = load_gsat()

    sat_cache = build_sat_merged_cache(gsat, sat)
    # ------------------------------------------------------------------
    # 1. Diagnose overlap and choose matched-GSAT band(s)
    # ------------------------------------------------------------------
    window_info = {}
    overlap_rows_all = []
    count_rows_all = []

    if MATCH_SCOPE == "global":
        # Retained as an optional diagnostic. This can legitimately fail when
        # different models occupy different absolute GSAT ranges.
        info = choose_match_window(sat_cache, SCENARIOS)
        for scenario in SCENARIOS:
            for model in MODELS:
                window_info[(scenario, model)] = info
        overlap_rows_all.extend(info["support_rows"])
        count_rows_all.extend(info["count_rows"])
        print(
            f"Global matched-GSAT band: {info['low']:.2f}-{info['high']:.2f} K "
            f"(center={info['center']:.2f}; source={info['source']})"
        )

    elif MATCH_SCOPE == "per_scenario":
        for scenario in SCENARIOS:
            info = choose_match_window(sat_cache, (scenario,))
            for model in MODELS:
                window_info[(scenario, model)] = info
            overlap_rows_all.extend(info["support_rows"])
            count_rows_all.extend(info["count_rows"])
            print(
                f"{SCENARIO_LABELS[scenario]} matched-GSAT band: "
                f"{info['low']:.2f}-{info['high']:.2f} K "
                f"(center={info['center']:.2f}; source={info['source']})"
            )

    elif MATCH_SCOPE == "per_model":
        print("Choosing matched-GSAT windows separately for each model and scenario...")
        for scenario in SCENARIOS:
            for model in MODELS:
                info = choose_match_window_model(sat_cache, scenario, model)
                window_info[(scenario, model)] = info
                overlap_rows_all.extend(info["support_rows"])
                count_rows_all.extend(info["count_rows"])
                if np.isfinite(info.get("low", np.nan)):
                    print(
                        f"  {SCENARIO_LABELS[scenario]:8s} {model:14s}: "
                        f"overlap [{info['common_low']:.2f}, {info['common_high']:.2f}] K; "
                        f"match {info['low']:.2f}-{info['high']:.2f} K "
                        f"({info['source']})"
                    )
                else:
                    print(
                        f"  {SCENARIO_LABELS[scenario]:8s} {model:14s}: "
                        "NO phase-overlapping GSAT range"
                    )
    else:
        raise ValueError("MATCH_SCOPE must be 'global', 'per_scenario', or 'per_model'")

    overlap_df = pd.DataFrame(overlap_rows_all).drop_duplicates()
    counts_df = pd.DataFrame(count_rows_all).drop_duplicates()

    overlap_path = FIG_DIR / "FIG3_matched_GSAT_support_diagnostics.csv"
    counts_path = FIG_DIR / "FIG3_matched_GSAT_window_counts.csv"
    overlap_df.to_csv(overlap_path, index=False)
    counts_df.to_csv(counts_path, index=False)
    print(f"Saved {overlap_path}")
    print(f"Saved {counts_path}")

    print("\nMatched-window counts:")
    if counts_df.empty:
        print("No valid matched-GSAT windows were found.")
    else:
        print(
            counts_df[
                [
                    "scenario", "model", "period", "match_low", "match_high",
                    "n_points_matched", "n_members_matched",
                ]
            ].to_string(index=False)
        )
    regression_summaries = []
    matched_summaries = []

    for scenario in SCENARIOS:
        match_windows = {model: window_info[(scenario, model)] for model in MODELS}
        fig, reg_summary, matched_summary = plot_scenario_figure(
            scenario,
            gsat,
            sat,
            sia,
            sat_cache,
            match_windows,
        )

        for extension in ("png", "pdf"):
            out = FIG_DIR / f"FIG3_three_row_{scenario}_jointHDR_matchedGSAT.{extension}"
            fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
            print(f"Saved {out}")
        plt.close(fig)

        regression_summaries.append(reg_summary)
        matched_summaries.append(matched_summary)

    regression_df = pd.concat(regression_summaries, ignore_index=True)
    matched_df = pd.concat(matched_summaries, ignore_index=True)

    reg_path = FIG_DIR / "FIG3_three_row_regression_bootstrap_summary.csv"
    matched_path = FIG_DIR / "FIG3_matched_GSAT_bootstrap_summary.csv"
    regression_df.to_csv(reg_path, index=False)
    matched_df.to_csv(matched_path, index=False)
    print(f"Saved {reg_path}")
    print(f"Saved {matched_path}")

    print("\nMatched-GSAT Delta T summary:")
    print(
        matched_df[
            [
                "scenario",
                "model",
                "match_low",
                "match_high",
                "mean_up",
                "mean_down",
                "delta",
                "ci_low",
                "ci_high",
                "n_members_up",
                "n_members_down",
                "bootstrap_mode",
            ]
        ].to_string(index=False)
    )
# %%
if __name__ == "__main__":
    main()
# %%