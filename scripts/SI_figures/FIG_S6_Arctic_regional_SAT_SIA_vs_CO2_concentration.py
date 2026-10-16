#!/usr/bin/env python3
# %%
"""Extended Figure S5: Arctic SAT and SIA relationships with atmospheric CO2."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(
    "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/"
    "Atlantic_capacitor_on_Arctic"
)
SAT_DIR = ROOT / "docs/data/Regional_diagnosis/regional_mean"
SIA_DIR = ROOT / "docs/data/Regional_diagnosis/SIA"
CO2_DIR = Path(
    "/work/mh0033/m301036/Land_surf_temp/Observational_constraints/"
    "CO2_concentration_TS/greenhouse_ssp"
)
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
SAT_FILE_MODELS = {
    "MIROC6": "MIROC",
    "MIROC-ES2L": "MIROC-ES2L",
    "MPI-ESM1-2-LR": "MPI-ESM-LR",
    "CanESM5": "CanESM5",
}
MODEL_MARKERS = {
    "MIROC6": "o",
    "MIROC-ES2L": "s",
    "MPI-ESM1-2-LR": "^",
    "CanESM5": "D",
}

PERIODS = {
    "mid-term": (2015, 2049),
    "long-term": (2065, 2099),
}
PERIOD_LABELS = {
    "mid-term": "CO$_2$ increasing (2015-2049)",
    "long-term": "CO$_2$ decreasing (2065-2099)",
}
PERIOD_LINESTYLES = {
    "mid-term": "-",
    "long-term": "--",
}
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
# %%
def period_for_year(year):
    for period, (start, end) in PERIODS.items():
        if start <= year <= end:
            return period
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

def load_sat():
    frames = []
    variable = "ArcticOcean66.5N"

    for model in MODELS:
        file_model = SAT_FILE_MODELS[model]
        path = SAT_DIR / f"{file_model}_ArcticOcean66.5N_RegionalMean_1850_2100.nc"
        with xr.open_dataset(path) as ds:
            da = ds[variable].sel(scenario=list(SCENARIOS), year=slice(2015, 2099)).load()

        frame = da.to_dataframe(name="value").reset_index()
        frame["model"] = model
        frame["scenario"] = frame["scenario"].astype(str)
        frame["member"] = frame["member"].astype(str)
        frame["year"] = frame["year"].astype(int)
        frame["period"] = frame["year"].map(period_for_year)
        frames.append(frame.dropna(subset=["value", "period"]))

    return pd.concat(frames, ignore_index=True)[
        ["model", "scenario", "member", "year", "period", "value"]
    ]

def load_sia():
    frames = []

    for model in MODELS:
        for scenario in SCENARIOS:
            path = (
                SIA_DIR
                / f"{model}_{scenario}_ArcticOcean66p5N_SIA_09_ano_wrt_1850-1900.nc"
            )
            with xr.open_dataset(path) as ds:
                if "SIA_anom" not in ds:
                    raise KeyError(f"SIA_anom missing in {path}")
                da = ds["SIA_anom"].sel(time=slice("2015-01-01", "2099-12-31")).load()

            if "member" not in da.dims:
                da = da.expand_dims(member=["r1i1p1f1"])

            invalid_members = da.isnull().any("time")
            valid_members = da["member"].where(~invalid_members, drop=True)
            da = da.sel(member=valid_members)

            frame = da.to_dataframe(name="value").reset_index()
            frame["model"] = model
            frame["scenario"] = scenario
            frame["member"] = frame["member"].astype(str)
            frame["year"] = frame["time"].map(extract_year).astype("Int64")
            frame = frame.dropna(subset=["year"])
            frame["year"] = frame["year"].astype(int)
            frame["period"] = frame["year"].map(period_for_year)
            frames.append(frame.dropna(subset=["value", "period"]))

    return pd.concat(frames, ignore_index=True)[
        ["model", "scenario", "member", "year", "period", "value"]
    ]

def load_co2():
    frames = []

    for scenario in SCENARIOS:
        path = CO2_DIR / f"greenhouse_{scenario}.nc"
        with xr.open_dataset(path, decode_times=False) as ds:
            da = ds["CO2"].squeeze(drop=True).sel(time=slice(1980, 2099)).load()

        frame = da.to_dataframe(name="CO2").reset_index()
        frame["scenario"] = scenario
        frame["year"] = frame["time"].astype(int)
        reference = frame.loc[frame["year"] == 1980, "CO2"]
        if len(reference) != 1:
            raise ValueError(f"Expected one 1980 CO2 value for {scenario}, found {len(reference)}")
        frame["CO2_log1980"] = np.log(frame["CO2"] / reference.iloc[0])
        frame["period"] = frame["year"].map(period_for_year)
        frames.append(frame[["scenario", "year", "period", "CO2", "CO2_log1980"]])

    return pd.concat(frames, ignore_index=True)

def merge_with_co2(target, co2):
    return target.merge(
        co2[["scenario", "year", "CO2", "CO2_log1980"]],
        on=["scenario", "year"],
        how="inner",
        validate="many_to_one",
    )

def linear_fit(x, y):
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[finite]
    y = np.asarray(y, dtype=float)[finite]
    if x.size < 2 or np.allclose(x, x[0]):
        return None, np.nan

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = np.sum((y - predicted) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    r_squared = np.nan if total == 0 else 1.0 - residual / total
    return (slope, intercept), r_squared

def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(
        axis="both", which="major", top=False, right=False, direction="out",
        labelsize=14, length=7, width=1.5,
    )

def set_limits(ax, x_values, y_values, add_annotation_space=True):
    x = np.concatenate(x_values)
    y = np.concatenate(y_values)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    x_pad = 0.04 * (x_span if x_span else 1.0)
    y_pad = 0.08 * (y_span if y_span else 1.0)
    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
    upper_pad = 0.40 * (y_span if y_span else 1.0) if add_annotation_space else y_pad
    ax.set_ylim(y.min() - y_pad, y.max() + upper_pad)

def model_legend_handles(period):
    return [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[model],
            linestyle=PERIOD_LINESTYLES[period],
            color=PERIOD_MODEL_COLORS[period][model],
            markerfacecolor="none",
            markeredgewidth=1.3,
            markersize=7,
            linewidth=2.0,
            label=MODEL_DISPLAY_NAMES[model],
        )
        for model in MODELS
    ]

def plot_panel(ax, data, scenario, period, x_column, x_label, y_label):
    style_axis(ax)
    x_values = []
    y_values = []
    r_squared = {}

    for model in MODELS:
        subset = data[
            (data["model"] == model)
            & (data["scenario"] == scenario)
            & (data["period"] == period)
        ]
        if subset.empty:
            continue

        x = subset[x_column].to_numpy(dtype=float)
        y = subset["value"].to_numpy(dtype=float)
        color = PERIOD_MODEL_COLORS[period][model]

        ax.scatter(
            x,
            y,
            s=35,
            marker=MODEL_MARKERS[model],
            facecolors="none",
            edgecolors=color,
            linewidths=1.0,
            alpha=0.38 if period == "mid-term" else 0.30,
            zorder=3,
        )

        fit, score = linear_fit(x, y)
        if fit is not None:
            x_fit = np.linspace(x.min(), x.max(), 200)
            slope, intercept = fit
            ax.plot(
                x_fit,
                slope * x_fit + intercept,
                color=color,
                linestyle=PERIOD_LINESTYLES[period],
                linewidth=2.2,
                zorder=4,
            )
            r_squared[model] = score

        x_values.append(x)
        y_values.append(y)

    if not x_values:
        raise ValueError(f"No matched data for {scenario}, {period}, {y_label}")

    set_limits(ax, x_values, y_values)
    if y_label.startswith("September Arctic SIA"):
        ax.set_ylim(top=6)
    ax.axhline(0, color="0.72", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    tag = "MT" if period == "mid-term" else "LT"
    for row, model in enumerate(MODELS):
        if model not in r_squared:
            continue
        ax.text(
            0.02,
            0.98 - row * 0.085,
            f"{MODEL_DISPLAY_NAMES[model]}: $R^2_{{{tag}}}$={r_squared[model]:.2f}",
            transform=ax.transAxes,
            color=PERIOD_MODEL_COLORS[period][model],
            fontsize=15,
            ha="left",
            va="top",
        )

def plot_figure(sat, sia, x_column, x_label, output_stem):
    fig = plt.figure(figsize=(16, 20), constrained_layout=True)
    outer = fig.add_gridspec(5, 1, height_ratios=[0.10, 1.0, 0.10, 1.0, 0.15])
    periods = ("mid-term", "long-term")
    metrics = (
        (sat, "Arctic SAT anomaly (°C)"),
        (sia, "September Arctic SIA anomaly\n(million km²)"),
    )
    letters = iter("abcdefgh")

    for scenario_index, scenario in enumerate(SCENARIOS):
        # One centered scenario heading spans its four panels.
        heading = fig.add_subplot(outer[2 * scenario_index])
        heading.set_axis_off()
        heading.text(
            0.5, 0.5, SCENARIO_LABELS[scenario],
            transform=heading.transAxes, ha="center", va="center",
            fontsize=20, fontweight="bold",
        )
        grid = outer[2 * scenario_index + 1].subgridspec(2, 2)
        for metric_index, (data, y_label) in enumerate(metrics):
            for column, period in enumerate(periods):
                ax = fig.add_subplot(grid[metric_index, column])
                plot_panel(ax, data, scenario, period, x_column, x_label, y_label)
                if column != 0:
                    ax.set_ylabel("")
                if scenario_index != len(SCENARIOS) - 1 or metric_index != len(metrics) - 1:
                    ax.set_xlabel("")
                if metric_index == 0:
                    ax.set_title(PERIOD_LABELS[period], fontsize=16, fontweight="bold", pad=14)
                ax.text(
                    -0.09, 1.02, next(letters), transform=ax.transAxes,
                    fontsize=22, fontweight="bold", ha="left", va="bottom",
                )

    # A shared 2-by-2 model legend below both scenario blocks.
    legend_ax = fig.add_subplot(outer[4])
    legend_ax.set_axis_off()
    handles_up = model_legend_handles("mid-term")
    handles_down = model_legend_handles("long-term")
    legend_ax.legend(
        handles=list(zip(handles_up, handles_down)),
        labels=[MODEL_DISPLAY_NAMES[m] for m in MODELS],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
        loc="center", ncol=2, frameon=False, fontsize=14,
        handlelength=4.5, columnspacing=3.0, labelspacing=1.0,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = FIG_DIR / f"{output_stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved {path}")
    plt.close(fig)

def validate_inputs(sat, sia, co2):
    for name, data in (("SAT", sat), ("SIA", sia)):
        counts = data.groupby(["model", "scenario", "period"]).size()
        expected = pd.MultiIndex.from_product(
            [MODELS, SCENARIOS, PERIODS], names=["model", "scenario", "period"]
        )
        missing = expected.difference(counts.index)
        if len(missing):
            raise ValueError(f"{name} is missing model/scenario/period combinations: {missing}")
        print(f"{name} matched sample counts:")
        print(counts.to_string())

    log_values = co2["CO2_log1980"].to_numpy()
    if not np.isfinite(log_values).all():
        raise ValueError("CO2_log1980 contains non-finite values")

    reference = co2[co2["year"] == 1980].set_index("scenario")["CO2_log1980"]
    if not np.allclose(reference.reindex(SCENARIOS).to_numpy(), 0.0):
        raise ValueError("CO2_log1980 is not zero in 1980")

    for scenario in SCENARIOS:
        scenario_co2 = co2[co2["scenario"] == scenario].set_index("year")["CO2"]
        if scenario_co2.loc[2049] <= scenario_co2.loc[2015]:
            raise ValueError(f"{scenario} does not increase over 2015-2049")
        if scenario_co2.loc[2099] >= scenario_co2.loc[2065]:
            raise ValueError(f"{scenario} does not decrease over 2065-2099")
# %%
def main():
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 22,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "savefig.transparent": True,
        }
    )

    sat = load_sat()
    sia = load_sia()
    co2 = load_co2()
    validate_inputs(sat, sia, co2)

    sat_co2 = merge_with_co2(sat, co2)
    sia_co2 = merge_with_co2(sia, co2)
    if len(sat_co2) != len(sat) or len(sia_co2) != len(sia):
        raise ValueError("CO2 merge dropped SAT or SIA observations")

    plot_figure(
        sat_co2,
        sia_co2,
        x_column="CO2",
        x_label="Atmospheric CO$_2$ concentration (ppm)",
        output_stem="Extended_Data_FIG5_Arctic_SAT_SIA_vs_CO2_linear",
    )
    # plot_figure(
    #     sat_co2,
    #     sia_co2,
    #     x_column="CO2_log1980",
    #     x_label=r"$\ln(CO_2 / CO_{2,1980})$",
    #     output_stem="Extended_Data_FIG5_Arctic_SAT_SIA_vs_CO2_log1980",
    # )
# %%
if __name__ == "__main__":
    main()
# %%
