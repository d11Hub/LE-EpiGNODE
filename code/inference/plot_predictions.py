"""Plot observations and checkpoint forecasts, without evaluation metrics."""
from __future__ import annotations

import math
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml


def draw_panels(path, x, observed, predicted, titles, boundary, ylabel, dated):
    columns = min(4, len(titles))
    rows = math.ceil(len(titles) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.1 * columns, 2.8 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= len(titles):
            axis.set_visible(False)
            continue
        axis.plot(x, observed[:, index], color="#222222", lw=1, label="Observed")
        axis.plot(x, predicted[:, index], color="#2474b5", lw=1.1, label="Model prediction")
        axis.axvspan(x[boundary], x[-1], color="#e8eff7", alpha=0.65, zorder=-1)
        axis.axvline(x[boundary], color="#9a4d10", lw=0.9, ls="--", label="Forecast origin")
        axis.set_title(titles[index], fontsize=10)
        axis.set_xlabel("Date" if dated else "Simulation day")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.15)
        if dated:
            locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(scenario: str, kind: str, data: Path, config_path: Path, output: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    states_name = "states_age_noisy.npy" if scenario.startswith("multiscale") else "states_age.npy"
    states = np.load(data / states_name)
    days, nodes, _ = states.shape
    cities, ages = int(config["num_cities"]), int(config["num_ages"])
    factor = float(config["re_factor"])
    observed = states[..., -1].astype(np.float64)
    if scenario == "ontario":
        raw = np.loadtxt(data / "states_age_raw_flat.csv", delimiter=",", skiprows=1).reshape(states.shape)[..., -1]
        nonzero = raw != 0
        factor = float(np.median(observed[nonzero] / raw[nonzero]))
        observed = raw
    elif kind == "synthetic":
        observed /= factor
    predicted = np.load(output / "pred_u.npy")[..., 0].astype(np.float64) / factor
    predicted_city = np.load(output / "pred_u_city.npy")[..., 0].astype(np.float64) / factor
    if kind == "real":
        dates = pd.read_csv(data / "aligned_dates.csv").iloc[:, 0]
        x = pd.to_datetime(dates).to_numpy()
        if len(x) != days:
            raise ValueError("Date count does not match the state trajectory")
        labels = json.loads((data / "node_labels.json").read_text(encoding="utf-8"))
        city_names, age_names = labels["cities"], labels["ages"]
        if len(city_names) != cities or len(age_names) != ages:
            raise ValueError('Display-label counts do not match the provided node order')
    else:
        time_file = data / "time.npy"
        x = np.load(time_file).reshape(-1) if time_file.exists() else np.arange(days)
        city_names = [f"City {i + 1}" for i in range(cities)]
        age_names = [f"Age group {i + 1}" for i in range(ages)]
    boundary = int(config["training_time"])
    if not 0 < boundary < days or nodes != cities * ages:
        raise ValueError("Invalid forecast boundary or node order")
    titles = [city + " / " + age for city in city_names for age in age_names]
    draw_panels(output / "case_forecast.png", x, observed, predicted, titles, boundary,
                "Daily cases (gammaI)", kind == "real")
    observed_city = observed.reshape(days, cities, ages).sum(axis=2)
    draw_panels(output / "city_forecast.png", x, observed_city, predicted_city, city_names, boundary,
                "Daily cases (gammaI)", kind == "real")
    mobility_file = output / "pred_mobility.npy"
    if mobility_file.exists():
        observation_file = data / ("Baidu_Migration_noisy_series.npy" if scenario.startswith("multiscale")
                                   else "Baidu_Migration_series.npy")
        mobility = np.load(observation_file)
        forecast = np.load(mobility_file)
        edges = [(i, j) for i in range(cities) for j in range(cities) if i != j]
        observed_edges = np.column_stack([mobility[:, i, j] for i, j in edges])
        predicted_edges = np.column_stack([forecast[:, i, j] for i, j in edges])
        edge_titles = [city_names[i] + " -> " + city_names[j] for i, j in edges]
        ylabel = "Daily OD flow" if scenario == "spain" else "Mobility index"
        draw_panels(output / "mobility_forecast.png", x, observed_edges, predicted_edges, edge_titles,
                    boundary, ylabel, kind == "real")
