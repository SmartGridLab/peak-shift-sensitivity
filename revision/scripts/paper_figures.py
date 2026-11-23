"""Rendering routines adapted from the revised manuscript's Figure 5–8 scripts."""

from datetime import date
from math import hypot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


def save_figure(figure, path):
    svg = path.with_suffix(".svg")
    figure.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8")
    figure.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None})
    plt.close(figure)


def plot_shift(distributions, slots, out):
    boxes = boxplot_stats(distributions, whis=1.5)
    ink, grey, blue, fill = "#233341", "#788895", "#007da5", "#e5ebef"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 8,
        "axes.edgecolor": "#4b5563", "axes.linewidth": .8, "xtick.color": "#374151",
        "ytick.color": "#374151", "text.color": ink, "axes.labelcolor": ink,
        "svg.fonttype": "none", "svg.hashsalt": "fig5-annual-20261122", "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.45, 3.15))
    fig.subplots_adjust(left=.18, right=.98, bottom=.23, top=.86)
    xs = [slot / 2 for slot in slots]
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e9ed", linewidth=.7)
    ax.axhline(0, color="#6b7280", linewidth=.95, zorder=1)
    box_props = {"facecolor": fill, "edgecolor": grey, "linewidth": .9}
    line_props = {"color": grey, "linewidth": .85}
    flier_props = {"marker": "o", "markerfacecolor": "white", "markeredgecolor": grey,
                   "markeredgewidth": .7, "markersize": 2.3, "linestyle": "none"}
    ax.bxp(boxes, positions=xs, widths=.29, manage_ticks=False, patch_artist=True, showfliers=True,
           boxprops=box_props, whiskerprops=line_props, capprops=line_props,
           medianprops={"color": "#283b48", "linewidth": .85}, flierprops=flier_props)
    ax.scatter(xs, [box["mean"] for box in boxes], color=blue, s=12, edgecolors="white", linewidths=.4, zorder=4)
    ax.set(xlim=(-4.3, 4.3), ylim=(-25, 15), xticks=range(-4, 5), yticks=range(-25, 16, 5))
    if any(not -25 <= value <= 15 for series in distributions for value in series):
        raise ValueError("Shift results fall outside the manuscript plotting range")
    ax.set_xticklabels(["−4", "−3", "−2", "−1", "0", "+1", "+2", "+3", "+4"])
    ax.xaxis.set_minor_locator(MultipleLocator(.5))
    ax.tick_params(axis="both", direction="out", length=2.5, pad=3)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.set_xlabel("Additional circular shift [h]\n30-min steps, negative = earlier", labelpad=7)
    ax.set_ylabel("Peak reduction [%]", labelpad=6)
    ax.spines[["top", "right"]].set_visible(False)
    legend = [
        Patch(**box_props, label="Middle 50%"),
        Line2D([], [], color="#283b48", linewidth=.85, label="Median"),
        Line2D([], [], color=blue, marker="o", linestyle="none", markersize=3.5, label="Mean"),
        Line2D([], [], **(flier_props | {"markersize": 3.2}), label="Beyond whiskers"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .99),
               ncol=4, frameon=False, handlelength=1.2, columnspacing=.9, handletextpad=.45)
    save_figure(fig, out / "fig5")


def plot_baseline(distributions, out):
    boxes = boxplot_stats(distributions, whis=1.5)
    ink, grey, blue, fill = "#233341", "#788895", "#007da5", "#e5ebef"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 8,
        "axes.edgecolor": "#4b5563", "axes.linewidth": .8, "xtick.color": "#374151",
        "ytick.color": "#374151", "text.color": ink, "axes.labelcolor": ink,
        "svg.fonttype": "none", "svg.hashsalt": "fig6-annual-20261122", "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.45, 3.15))
    fig.subplots_adjust(left=.15, right=.98, bottom=.23, top=.86)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e9ed", linewidth=.7)
    box_style = {"facecolor": fill, "edgecolor": grey, "linewidth": .9}
    line_style = {"color": grey, "linewidth": .85}
    flier_style = {"marker": "o", "markerfacecolor": "white", "markeredgecolor": grey,
                   "markeredgewidth": .7, "markersize": 2.5, "linestyle": "none"}
    positions = [1, 2, 3]
    ax.bxp(boxes, positions=positions, widths=.46, manage_ticks=False, patch_artist=True,
           boxprops=box_style, whiskerprops=line_style, capprops=line_style, showfliers=True,
           medianprops={"color": "#283b48", "linewidth": .85}, flierprops=flier_style)
    ax.scatter(positions, [box["mean"] for box in boxes], color=blue, s=12,
               edgecolors="white", linewidths=.4, zorder=4)
    ax.set(xlim=(.5, 3.5), ylim=(0, 16), xticks=positions, yticks=[0, 5, 10, 15])
    ax.set_xticklabels(["Original\nLSTM", "Seasonal\nnaïve", "Perfect foresight\n(actual demand)"])
    ax.tick_params(axis="both", direction="out", length=2.5, pad=3)
    ax.set_xlabel("Planning inputs", labelpad=8)
    ax.set_ylabel("Peak reduction [%]", labelpad=6)
    ax.spines[["top", "right"]].set_visible(False)
    legend = [
        Patch(**box_style, label="Middle 50%"),
        Line2D([], [], color="#283b48", linewidth=.85, label="Median"),
        Line2D([], [], color=blue, marker="o", linestyle="none", markersize=3.5, label="Mean"),
        Line2D([], [], **(flier_style | {"markersize": 3.2}), label="Beyond whiskers"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .99),
               ncol=4, frameon=False, handlelength=1.2, columnspacing=.9, handletextpad=.45)
    save_figure(fig, out / "fig6")


def plot_corrections(effects, out):
    conditions = {
        "timing_oracle": ("Peak-time\nalignment", "Peak-time"),
        "magnitude_oracle": ("Magnitude\nscaling", "Magnitude"),
        "both_oracle": ("Combined\ncorrection", "Combined"),
    }
    boxes = {condition: boxplot_stats(list(effects[condition].values()), whis=1.5)[0]
             for condition in conditions}
    near_limits = (-5, 2.5)
    outside_near = {condition: {day for day, value in values.items()
                                if not near_limits[0] <= value <= near_limits[1]}
                    for condition, values in effects.items()}
    ink, grey, blue, fill, grid = "#233341", "#788895", "#007da5", "#e5ebef", "#e6e9ed"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.3, "text.color": ink,
                         "axes.labelcolor": ink, "xtick.color": ink, "svg.fonttype": "none",
                         "svg.hashsalt": "fig7-annual-20261122", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.45, 4.64))
    near = fig.add_axes([.25, .478, .50, .427])
    full = fig.add_axes([.25, .12, .73, .224])
    centers = [2.55, 1.55, .55]
    dot_height = .40

    def layout(ax, limits, ticks, labels, bottom_label_pad):
        ax.set(xlim=limits, ylim=(0, 3.23), xticks=ticks, yticks=centers)
        ax.set_xticklabels([f"{v:+d}".replace("-", "−") if v else "0" for v in ticks])
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", length=0, pad=9)
        ax.tick_params(axis="x", length=2.5, pad=3)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=grid, linewidth=.7)
        ax.axvline(0, color=ink, linewidth=.95, zorder=1)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(grey)
        ax.spines["bottom"].set_linewidth(.7)
        ax.set_xlabel("Change in peak reduction [pt]\nrelative to original LSTM", labelpad=bottom_label_pad)

    layout(near, near_limits, [-4, -2, 0, 2], [item[0] for item in conditions.values()], 4)
    layout(full, (-26, 3), [-25, -20, -15, -10, -5, 0], [item[1] for item in conditions.values()], 4)
    if any(not -26 <= value <= 3 for changes in effects.values() for value in changes.values()):
        raise ValueError("Paired effects fall outside the manuscript plotting range")

    # Place daily dots at their exact horizontal values. Vertical spacing is visual only.
    def dot_offsets(ax, values, diameter, step, low, high):
        x_points = ax.get_position().width * fig.get_figwidth() * 72 / (ax.get_xlim()[1] - ax.get_xlim()[0])
        y_points = ax.get_position().height * fig.get_figheight() * 72 / (ax.get_ylim()[1] - ax.get_ylim()[0])
        candidates = sorted((i * step for i in range(low, high + 1)), key=lambda y: (abs(y), y))
        placed = {}
        for day, value in sorted(values.items(), key=lambda item: (item[1], item[0])):
            offset = next((option for option in candidates
                           if all(hypot((value - values[other]) * x_points,
                                        (option - previous) * y_points) >= diameter
                                  for other, previous in placed.items())), None)
            if offset is None:
                # The compact overview may overlap observations, as in the prior figure.
                if ax is not full:
                    raise ValueError("Daily dots exceed the manuscript detail-panel space")
                return {day: (i % 7 - 3) * .073 for i, day in enumerate(values)}
            placed[day] = offset
        return placed


    for condition, center in zip(conditions, centers):
        visible = {day: v for day, v in effects[condition].items() if day not in outside_near[condition]}
        for ax, values, y, size, step, low, high in [
            (near, visible, center + dot_height, 2.5, .061, -4, 4),
            (full, effects[condition], center, 2.25, .045, -7, 7),
        ]:
            offsets = dot_offsets(ax, values, size, step, low, high)
            ax.scatter(list(values.values()), [y + offsets[d] for d in values],
                       s=size**2, color=grey, linewidths=0, zorder=3)
            if ax is full:
                for day in sorted(outside_near[condition]):
                    value = values[day]
                    lift = 9 if day == "2026-03-03" else -7.5
                    ax.annotate(f"{date.fromisoformat(day):%b} {date.fromisoformat(day).day}  {value:.2f}".replace("-", "−"),
                                (value, y + offsets[day]), xytext=(4, lift), textcoords="offset points",
                                ha="left", va="center", fontsize=6.8, color=ink,
                                arrowprops={"arrowstyle": "-", "lw": .5, "color": grey, "shrinkA": 1, "shrinkB": 2})

    near.bxp([boxes[condition] for condition in conditions], positions=centers, widths=.13,
             vert=False, manage_ticks=False, patch_artist=True, showfliers=False,
             boxprops={"facecolor": fill, "edgecolor": grey, "linewidth": .7},
             capprops={"color": grey, "linewidth": .7}, whiskerprops={"color": grey, "linewidth": .7},
             medianprops={"color": ink, "linewidth": .95})
    fig.text(.025, .921, "(a)  Detail near zero", fontsize=9.0, fontweight="bold")
    fig.text(.025, .365, "(b)  Full range", fontsize=9.0, fontweight="bold")
    fig.text(.875, .894, "Mean change\n[pt]", ha="center", va="top")
    fig.add_artist(Line2D([.771, .771], [.485, .902], color=grid, lw=.7, transform=fig.transFigure))
    for condition, center in zip(conditions, centers):
        y = near.get_position().y0 + near.get_position().height * center / 3.23
        label = f"{boxes[condition]['mean']:+.2f}".replace("-", "−")
        fig.text(.875, y, label, ha="center", va="center", color=blue, fontsize=9.0)
    fig.legend(handles=[
        Line2D([], [], marker="o", markersize=3, linestyle="none", color=grey, label="Daily change"),
        Patch(facecolor=fill, edgecolor=grey, lw=.7, label="Middle 50%"),
        Line2D([], [], color=ink, lw=1, label="Median"),
    ], loc="upper center", bbox_to_anchor=(.5, .991), ncol=3, frameon=False,
               handlelength=1.15, handletextpad=.45, columnspacing=1.05)
    save_figure(fig, out / "fig7")


def half_hour_label(slot):
    return f"{slot // 2:02d}:{slot % 2 * 30:02d}"


def plot_waveform(values, case, capacity, out):
    morning, evening = (case[key + "_slot"] / 2 for key in ("morning", "evening"))
    x = [slot / 2 for slot in range(48)]
    edges = x + [24]
    ink, blue, red, orange, grey, grid = "#233341", "#007da5", "#b3352d", "#ca711d", "#83919c", "#e0e5e9"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.35, "text.color": ink,
                         "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink,
                         "svg.fonttype": "none", "svg.hashsalt": "fig8-annual-20261122", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.45, 4.03))
    demand_ax = fig.add_axes([.147, .55, .799, .331])
    power_ax = fig.add_axes([.147, .305, .799, .178], sharex=demand_ax)
    soc_ax = fig.add_axes([.147, .118, .799, .126], sharex=demand_ax)

    for ax, ylim, yticks, label in [
        (demand_ax, (20, 51), [20, 30, 40, 50], "Demand [GW]"),
        (power_ax, (-11, 11), [-10, 0, 10], "Power [GW]"),
        (soc_ax, (0, 103), [0, 50, 100], "SoC [%]"),
    ]:
        ax.set(xlim=(0, 24), ylim=ylim, yticks=yticks, ylabel=label)
        ax.yaxis.set_label_coords(-.115, .5)
        ax.set_axisbelow(True)
        ax.grid(color=grid, linewidth=.65)
        ax.axvline(morning, color=grey, linewidth=.65, linestyle=(0, (1.6, 2.2)), zorder=1)
        ax.axvline(evening, color=grey, linewidth=.65, linestyle=(0, (1.6, 2.2)), zorder=1)
        ax.tick_params(length=2.5, pad=3)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color(ink)
            ax.spines[spine].set_linewidth(.65)

    for ax in [demand_ax, power_ax]:
        ax.tick_params(axis="x", labelbottom=False, length=0)
    soc_ax.set_xticks([0, 6, 12, 18, 24], ["00:00", "06:00", "12:00", "18:00", "24:00"])
    soc_ax.set_xlabel("Time of day [JST]", labelpad=5)

    curves = [
        ("actual_demand_kw", ink, "Actual"),
        ("controlled_actual_kw", red, "After control"),
        ("forecast_demand_kw", blue, "Aligned forecast"),
    ]
    handles = []
    for field, color, label in curves:
        series = [v / 1e6 for v in values[field]]
        handle, = demand_ax.plot(x, series, color=color, label=label, linewidth=1.08,
                                 marker="o", markersize=1.6, zorder=3)
        imax = max(range(48), key=series.__getitem__)
        demand_ax.plot(x[imax], series[imax], color=color, marker="o",
                       markerfacecolor="white", markeredgewidth=.95, markersize=4.1, zorder=5)
        handles.append(handle)
    handles.append(Line2D([], [], marker="o", markersize=4.1, markerfacecolor="white",
                          color=ink, linestyle="none", label="Daily maximum"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.555, .994),
               ncol=2, frameon=False, columnspacing=1.5, handlelength=1.65)

    after_peak, before_peak = case["controlled_peak_gw"], case["uncontrolled_peak_gw"]
    for x_end, y in [(morning, after_peak), (evening, before_peak)]:
        demand_ax.hlines(y, 1.0, x_end, color=grey, linestyle=(0, (1.5, 2.4)), linewidth=.55)
    demand_ax.annotate("", xy=(1.0, after_peak), xytext=(1.0, before_peak),
                       arrowprops={"arrowstyle": "<->", "color": red, "lw": .75, "shrinkA": 0, "shrinkB": 0})
    demand_ax.text(1.55, 44.0, f"+{case['amplification_gw']:.2f} GW", va="center",
                   color=red, fontweight="bold", fontsize=7.1)
    demand_ax.annotate(half_hour_label(case["morning_slot"]), (morning, after_peak), xytext=(7.15, 49.0), ha="right", va="center",
                       color=red, arrowprops={"arrowstyle": "-", "lw": .6, "color": red, "shrinkB": 3})
    demand_ax.annotate(half_hour_label(case["evening_slot"]), (evening, before_peak), xytext=(20.2, 45.8), va="center",
                       color=ink, arrowprops={"arrowstyle": "-", "lw": .6, "color": ink, "shrinkB": 3})

    power_ax.axhline(0, color=ink, linewidth=.65)
    for field, scale, color in [("charge_kw", 1e6, orange), ("discharge_kw", -1e6, blue)]:
        power = [max(0, v) / scale for v in values[field]]
        power_ax.stairs(power, edges, color=color, linewidth=1.05, zorder=3)
    power_ax.plot(morning + .25, case["morning_charge_gw"], "o", color=orange, markersize=3)
    power_ax.plot(evening + .25, -case["evening_discharge_gw"], "o", color=blue, markersize=3)
    fig.text(.535, .496, "Charging (+) ", color=orange, ha="right")
    fig.text(.545, .496, "/  Discharging (−)", color=blue, ha="left")

    soc = [values["soc_start_kwh"][0]] + values["soc_end_kwh"]
    soc_ax.plot(edges, [100 * energy / capacity for energy in soc], color=ink, linewidth=1.05)
    save_figure(fig, out / "fig8")
