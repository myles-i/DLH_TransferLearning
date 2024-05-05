import matplotlib.pyplot as plt
import pandas as pd

from report import common


def make_epoch_table(history_all, decimals=1):
    table = (
        history_all.groupby(["weight_type", "seed"])
        .max()
        .groupby("weight_type")
        .agg(
            Mean=pd.NamedAgg("epoch", "mean"),
            Std=pd.NamedAgg("epoch", "std"),
        )
        .round(decimals)
        .reindex(["random", "1", "10", "20", "100"])
        .rename(
            index={
                "random": "Random",
                "1": "Pre-train 1",
                "10": "Pre-train 10",
                "20": "Pre-train 20",
                "100": "Pre-train 100",
            }
        )
        .reset_index()
        .rename(
            columns={
                "weight_type": "Scenario",
            }
        )
    )
    # Explicitly select columns in desired order.
    table = table[["Scenario", "Mean", "Std"]]
    return table


def plot_f1_by_epoch(history_all):
    data = common._prepare_f1_for_plot(history_all)
    fig, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Macro F1")
    ax.plot(
        "epoch",
        "mean",
        color="tab:red",
        data=data.loc["random"],
        label="Random",
    )
    ax.plot(
        "epoch",
        "mean",
        color="tab:purple",
        data=data.loc["1"],
        label="Pre-train 1",
    )
    ax.plot(
        "epoch",
        "mean",
        color="tab:blue",
        data=data.loc["10"],
        label="Pre-train 10",
    )
    ax.plot(
        "epoch",
        "mean",
        color="tab:orange",
        data=data.loc["20"],
        label="Pre-train 20",
    )
    ax.plot(
        "epoch",
        "mean",
        color="tab:green",
        data=data.loc["100"],
        label="Pre-train 100",
    )
    ax.legend()
    # Replicate y-axis in Figure 3(a) of the paper.
    ax.set_ylim([0.5, 1.0])


def plot_f1_by_epoch_with_range(history_all):
    # Comparable to Figure 3(a) in the paper.
    data = common._prepare_f1_for_plot(history_all)
    fig, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Macro F1")
    ax.plot(
        "epoch",
        "mean",
        color="tab:red",
        data=data.loc["random"],
        label="Random",
    )
    ax.plot(
        "epoch",
        "mean",
        color="tab:blue",
        data=data.loc["100"],
        label="Pre-train 100",
    )
    ax.legend()

    # Add min, max range
    ax.fill_between(
        "epoch",
        "min",
        "max",
        data=data.loc["random"],
        color="tab:red",
        alpha=0.15,
    )
    ax.fill_between(
        "epoch",
        "min",
        "max",
        data=data.loc["100"],
        color="tab:blue",
        alpha=0.15,
    )
    # Replicate y-axis in Figure 3(a) of the paper.
    ax.set_ylim([0.5, 1.0])


def make_f1_table(predictions):
    """
    Actual function to build a combined table with average macro F1 and
    average per-class F1 scores for each scenario.
    """
    macro_f1 = common._make_macro_f1_table(predictions)
    per_class_f1 = common._make_per_class_f1_table(predictions)
    table = (
        macro_f1.join(per_class_f1)
        .reindex(["random", "10", "20"])
        .rename(
            index={
                "random": "Random",
                "10": "Pre-train 10",
                "20": "Pre-train 20",
            }
        )
        .reset_index()
        # Using same names as Table 1 in the paper.
        .rename(
            columns={
                "weight_type": "Type",
                "macro_f1": "F1",
                "N_f1": "F1n",  # Normal
                "A_f1": "F1a",  # AF
                "O_f1": "F1o",  # Other
                "~_f1": "F1p",  # Noisy
            }
        )
    )
    # Explicitly select columns in desired order.
    table = table[["Type", "F1", "F1n", "F1a", "F1o", "F1p"]]
    return table
