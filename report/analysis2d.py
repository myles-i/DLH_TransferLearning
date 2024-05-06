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
        .reindex(["random", "1", "10", "20", "88"])
        .rename(
            index={
                "random": "Random",
                "1": "Pre-train 1",
                "10": "Pre-train 10",
                "20": "Pre-train 20",
                "88": "Pre-train 88",
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
        data=data.loc["88"],
        label="Pre-train 88",
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
    ax.plot("epoch", "mean", data=data.loc["88"], label="Pre-train 88")
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
    ax.fill_between("epoch", "min", "max", data=data.loc["88"], alpha=0.15)
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
        .reindex(["random", "1", "10", "20", "88"])
        .rename(
            index={
                "random": "Random",
                "1": "Pre-train 1",
                "10": "Pre-train 10",
                "20": "Pre-train 20",
                "88": "Pre-train 88",
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


def _format_f1_by_pretrain_percentage(predictions):
    """
    Helper for plot_f1_by_pretrain_percentage().

    Input:
        dict that is from a predictions_all.pkl file, expected
        to be produced by a collation job of all the trials for
        all scenarios for a particular model configuration.

    Output:
        dataframe with weight_type (int), f1 (float) columns
    """
    macro = []
    for key, value in predictions.items():
        weight_type, seed = key.split("_")
        identifier = {"weight_type": weight_type, "seed": seed}
        # Details about my_f1 are in the Evaluation > Fine-tuning section.
        macro_f1 = common.my_f1(
            value["y_true"], value["y_prob"], average="macro"
        )
        macro_f1 = {"f1": macro_f1}
        macro.append(identifier | macro_f1)

    macro_scores = pd.DataFrame(macro)
    macro_df = (
        macro_scores.groupby(["weight_type"])[["f1"]]
        .mean()
        # Random scenario is considered 0% pre-training.
        .rename(index={"random": 0})
        # Ensure output is in tabular format
        .reset_index()
        .rename(columns={"weight_type": "scenario"})
    )
    # When 'random' is part of weight_type column, the column is type 'object',
    # which matplotlib plot does not like because plot expects numeric types.
    # But since we've mapped 'random' to the number zero, we simply cast to
    # fix this issue.
    macro_df["scenario"] = macro_df["scenario"].astype(int)
    # matplotlib appears to connect dots in the order they appear in the
    # input to plot, so we sort in ascending order of pre-train percentage.
    macro_df = macro_df.sort_values("scenario")
    return macro_df


def plot_f1_by_pretrain_percentage(predictions1d, predictions2d):
    """
    Make a macro F1 plot versus pre-train percentage.

    We compare 1-D results to 2-D results.

    Both predictions1d and predictions2d come from the pickle that is a
    collation of all the trials for the model type, across the different
    scenarios.
    """
    data_1d = _format_f1_by_pretrain_percentage(predictions1d)
    data_2d = _format_f1_by_pretrain_percentage(predictions2d)

    fig, ax = plt.subplots()
    ax.set_xlabel("Pre-train Percentage (%)")
    ax.set_ylabel("Average Macro F1")
    # ^ marker is triangle
    ax.plot("scenario", "f1", data=data_1d, marker="^", label="1-D Model")
    # s marker is square
    ax.plot("scenario", "f1", data=data_2d, marker="s", label="2-D Model")
    ax.legend()
    ax.set_ylim([0.7, 0.8])
    ax.grid()
