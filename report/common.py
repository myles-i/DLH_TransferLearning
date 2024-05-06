import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def my_f1(y_true, y_prob, average="macro"):
    # set average=None to get per-class F1 scores.
    y_pred = y_prob >= np.max(y_prob, axis=1)[:, None]
    return f1_score(y_true, y_pred, average=average)


def _prepare_f1_for_plot(history_all):
    """
    For use by functions that plot F1 by epoch.
    """
    data = (
        history_all.groupby(["weight_type", "epoch"])
        .agg(
            min=pd.NamedAgg("f1", "min"),
            max=pd.NamedAgg("f1", "max"),
            mean=pd.NamedAgg("f1", "mean"),
        )
        # Move epoch from index to column
        .reset_index("epoch")
    )
    return data


def _make_macro_f1_table(predictions):
    """
    Build table with average macro F1 scores for each scenario.

    For use by make_f1_table().
    """
    macro = []
    for key, value in predictions.items():
        weight_type, seed = key.split("_")
        identifier = {"weight_type": weight_type, "seed": seed}
        # Details about my_f1 are in the Evaluation > Fine-tuning section.
        macro_f1 = my_f1(value["y_true"], value["y_prob"], average="macro")
        macro_f1 = {"f1": macro_f1}
        macro.append(identifier | macro_f1)

    macro_scores = pd.DataFrame(macro)
    macro_df = macro_scores.groupby(["weight_type"])[["f1"]].agg(
        mean=pd.NamedAgg("f1", "mean"),
        std=pd.NamedAgg("f1", "std"),
    )
    macro_df["macro_f1"] = _make_mean_std_column(
        macro_df["mean"], macro_df["std"]
    )
    macro_df = macro_df[["macro_f1"]]
    return macro_df


def _make_per_class_f1_table(predictions):
    """
    Build table with average per-class F1 scores for each scenario.

    For use by make_f1_table().
    """
    per_class = []
    for key, value in predictions.items():
        weight_type, seed = key.split("_")
        identifier = {"weight_type": weight_type, "seed": seed}

        class_f1 = my_f1(value["y_true"], value["y_prob"], average=None)
        class_f1 = dict(zip(value["classes"], class_f1))
        per_class.append(identifier | class_f1)

    per_class_scores = pd.DataFrame(per_class)
    per_class_macro = per_class_scores.groupby(["weight_type"])[
        ["N", "A", "O", "~"]
    ].agg(
        **{
            "N_mean": pd.NamedAgg("N", "mean"),
            "A_mean": pd.NamedAgg("A", "mean"),
            "O_mean": pd.NamedAgg("O", "mean"),
            "~_mean": pd.NamedAgg("~", "mean"),
            "N_std": pd.NamedAgg("N", "std"),
            "A_std": pd.NamedAgg("A", "std"),
            "O_std": pd.NamedAgg("O", "std"),
            "~_std": pd.NamedAgg("~", "std"),
        }
    )

    per_class_macro["N_f1"] = _make_mean_std_column(
        per_class_macro["N_mean"], per_class_macro["N_std"]
    )
    per_class_macro["A_f1"] = _make_mean_std_column(
        per_class_macro["A_mean"], per_class_macro["A_std"]
    )
    per_class_macro["O_f1"] = _make_mean_std_column(
        per_class_macro["O_mean"], per_class_macro["O_std"]
    )
    per_class_macro["~_f1"] = _make_mean_std_column(
        per_class_macro["~_mean"],
        per_class_macro["~_std"],
    )
    per_class_macro = per_class_macro[["N_f1", "A_f1", "O_f1", "~_f1"]]
    return per_class_macro


def _make_mean_std_column(mean_col, std_col, decimals=3):
    """
    Helper to produce column values formatted like in Table 1 of the paper.
    """
    return (
        mean_col.astype(str)
        # 1: decimal point's index, e.g.: '0.xxx'
        # 1 + decimals + 1: (decimals + 1) is total string length, counting
        # the decimal point itself. The (1 +) is because we started the
        # substring at index 1.
        .str[1 : 1 + decimals + 1]
        .str.pad(decimals + 1, side="right", fillchar="0")
        + " (\u00b1 "  # plus minus symbol
        + std_col.astype(str)
        .str[1 : 1 + decimals + 1]
        .str.pad(decimals + 1, side="right", fillchar="0")
        + ")"
    )
