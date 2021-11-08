import json
import math

import numpy as np
from sklearn import metrics


def _check_binary(y_true):
    if len(np.unique(y_true)) > 2:
        raise ValueError(
            "Only binary classification is supported for roc_curve."
        )


def _dump(content, output_file):
    with open(output_file, "w") as f:
        json.dump(content, f, indent=4)


def log_classification_report(y_true, y_pred, output_file, **kwargs):
    kwargs["output_dict"] = True

    classification_report = metrics.classification_report(
        y_true=y_true, y_pred=y_pred, **kwargs
    )

    _dump(classification_report, output_file)

    return classification_report


def log_roc_curve(y_true, y_score, output_file, **kwargs):
    _check_binary(y_true)

    fpr, tpr, roc_thresholds = metrics.roc_curve(
        y_true=y_true, y_score=y_score, **kwargs
    )

    roc = {
        "roc": [
            {"fpr": fp, "tpr": tp, "threshold": t}
            for fp, tp, t in zip(fpr, tpr, roc_thresholds)
        ]
    }

    _dump(roc, output_file)

    return roc


def log_precision_recall_curve(y_true, probas_pred, output_file, **kwargs):
    _check_binary(y_true)

    precision, recall, prc_thresholds = metrics.precision_recall_curve(
        y_true=y_true, probas_pred=probas_pred, **kwargs
    )

    # ROC has a drop_intermediate arg that reduces the number of points.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve.
    # PRC lacks this arg, so we manually reduce to 1000 as a rough estimate.
    nth_point = math.ceil(len(prc_thresholds) / 1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]

    prc = {
        "prc": [
            {"precision": p, "recall": r, "threshold": t}
            for p, r, t in prc_points
        ]
    }
    _dump(prc, output_file)

    return prc


def log_confusion_matrix(y_true, y_pred, output_file, **kwargs):
    confusion_matrix_disp = metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, **kwargs
    )

    confusion_matrix_disp.plot()
    confusion_matrix_disp.figure_.savefig(output_file)

    return confusion_matrix_disp
