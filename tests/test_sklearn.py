import json
from pathlib import Path

import pytest

from dvclive.sklearn import (
    log_classification_report,
    log_confusion_matrix,
    log_precision_recall_curve,
    log_roc_curve,
)

# pylint: disable=redefined-outer-name, unused-argument


@pytest.fixture
def y_true_y_pred_y_score():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X, y = make_classification(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    return y_test, y_pred, y_score


def test_log_classification_report(tmp_dir, y_true_y_pred_y_score, mocker):
    from dvclive.sklearn import metrics

    y_true, y_pred, _ = y_true_y_pred_y_score

    spy = mocker.spy(metrics, "classification_report")

    classification_report = log_classification_report(
        y_true=y_true, y_pred=y_pred, output_file="classification_report.json"
    )

    spy.assert_called_once_with(y_true, y_pred, output_dict=True)

    assert (
        json.dumps(classification_report, indent=4)
        == Path("classification_report.json").read_text()
    )


def test_log_roc_curve(tmp_dir, y_true_y_pred_y_score, mocker):
    from dvclive.sklearn import metrics

    y_true, _, y_score = y_true_y_pred_y_score

    spy = mocker.spy(metrics, "roc_curve")

    roc = log_roc_curve(y_true=y_true, y_score=y_score, output_file="roc.json")

    spy.assert_called_once_with(y_true, y_score)

    assert json.dumps(roc, indent=4) == Path("roc.json").read_text()


def test_log_prc_curve(tmp_dir, y_true_y_pred_y_score, mocker):
    from dvclive.sklearn import metrics

    y_true, _, y_score = y_true_y_pred_y_score

    spy = mocker.spy(metrics, "precision_recall_curve")

    prc = log_precision_recall_curve(
        y_true=y_true, probas_pred=y_score, output_file="prc.json"
    )

    spy.assert_called_once_with(y_true, y_score)

    assert json.dumps(prc, indent=4) == Path("prc.json").read_text()


def test_log_confusion_matrix(tmp_dir, y_true_y_pred_y_score, mocker):
    y_true, y_pred, _ = y_true_y_pred_y_score

    mocker.patch("sklearn.metrics.ConfusionMatrixDisplay.from_predictions")

    mock = log_confusion_matrix(
        y_true=y_true, y_pred=y_pred, output_file="confusion_matrix.png"
    )

    mock.figure_.savefig.assert_called_with("confusion_matrix.png")
