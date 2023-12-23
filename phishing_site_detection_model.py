import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
)
import joblib
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml
from skl2onnx.common.data_types import StringTensorType
from skl2onnx import convert_sklearn
import matplotlib.pyplot as plt

PLOT = True


def classification_metrics(y_pred_proba, y_pred, y_val):
    """
    Classification metrics
    """
    y_pred_proba = y_pred_proba
    fpr_roc, tpr_roc, thresholds = roc_curve(y_val, y_pred_proba)

    # Classification report
    metrics = classification_report(y_val, y_pred, output_dict=True)
    print(classification_report(y_val, y_pred))

    # ROC AUC
    roc_auc = auc(fpr_roc, tpr_roc)

    # ROC AUCPR
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    roc_aucpr = auc(recall, precision)
    baseline = len(y_val[y_val == 1]) / len(y_val)

    # TPR/FPR
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)
    fdr = fp / (fp + tp)
    print(
        "FPR",
        "FDR",
        "TPR",
        "AUC",
        "AUCPR =",
        np.round(fpr, 2),
        np.round(fdr, 2),
        np.round(tpr, 2),
        np.round(roc_auc, 2),
        np.round(roc_aucpr, 2),
    )

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    cm = pd.DataFrame(cm)
    print("Confusion Matrix: \n", cm)

    metrics["fpr"] = fpr
    metrics["fdr"] = fdr
    metrics["tpr"] = tpr
    metrics["roc_auc"] = roc_auc
    metrics["roc_aucpr"] = roc_aucpr
    metrics["baseline"] = baseline
    # metrics["confusion_matrix"] = cm.values

    if PLOT == True:
        # ROC
        plt.figure()
        plt.plot(fpr_roc, tpr_roc)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC AUC %f" % roc_auc)
        plt.savefig("metrics/test_roc_curve.png")

        # Precision-Recall
        plt.figure()
        plt.plot(recall, precision)
        plt.axhline(y=baseline, color="gray")
        plt.ylim([0, 1.1])
        plt.xlabel("Recall/TPR")
        plt.ylabel("Precision")
        plt.title("AUCPR %f" % roc_aucpr)
        plt.legend(["PR Curve", "Baseline/Positive Rate"])
        plt.savefig("metrics/test_pr_curve.png")

    return metrics


def model_training_pipeline(df_train):

    # Create the pipeline
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char")),
            #("hashingvectorizer", HashingVectorizer(n_features=500)),
            ("classifier", RandomForestClassifier()),
        ]
    )

    y_train = df_train["result"].values

    pipe.fit(df_train["contents"], y_train)

    return pipe


def pipe_to_pmml(pipe, path):
    """
    Pipe to PMML
    """

    pipeline = make_pmml_pipeline(pipe)
    sklearn2pmml(pipeline, path)

    return None


def pipe_to_pkl(pipe, path):
    """
    Pipe to Pkl
    """

    joblib.dump(pipe, path)

    return None


def pipe_to_onnx(pipe, path):
    """
    Pipe to ONNX (Can't handle tfidf)
    """
    initial_type = [('string_input', StringTensorType())]
    onx = convert_sklearn(pipe, initial_types=initial_type)
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())

    return None
