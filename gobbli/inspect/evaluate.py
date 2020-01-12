from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import altair as alt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from gobbli.util import (
    escape_line_delimited_text,
    pred_prob_to_pred_label,
    truncate_text,
)


@dataclass
class ClassificationError:
    """
    Describes an error in classification.  Reports the original text,
    the true label, and the predicted probability.

    Args:
      X: The original text.
      y_true: The true label.
      y_pred_proba: The model predicted probability for each class.
    """

    X: str
    y_true: str
    y_pred_proba: Dict[str, float]

    @property
    def y_pred(self) -> str:
        """
        Returns:
          The predicted class for this observation.
        """
        return max(self.y_pred_proba, key=lambda k: self.y_pred_proba[k])


MetricFunc = Callable[[Sequence[str], pd.DataFrame], float]
"""
A function used to calculate some metric.  It should accept a sequence of true labels (y_true)
and a dataframe of shape (n_samples, n_classes) containing predicted probabilities; it should
output a real number.
"""


DEFAULT_METRICS: Dict[str, MetricFunc] = {
    "Weighted F1 Score": lambda y_true, y_pred_proba: f1_score(
        y_true, pred_prob_to_pred_label(y_pred_proba), average="weighted"
    ),
    "Weighted Precision Score": lambda y_true, y_pred_proba: precision_score(
        y_true, pred_prob_to_pred_label(y_pred_proba), average="weighted"
    ),
    "Weighted Recall Score": lambda y_true, y_pred_proba: recall_score(
        y_true, pred_prob_to_pred_label(y_pred_proba), average="weighted"
    ),
    "Accuracy": lambda y_true, y_pred_proba: accuracy_score(
        y_true, pred_prob_to_pred_label(y_pred_proba)
    ),
}
"""
The default set of metrics to evaluate classification models with.  Users may want to extend
this.
"""


@dataclass
class ClassificationEvaluation:
    """
    Provides several methods for evaluating the results from a classification problem.

    Args:
      labels: The set of unique labels in the dataset.
      X: The list of texts that were classified.
      y_true: The true labels for the dataset.
      y_pred_proba: A dataframe containing a row for each observation in X and a
        column for each label in the training data.  Cells are predicted probabilities.
    """

    labels: List[str]
    X: List[str]
    y_true: List[str]
    y_pred_proba: pd.DataFrame
    metric_funcs: Optional[Dict[str, Callable[[Sequence, Sequence], float]]] = None

    def __post_init__(self):
        if not len(self.y_true) == self.y_pred_proba.shape[0]:
            raise ValueError(
                "y_true and y_pred_proba must have the same number of observations"
            )

        self.y_pred_series = pd.Series(self.y_pred)
        self.y_true_series = pd.Series(self.y_true)

        self.error_pred_prob = self.y_pred_proba[
            self.y_pred_series != self.y_true_series
        ]

    @property
    def y_pred(self) -> List[str]:
        """
        Returns:
          The predicted class for each observation.
        """
        if len(self.y_pred_proba) == 0:
            return []
        else:
            return pred_prob_to_pred_label(self.y_pred_proba)

    def metrics(self) -> Dict[str, float]:
        """
        Returns:
          A dictionary containing various metrics of model performance on the test dataset.
        """
        metric_funcs = self.metric_funcs
        if metric_funcs is None:
            metric_funcs = DEFAULT_METRICS

        return {
            name: metric_func(self.y_true, self.y_pred_proba)
            for name, metric_func in metric_funcs.items()
        }

    def metrics_report(self) -> str:
        """
        Returns:
          A nicely formatted human-readable report describing metrics of model performance
          on the test dataset.
        """
        metric_string = "\n".join(
            f"{name}: {metric}" for name, metric in self.metrics().items()
        )
        return (
            "Metrics:\n"
            "--------\n"
            f"{metric_string}\n\n"
            "Classification Report:\n"
            "----------------------\n"
            f"{classification_report(self.y_true, self.y_pred)}\n"
        )

    def plot(self) -> alt.Chart:
        """
        Returns:
          An Altair chart visualizing predicted probabilities and true classes to visually identify
          where errors are being made.
        """
        pred_prob_df = self.y_pred_proba.copy()
        pred_prob_df["True Class"] = self.y_true

        plot_df = pred_prob_df.melt(
            id_vars=["True Class"], var_name="Class", value_name="Predicted Probability"
        )
        plot_df["Belongs to Class"] = plot_df["True Class"] == plot_df["Class"]

        charts = []
        # Ideally, we'd use alt.Row to create a faceted chart, but streamlit doesn't
        # respect an Altair chart's height unless it's layered, and you can't layer
        # faceted charts.  So we manually concatenate a bunch of layered charts.
        uniq_cls = plot_df["Class"].unique()
        uniq_cls.sort()
        for cls in uniq_cls:
            charts.append(
                # Layer needed to get streamlit to set chart height
                alt.layer(
                    alt.Chart(
                        plot_df[plot_df["True Class"] == cls], title=cls, height=40
                    )
                    .mark_circle(size=8)
                    .encode(
                        x=alt.X(
                            "Predicted Probability", type="quantitative", title=None
                        ),
                        y=alt.Y(
                            "jitter",
                            type="quantitative",
                            title=None,
                            axis=alt.Axis(
                                values=[0], ticks=True, grid=False, labels=False
                            ),
                            scale=alt.Scale(),
                        ),
                        color=alt.Color("Belongs to Class", type="nominal"),
                    )
                    .transform_calculate(
                        # Generate Gaussian jitter with a Box-Muller transform
                        jitter="sqrt(-2*log(random()))*cos(2*PI*random())/32"
                    )
                    .properties(height=40)
                )
            )
        return alt.vconcat(*charts)

    def errors_for_label(self, label: str, k: int = 10):
        """
        Output the biggest mistakes for the given class by the classifier.

        Args:
          label: The label to return errors for.
          k: The number of results to return for each of false positives and false negatives.

        Returns:
          A 2-tuple.  The first element is a list of the top ``k`` false positives, and the
          second element is a list of the top ``k`` false negatives.
        """
        pred_label = self.y_pred_series == label
        true_label = self.y_true_series == label

        # Order false positives/false negatives by the degree of the error;
        # i.e. we want the false positives with highest predicted probability first
        # and false negatives with lowest predicted probability first
        # Take the top `k` of each
        false_positives = (
            self.error_pred_prob.loc[pred_label & ~true_label]
            .sort_values(by=label, ascending=False)
            .iloc[:k]
        )
        false_negatives = (
            self.error_pred_prob.loc[~pred_label & true_label]
            .sort_values(by=label, ascending=True)
            .iloc[:k]
        )

        def create_classification_errors(
            y_pred_proba: pd.DataFrame,
        ) -> List[ClassificationError]:
            classification_errors = []
            for ndx, row in y_pred_proba.iterrows():
                classification_errors.append(
                    ClassificationError(
                        X=self.X[ndx],
                        y_true=self.y_true[ndx],
                        y_pred_proba=row.to_dict(),
                    )
                )
            return classification_errors

        return (
            create_classification_errors(false_positives),
            create_classification_errors(false_negatives),
        )

    def errors(
        self, k: int = 10
    ) -> Dict[str, Tuple[List[ClassificationError], List[ClassificationError]]]:
        """
        Output the biggest mistakes for each class by the classifier.

        Args:
          k: The number of results to return for each of false positives and false negatives.

        Returns:
          A dictionary whose keys are label names and values are 2-tuples.  The first
          element is a list of the top ``k`` false positives, and the second element is a list
          of the top ``k`` false negatives.
        """
        errors = {}
        for label in self.labels:
            errors[label] = self.errors_for_label(label, k=k)

        return errors

    def errors_report(self, k: int = 10) -> str:
        """
        Args:
          k: The number of results to return for each of false positives and false negatives.

        Returns:
          A nicely-formatted human-readable report describing the biggest mistakes made by
          the classifier for each class.
        """
        errors = self.errors(k=k)
        output = "Errors Report\n" "------------\n\n"

        for label, (false_positives, false_negatives) in errors.items():

            def make_errors_str(errors: List[ClassificationError]) -> str:
                return "\n".join(
                    (
                        f"True Class: {e.y_true}\n"
                        f"Predicted Class: {e.y_pred} (Probability: {e.y_pred_proba[e.y_pred]})\n"
                        f"Text: {truncate_text(escape_line_delimited_text(e.X), 500)}\n"
                    )
                    for e in errors
                )

            false_positives_str = make_errors_str(false_positives)
            if len(false_positives_str) == 0:
                false_positives_str = "None"
            false_negatives_str = make_errors_str(false_negatives)
            if len(false_negatives_str) == 0:
                false_negatives_str = "None"

            output += (
                " -------\n"
                f"| CLASS: {label}\n"
                " -------\n\n"
                "False Positives\n"
                "***************\n\n"
                f"{false_positives_str}\n\n"
                "False Negatives\n"
                "***************\n\n"
                f"{false_negatives_str}\n\n"
            )

        return output
