from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

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
    as_multiclass,
    as_multilabel,
    escape_line_delimited_text,
    is_multilabel,
    multilabel_to_indicator_df,
    pred_prob_to_pred_label,
    pred_prob_to_pred_multilabel,
    truncate_text,
)


@dataclass
class ClassificationError:
    """
    Describes an error in classification.  Reports the original text,
    the true label, and the predicted probability.

    Args:
      X: The original text.
      y_true: The true label(s).
      y_pred_proba: The model predicted probability for each class.
    """

    X: str
    y_true: Union[str, List[str]]
    y_pred_proba: Dict[str, float]

    @property
    def y_pred(self) -> str:
        """
        Returns:
          The class with the highest predicted probability for this observation.
        """
        return max(self.y_pred_proba, key=lambda k: self.y_pred_proba[k])

    def y_pred_multilabel(self, threshold: float = 0.5) -> List[str]:
        """
        Args:
          threshold: The predicted probability threshold for predictions

        Returns:
          The predicted labels for this observation (predicted probability greater than
          the given threshold)
        """
        return pred_prob_to_pred_multilabel(self.y_pred_proba, threshold)


MetricFunc = Callable[[Sequence[str], pd.DataFrame], float]
"""
A function used to calculate some metric.  It should accept a sequence of true labels (y_true)
and a dataframe of shape (n_samples, n_classes) containing predicted probabilities; it should
output a real number.
"""


DEFAULT_METRICS: Dict[str, MetricFunc] = {
    "Weighted F1 Score": lambda y_true, y_pred: f1_score(
        y_true, y_pred, average="weighted"
    ),
    "Weighted Precision Score": lambda y_true, y_pred: precision_score(
        y_true, y_pred, average="weighted"
    ),
    "Weighted Recall Score": lambda y_true, y_pred: recall_score(
        y_true, y_pred, average="weighted"
    ),
    "Accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
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
    y_true: Union[List[str], List[List[str]]]
    y_pred_proba: pd.DataFrame
    metric_funcs: Optional[Dict[str, Callable[[Sequence, Sequence], float]]] = None

    def __post_init__(self):
        if not len(self.y_true) == self.y_pred_proba.shape[0]:
            raise ValueError(
                "y_true and y_pred_proba must have the same number of observations"
            )

        self.multilabel = is_multilabel(self.y_true)

    @property
    def y_true_multiclass(self) -> List[str]:
        return as_multiclass(self.y_true, self.multilabel)

    @property
    def y_true_multilabel(self) -> pd.DataFrame:
        return multilabel_to_indicator_df(
            as_multilabel(self.y_true, self.multilabel), self.labels
        )

    @property
    def y_pred_multiclass(self) -> List[str]:
        """
        Returns:
          Predicted class for each observation (assuming multiclass context).
        """
        return pred_prob_to_pred_label(self.y_pred_proba)

    @property
    def y_pred_multilabel(self) -> pd.DataFrame:
        """
        Returns:
          Indicator dataframe containing a 0 if each label wasn't predicted and 1 if
          it was for each observation.
        """
        return pred_prob_to_pred_multilabel(self.y_pred_proba).astype("int")

    def metrics(self) -> Dict[str, float]:
        """
        Returns:
          A dictionary containing various metrics of model performance on the test dataset.
        """
        metric_funcs = self.metric_funcs
        if metric_funcs is None:
            metric_funcs = DEFAULT_METRICS

        if self.multilabel:
            y_true: Union[List[str], pd.DataFrame] = self.y_true_multilabel
            y_pred: Union[List[str], pd.DataFrame] = self.y_pred_multilabel
        else:
            y_true = self.y_true_multiclass
            y_pred = self.y_pred_multiclass

        return {
            name: metric_func(y_true, y_pred)
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

        if self.multilabel:
            y_true: Union[pd.DataFrame, List[str]] = self.y_true_multilabel
            y_pred: Union[pd.DataFrame, List[str]] = self.y_pred_multilabel
            # Since these are indicator dataframes, the "labels" are indices
            labels: Union[List[str], List[int]] = list(range(len(self.labels)))
        else:
            y_true = self.y_true_multiclass
            y_pred = self.y_pred_multiclass
            # Since these are lists of labels, the "labels" are the strings themselves
            labels = self.labels

        return (
            "Metrics:\n"
            "--------\n"
            f"{metric_string}\n\n"
            "Classification Report:\n"
            "----------------------\n"
            f"{classification_report(y_true, y_pred, labels=labels, target_names=self.labels)}\n"
        )

    def plot(self, sample_size: Optional[int] = None) -> alt.Chart:
        """
        Args:
          sample_size: Optional number of points to sample for the plot.  Unsampled
            plots may be difficult to save due to their size.

        Returns:
          An Altair chart visualizing predicted probabilities and true classes to visually identify
          where errors are being made.
        """
        # Since multilabel is a generalization of the multiclass paradigm, implement
        # this visualization the same for multiclass and multilabel using the multilabel
        # format
        pred_prob_df = self.y_pred_proba
        true_df = self.y_true_multilabel

        if sample_size is not None:
            # Avoid errors due to sample being larger than the population if the number
            # of observations is smaller than the sample size
            pred_prob_df = pred_prob_df.sample(
                n=min(sample_size, pred_prob_df.shape[0])
            )
            true_df = true_df.iloc[pred_prob_df.index]

        charts = []

        if self.multilabel:
            legend_label = "Has Label"
        else:
            legend_label = "Belongs to Class"

        for label in self.labels:
            # Plot the predicted probabilities for given label for all observations
            plot_df = (
                pred_prob_df[[label]]
                .rename({label: "Predicted Probability"}, axis="columns")
                .join(
                    true_df[[label]]
                    .astype("bool")
                    .rename({label: legend_label}, axis="columns")
                )
            )

            charts.append(
                alt.layer(
                    alt.Chart(plot_df, title=label, height=40)
                    .mark_circle(size=8)
                    .encode(
                        x=alt.X(
                            "Predicted Probability",
                            type="quantitative",
                            title=None,
                            scale=alt.Scale(domain=(0.0, 1.0)),
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
                        color=alt.Color(legend_label, type="nominal"),
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
        Output the biggest mistakes for the given class by the classifier

        Args:
          label: The label to return errors for.
          k: The number of results to return for each of false positives and false negatives.

        Returns:
          A 2-tuple.  The first element is a list of the top ``k`` false positives, and the
          second element is a list of the top ``k`` false negatives.
        """
        pred_label = self.y_pred_multilabel[label].astype("bool")
        true_label = self.y_true_multilabel[label].astype("bool")

        # Order false positives/false negatives by the degree of the error;
        # i.e. we want the false positives with highest predicted probability first
        # and false negatives with lowest predicted probability first
        # Take the top `k` of each
        false_positives = (
            self.y_pred_proba.loc[pred_label & ~true_label]
            .sort_values(by=label, ascending=False)
            .iloc[:k]
        )
        false_negatives = (
            self.y_pred_proba.loc[~pred_label & true_label]
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
                if self.multilabel:
                    return "\n".join(
                        (
                            f"Correct Value: {label in e.y_true}\n"
                            f"Predicted Probability: {e.y_pred_proba[label]}"
                            f"Text: {truncate_text(escape_line_delimited_text(e.X), 500)}\n"
                        )
                        for e in errors
                    )
                else:
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

            header_name = "CLASS" if self.multilabel else "LABEL"

            output += (
                " -------\n"
                f"| {header_name}: {label}\n"
                " -------\n\n"
                "False Positives\n"
                "***************\n\n"
                f"{false_positives_str}\n\n"
                "False Negatives\n"
                "***************\n\n"
                f"{false_negatives_str}\n\n"
            )

        return output
