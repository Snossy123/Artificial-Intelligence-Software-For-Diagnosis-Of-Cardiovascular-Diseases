## Confusion Matrix

|                   | True Positive | False Positive |
|-------------------|---------------|----------------|
| output_CAD        | 5861          | 118            |
| output_CHF        | 5110          | 869            |
| output_MI         | 5659          | 320            |
| output_Normal     | 5824          | 155            |

|                   | False Negative | True Negative |
|-------------------|---------------|---------------|
| output_CAD        | 231           | 1762          |
| output_CHF        | 307           | 1686          |
| output_MI         | 940           | 1053          |
| output_Normal     | 176           | 1817          |

Accuracy: 0.79

| Metric            | Value         |
|-------------------|---------------|
| Micro Precision   | 0.81          |
| Micro Recall      | 0.79          |
| Micro F1-score    | 0.80          |
| Macro Precision   | 0.82          |
| Macro Recall      | 0.79          |
| Macro F1-score    | 0.80          |
| Weighted Precision| 0.82          |
| Weighted Recall   | 0.79          |
| Weighted F1-score | 0.80          |

## Classification Report

|                   | Precision | Recall  | F1-score | Support |
|-------------------|-----------|---------|----------|---------|
| output_CAD        | 0.94      | 0.88    | 0.91     | 1993    |
| output_CHF        | 0.66      | 0.85    | 0.74     | 1993    |
| output_MI         | 0.77      | 0.53    | 0.63     | 1993    |
| output_Normal     | 0.92      | 0.91    | 0.92     | 1993    |
|                   |           |         |          |         |
| Micro Avg         | 0.81      | 0.79    | 0.80     | 7972    |
| Macro Avg         | 0.82      | 0.79    | 0.80     | 7972    |
| Weighted Avg      | 0.82      | 0.79    | 0.80     | 7972    |
| Samples Avg       | 0.79      | 0.79    | 0.79     | 7972    |

The confusion matrix and classification report show the results of a classification model that was evaluated on a dataset with four output classes. The confusion matrix displays the number of true positives, false positives, true negatives, and false negatives for each class, while the classification report provides the precision, recall, and F1-score for each class, as well as the support (i.e., the number of samples in each class). The metrics can be used to evaluate the model's accuracy and identify areas for improvement.
