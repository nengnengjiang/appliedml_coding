###########################################################
#   overfitting checker   #
###########################################################

from sklearn.metrics import accuracy_score

# Suppose 'pipeline' is your trained pipeline or model
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")


###########################################################
#  Check on false positives through Confusion matrix #
###########################################################
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

y_test_pred_proba = best_model.predict_proba(X_test)[:, 1] 
y_test_pred = (y_test_pred_proba >= best_threshold).astype(int)

# 2.1 Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred, normalize='true')
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Not Harmful','Harmful'],
            yticklabels=['Not Harmful','Harmful'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 2.2 Classification Report
print("Classification Report:\n", classification_report(y_test, y_test_pred))

###########################################################
#  Check on test prob distribution, is this polarized better model trained or around 0.5 which is unconfident#
###########################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

y_test_scores = best_model.predict_proba(X_test)[:, 1]

plt.figure(figsize=(6,4))
sns.histplot(y_test_scores, kde=True, bins=20)
plt.title("Distribution of Predicted Probabilities (Positive Class)")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.show()

###########################################################
#  find threshold that match requirement for %precision or %recall #
###########################################################

import numpy as np
from sklearn.metrics import precision_recall_curve

# Suppose you have the ground truth y_test and predicted scores y_scores
y_test_scores = best_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_test_scores)

# thresholds array is shorter by 1 element than precision/recall,
# so keep that in mind when indexing.

# 1. Find threshold for recall >= 0.8
recall_requirement = 0.8
valid_idxs = np.where(recall >= recall_requirement)[0]

if len(valid_idxs) > 0:
    # Option A: pick the threshold that yields the highest precision among those valid idxs
    best_idx = valid_idxs[np.argmax(precision[valid_idxs])]
    chosen_threshold = thresholds[best_idx-1]  # shift by -1 if needed for thresholds alignment
    print(f"Threshold that achieves recall >= {recall_requirement}: {chosen_threshold:.3f}")
    print(f"Precision at that threshold: {precision[best_idx]:.3f}")
    print(f"Recall at that threshold:    {recall[best_idx]:.3f}")
else:
    print(f"No threshold found with recall >= {recall_requirement}.")

# 2. Find threshold for precision >= 0.9
precision_requirement = 0.9
valid_idxs = np.where(precision >= precision_requirement)[0]

if len(valid_idxs) > 0:
    # Option B: pick the threshold that yields the highest recall among those valid idxs
    best_idx = valid_idxs[np.argmax(recall[valid_idxs])]
    chosen_threshold = thresholds[best_idx-1]
    print(f"Threshold that achieves precision >= {precision_requirement}: {chosen_threshold:.3f}")
    print(f"Precision at that threshold: {precision[best_idx]:.3f}")
    print(f"Recall at that threshold:    {recall[best_idx]:.3f}")
else:
    print(f"No threshold found with precision >= {precision_requirement}.")


###########################################################
#  check feature importance #
###########################################################

# Extract the classifier (LightGBM) and the preprocessor from the pipeline
lgb_clf = best_model.named_steps['clf']
preproc = best_model.named_steps['preprocessor']

tfidf_step = preproc.named_transformers_['text']
feature_names = tfidf_step.get_feature_names_out()

importances = lgb_clf.feature_importances_
print(len(importances), len(feature_names))  # should match

feat_importances = sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True
)
print("Top 20 features by importance:")
for word, imp in feat_importances[:20]:
    print(f"{word}: {imp}")

###########################################################
#  print out FALSE positives or negatives #
###########################################################
y_test_scores = best_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_pred_proba >= best_threshold).astype(int)
misclassified_idx = np.where((y_test_pred != y_test))[0]

# If you want to see false positives specifically
false_positives_idx = np.where((y_test_pred == 1) & (y_test == 0))[0]


# Print or examine a few examples
for i in false_positives_idx[:5]:
    print(f"Text: {X_test.iloc[i]}")
    print(f"True Label: {y_test.iloc[i]}, Pred Label: {y_test_pred[i]}")
    print("---")

