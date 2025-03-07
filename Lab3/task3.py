import numpy as np

# Load the dataset
data = np.load("credit_score_fairness_data.npy")
# contains 3 columns:
# ["Protected attribute", "True credit worthiness", "Algorithm prediction"]

# Extracting the relevant columns
protected_attribute = data[:, 0]  # Sensitive attribute (e.g., gender, race)
true_credit_worthiness = data[:, 1]  # Actual creditworthiness (ground truth)
predictions = data[:, 2]  # Algorithm's predicted creditworthiness

# Compute confusion matrix elements
TP = np.sum((true_credit_worthiness == 1) & (predictions == 1))  # True Positives
FP = np.sum((true_credit_worthiness == 0) & (predictions == 1))  # False Positives
FN = np.sum((true_credit_worthiness == 1) & (predictions == 0))  # False Negatives
TN = np.sum((true_credit_worthiness == 0) & (predictions == 0))  # True Negatives

# Calculate Equal Opportunity
# Equal Opportunity = True Positive Rate (TPR) = TP / (TP + FN)
# This measures whether qualified applicants (true creditworthy individuals) are treated fairly
tpr = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate Equalized Odds
# Equalized Odds requires both TPR and False Positive Rate (FPR) to be equal across groups
# FPR = FP / (FP + TN), this ensures non-creditworthy individuals are not unfairly approved
fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

# Print results
print("Equal Opportunity (TPR):", tpr)
print("False Positive Rate (FPR) for Equalized Odds:", fpr)

# Fairness Metric Choice:
# - Equal Opportunity ensures that individuals who deserve credit (true positives)
#   have an equal chance of being approved, making it a strong metric for fairness.
# - Equalized Odds is stricter and ensures fairness across both positive and negative cases.
# In credit scoring, Equal Opportunity may be more suitable as it ensures those deserving credit
# get fair access, without unduly harming overall approvals.

print("Most suitable fairness metric: Equal Opportunity, as it ensures fair credit approval for those who truly deserve it.")
