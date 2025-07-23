# Heart-Disease-Decision-Tree
Built a Decision Tree Classifier to predict heart disease (condition) using UCI Heart dataset.

Preprocessing:
Categorical features one-hot encoded (e.g., cp, thal, ca)

Features/labels split: X = df.drop("condition"), y = df["condition"]
Model: DecisionTreeClassifier
Tuned via GridSearchCV
Hyperparameters:
criterion: gini / entropy
max_depth: [3, 5, 10, None]
min_samples_split: [2, 5, 10]
min_samples_leaf: [4, 6, 8]

Best Params Example:
{'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 8, 'min_samples_split': 2}
Accuracy: ~77%
Balanced tree (~25 nodes): avoids overfitting

Tuning Strategy:
Start with wide ranges → narrow around best-performing hyperparameters

Evaluation:
Confusion matrix, precision, recall, f1-score
train_test_split with 20% test set
Tree visualized using plot_tree

Goal: Balanced, interpretable model with good generalization.
