# Support-Vector-Machines-SVM-
Support Vector Machines (SVM) are supervised learning models used for classification and regression tasks. They find the optimal decision boundary (hyperplane) that separates classes with the largest margin. With different kernels, SVMs can handle both linear and non-linear classification problems.
Step-by-Step Procedure
1. Load dataset – Use a binary classification dataset like the Breast Cancer dataset from sklearn.datasets.
2. Split data – Divide the dataset into training and testing sets.
3. Feature scaling – Standardize features for better SVM performance.
4. Train models – Train one SVM with a linear kernel and another with an RBF kernel.
5. Hyperparameter tuning – Experiment with C (regularization) and gamma (RBF kernel spread).
6. Evaluate – Use accuracy scores and cross-validation for model evaluation.
7. Visualization – For 2D data, plot the decision boundaries for both kernels.
