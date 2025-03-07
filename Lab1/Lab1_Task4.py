import numpy as np
import pandas as pd # pip install pandas
import matplotlib.pyplot as plt
from sklearn.svm import SVC #pip install scikit-learn
from sklearn.ensemble import RandomForestClassifier  # OWN MADE SCRIPT

# Step 1: Load the data
df = pd.read_pickle('Lab1_Task4_data.pkl')  # Load the dataset into a pandas DataFrame

# Step 2: Visualize the data
X = df[['Tissue Texture Score', 'Tissue Density Score']].values  # Extract features for visualization
y = df['Diagnosis']  # Extract target labels

# Step 3: Fitting a model to the training data (SVM classifier)
svm_clf = SVC(kernel='rbf', C=1000)  # Create an instance of the SVM classifier with a linear kernel
svm_clf.fit(X, y)  # Train the SVM classifier using the data

# Get the min and max values for the features to define the grid
x_texture_min, x_texture_max = int(min(df['Tissue Texture Score'])), int(max(df['Tissue Texture Score']))
x_density_min, x_density_max = int(min(df['Tissue Density Score'])), int(max(df['Tissue Density Score']))

# Step 4: Transforming the test data to find decision boundary
xx_texture, xx_density = np.meshgrid(np.linspace(x_texture_min, x_texture_max, num=100),
                                     np.linspace(x_density_min, x_density_max, num=100))  # Create a mesh grid for the decision boundary
xx_decision_boundary = np.stack((xx_texture.flatten(), xx_density.flatten()), axis=1)  # Flatten the grid for prediction
Z_SVM = svm_clf.predict(xx_decision_boundary)  # Predict the class labels for each point in the grid
Z_SVM = Z_SVM.reshape(xx_texture.shape)  # Reshape the prediction result to match the grid shape

# Step 5: Visualizing the decision boundary (SVM)
plt.figure(figsize=(10, 8))

# Subplot 1: SVM decision boundary
plt.subplot(2, 1, 1)  # Create the first subplot (top)
plt.contourf(xx_texture, xx_density, Z_SVM, alpha=0.4)  # Plot the decision boundary for SVM
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='rainbow')  # Plot the data points
plt.title("SVM Decision Boundary")
plt.xlabel("Tissue Texture Score")
plt.ylabel("Tissue Density Score")
plt.tight_layout()

# OWN MADE SCRIPT BELOW: 
# Fit a random forest classifier to the training data
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Create a random forest classifier
rf_clf.fit(X, y)  # Train the random forest classifier using the data

# Predict the decision boundary for the random forest classifier
Z_RF = rf_clf.predict(xx_decision_boundary)  # Get predictions for the grid
Z_RF = Z_RF.reshape(xx_texture.shape)  # Reshape predictions to match the grid

# Subplot 2: Random Forest decision boundary, copy paste from line 32-39
plt.subplot(2, 1, 2)  # Create the second subplot (bottom)
plt.contourf(xx_texture, xx_density, Z_RF, alpha=0.4)  # Plot the decision boundary for Random Forest
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='rainbow')  # Plot the data points
plt.title("Random Forest Decision Boundary")
plt.xlabel("Tissue Texture Score")
plt.ylabel("Tissue Density Score")
plt.tight_layout()

plt.show()  # Display the plots
