###### FEATURE SELECTION ######
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def custom_feature_selection(data_path):
    players_data = pd.read_csv(data_path)
    players_data = players_data.iloc[:, 5:-2]

    # Define target variable (to be predicted)
    target_variable = 'FG3M'

    # Calculate correlations
    correlations = players_data.corr()[target_variable].abs().sort_values(ascending=False)

    # Select top correlated features
    selected_features = correlations.index[1:10]

    # Extract features and target
    X = players_data[selected_features]
    y = players_data[target_variable].values

    # Split data: train, val, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Create model
    model = LinearRegression()

    # RFE
    feature_selector = RFE(model, n_features_to_select=9)
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    selected_features_rfe = X.columns[feature_selector.support_]

    # Print selected features
    print("Top 8 Correlated Features:", selected_features_rfe[1:10])

    return X_train_selected, X_test, y_train, y_test

"""# Training the 1st Iteration of the Model
After finding the top 8 correlated features, we now knew what to target in our model. This was the most complicated portion of our process. We did not know whether we wanted to use a simple pass-forward Neural Network, or use some other sort of Machine Learning Algorithm. Eventually, we compared Random Forest Classification, a simple forward-feeding CNN, and a LSTM-Based RNN. We found that of the 3, the RNN had by far the most efficacy in reducing loss (MSE) and variance (R-Squared) (most likely due to the complexity of the model).
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from copy import deepcopy


# Constants
FILE_PATH = '/content/data_2018_2023.csv'
FEATURES = ['FGA', 'PTS', 'MIN', 'FGM', 'TOV', 'STL', 'GP', 'GS']
TARGET = 'FG3M'
SEED = 42

# Set seed for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load data
def load_data(file_path):
    return pd.read_csv('/content/data_2018_2023.csv')

# Preprocess data
def preprocess_data(data):
    # Feature engineering
    data['PTS'] = data['PTS'] / data['GP']
    data = data.dropna(subset=FEATURES + [TARGET])
    api_data_array = winsorize(data[FEATURES].values, limits=[0.01, 0.01], axis=0)
    data[FEATURES] = pd.DataFrame(api_data_array, columns=FEATURES)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(data[FEATURES])
    poly_feature_names = poly.get_feature_names_out(FEATURES)
    data_poly = pd.DataFrame(X_poly, columns=poly_feature_names)
    power_transformer = PowerTransformer()
    X_power = power_transformer.fit_transform(data[FEATURES])
    data_power = pd.DataFrame(X_power, columns=FEATURES)
    data = pd.concat([data, data_poly, data_power], axis=1)

    # Prepare data
    exclude_columns = [col for col in data.columns if col not in (FEATURES + [TARGET])]
    X = data.drop(columns=exclude_columns)
    y = data[TARGET].values

    # Standardization using Min-Max Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=SEED)

# Create RNN model
def create_rnn_model(input_shape):
    model = keras.Sequential([
        layers.LSTM(64, activation='relu', input_shape=(None, input_shape)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

# Train RNN model
def train_rnn_model(model, X_train, y_train, epochs=500, batch_size=32):
    # Reshape input for sequence modeling
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

    # Train the model
    model.fit(
        X_train_reshaped, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.2
    )

# Evaluate RNN model
def evaluate_rnn_model(model, X_test, y_test):
    # Reshape input for sequence modeling
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    rnn_pred = model.predict(X_test_reshaped)
    rnn_mse = mean_squared_error(y_test, rnn_pred)
    rnn_rmse = np.sqrt(rnn_mse)
    rnn_mae = mean_absolute_error(y_test, rnn_pred)
    rnn_r2 = r2_score(y_test, rnn_pred)
    return rnn_pred, rnn_mse, rnn_rmse, rnn_mae, rnn_r2

# Plot results
def plot_results_rnn(y_true, y_pred, residual):
    # Scatter Plot for RNN Predictions
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred.flatten(), alpha=0.5)
    plt.title('RNN Predictions')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()

    # Residual Plot for RNN
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred.flatten(), residual, alpha=0.5)
    plt.title('Residual Plot - RNN')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.show()

    # Distribution of Predictions for RNN
    plt.figure(figsize=(8, 5))
    plt.hist(y_pred.flatten(), bins=30, alpha=0.5, label='RNN Predictions')
    plt.hist(y_true, bins=30, alpha=0.5, label='True Values')
    plt.title('Distribution of Predictions - RNN')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Main script
if __name__ == "__main__":
    api_data = load_data(FILE_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(api_data)

    # Apply Isolation Forest to identify and potentially remove outliers from the training data
    iso_forest = IsolationForest(contamination=0.05, random_state=SEED)
    outlier_labels = iso_forest.fit_predict(X_train)
    outlier_indices = np.where(outlier_labels == -1)[0]

    # Train the RNN model
    rnn_model_with_outliers = create_rnn_model(X_train.shape[1])
    train_rnn_model(rnn_model_with_outliers, X_train, y_train)

    # Evaluate the model on the test set  
    rnn_pred_with_outliers, rnn_mse_with_outliers, rnn_rmse_with_outliers, rnn_mae_with_outliers, rnn_r2_with_outliers = evaluate_rnn_model(rnn_model_with_outliers, X_test, y_test)

    # Print or use the evaluation metrics for the model 
    print("\nRNN Model Metrics:")
    print("MSE:", rnn_mse_with_outliers)
    print("RMSE:", rnn_rmse_with_outliers)
    print("MAE:", rnn_mae_with_outliers)
    print("R-squared:", rnn_r2_with_outliers)

    # Plot results for the model 
    plot_results_rnn(y_test, rnn_pred_with_outliers.flatten(), y_test - rnn_pred_with_outliers.flatten())

    # RNN Model Summary
    rnn_model_with_outliers.summary()

    # Organize code into functions for different tasks
def isolation_forest_outlier_removal(X_train, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination, random_state=SEED)
    outlier_labels = iso_forest.fit_predict(X_train)
    outlier_indices = np.where(outlier_labels == -1)[0]
    X_train_no_outliers = np.delete(X_train, outlier_indices, axis=0)
    return X_train_no_outliers

# Visualize the LSTM model architecture
def visualize_model_architecture(model):
    from tensorflow.keras.utils import plot_model
    plot_model(model, show_shapes=True, show_layer_names=True)

# Explore other outlier detection techniques
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope

def visualize_outlier_detection(X_train):
    # Standardize the data
    X_train_std = StandardScaler().fit_transform(X_train)

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)

    # Apply Elliptic Envelope for outlier detection
    envelope = EllipticEnvelope(contamination=0.05)
    outliers = envelope.fit_predict(X_train_std)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=outliers, cmap='viridis', alpha=0.8)
    plt.title('Outlier Detection using Elliptic Envelope')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# Visualize the LSTM model architecture
visualize_model_architecture(rnn_model_with_outliers)



# Plot learning curves
def plot_learning_curves(model, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    train_mean = -np.mean(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training error')
    plt.plot(train_sizes, test_mean, label='Validation error')
    plt.title('Learning Curves')
    plt.xlabel('Training Examples')
    plt.ylabel('Negative Mean Squared Error')
    plt.legend()
    plt.show()
