###### DATA SCAPRING ######
#!pip install
#!pip install tqdm

from nba_api.stats.endpoints import playercareerstats as player_stats_endpoint
from nba_api.stats.static import players as nba_players
import pandas as pd
from tqdm import tqdm
from requests.exceptions import Timeout as RequestTimeout

# Retrieve all NBA players
all_nba_players = nba_players.get_players()


# helper functions
def retrieve_player_stats(player_id, timeout=None):
    max_retry_attempts = 3
    retry_attempt = 0
    while retry_attempt < max_retry_attempts:
        try:
            player_stats_data = player_stats_endpoint.PlayerCareerStats(player_id=player_id, timeout=timeout)
            return player_stats_data.get_data_frames()[0]
        except RequestTimeout as e:
            retry_attempt += 1
            print(f"Request timeout occurred. Retrying... (Attempt {retry_attempt}/{max_retry_attempts})")
            continue
    raise Exception("Failed to retrieve data after multiple retries.")

def process_player_stats(all_nba_players, timeout_value, retrieve_player_stats):
    all_filtered_player_stats = []

    for player_info in tqdm(all_nba_players, desc="Processing Players"):
        player_id = player_info['id']
        try:
            player_stats_df = retrieve_player_stats(player_id=player_id, timeout=timeout_value)
            filtered_stats = filter_player_stats(player_stats_df, player_info)
            if filtered_stats:
                all_filtered_player_stats.append(filtered_stats)
        except Exception as exc:
            print(f"Error retrieving data for Player ID {player_id}: {exc}")

    return all_filtered_player_stats

def filter_player_stats(player_stats_df, player_info):
    filtered_stats = player_stats_df[(player_stats_df['SEASON_ID'] >= '2018') & (player_stats_df['SEASON_ID'] <= '2023')]
    if not filtered_stats.empty:
        filtered_stats['PlayerID'] = player_info['id']
        filtered_stats['FullName'] = player_info['full_name']
        return filtered_stats
    return None

# Process player stats
all_filtered_player_stats = process_player_stats(all_nba_players, timeout_value, retrieve_player_stats)

# Combine all filtered player statistics DataFrames into a single DataFrame
combined_filtered_player_stats_df = pd.concat(all_filtered_player_stats, ignore_index=True)

# Define the CSV file details
csv_filename = 'nba_players_career_stats_2018_2023.csv'
csv_file = "/Users/chennyjiang/Desktop/" + csv_filename

# Save the combined DataFrame to a CSV file
combined_filtered_player_stats_df.to_csv(csv_filepath, index=False)

print(f"CSV file saved to: {csv_filepath}")

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


###### BASELINE MODEL ######
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train_rfe, y_train)  # Use the filtered X_train_rfe data

# Predict on the validation set
y_pred_val = rf_regressor.predict(X_val)  # Use the filtered X_val data

# Evaluate the model on the validation set
mse_val = mean_squared_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

print("Mean Squared Error (MSE) on Validation Set:", mse_val)
print("R-squared (R2) on Validation Set:", r2_val)

#plotting
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_val, color='blue', alpha=0.5, label='Predicted')
plt.scatter(y_val, y_val, color='red', alpha=0.5, label='Actual')
plt.title('Actual vs. Predicted Values (Validation Set)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()
