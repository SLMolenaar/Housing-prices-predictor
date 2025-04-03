import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

class XGBoostModel:
    def __init__(self):
        # Load data
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')

        # Store test IDs for submission
        self.test_ids = self.test_data["Id"]

        # Drop ID columns to prevent them from being used as features
        self.train_data.drop(columns=['Id'], inplace=True, errors='ignore')
        self.test_data.drop(columns=['Id'], inplace=True, errors='ignore')

        # Convert categorical features to dummy variables (One-Hot Encoding)
        self.train_data = pd.get_dummies(self.train_data, drop_first=True)
        self.test_data = pd.get_dummies(self.test_data, drop_first=True)

        # Align train and test features (to handle categorical mismatches)
        all_features = set(self.train_data.columns).union(set(self.test_data.columns))
        self.train_data = self.train_data.reindex(columns=all_features, fill_value=0)
        self.test_data = self.test_data.reindex(columns=all_features, fill_value=0)

        # Target variable: Log transformation of 'SalePrice'
        self.y = np.log(self.train_data['SalePrice'])
        self.X = self.train_data.drop('SalePrice', axis=1)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=42)

        # Scaling
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Train XGBoost model with optimal parameters determined by grid search
        self.xgb_regressor = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            gamma=0,
            max_depth=3,
            colsample_bytree=0.5,
            reg_alpha=0.5,
            reg_lambda=0,
            subsample=0.8,
            objective='reg:squarederror',
            random_state=42
        )

        self.xgb_regressor.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=False)
        self.xgb_pred = self.xgb_regressor.predict(self.X_test)

    def plot(self):
        self.xgb_pred = np.exp(self.xgb_regressor.predict(self.X_test))
        self.y_test = np.exp(self.y_test)

        # XGBoost plot
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, self.xgb_pred, alpha=0.5, color="orange", ec='k')
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()], 'k--', lw=2, label="perfect model")
        std_y = np.std(self.y_test)
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min() + std_y, self.y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min() - std_y, self.y_test.max() - std_y], 'r--', lw=1)
        plt.title("XGBoost Predictions vs Actual")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        plt.show()

    def grid_search(self):
        print("Grid Search ...")
        rmse_scorer = make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False)

        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 1],
            'n_estimators': [50, 100, 200],
            'gamma': [0, 0.1, 0.5],
            'subsample': [0.5, 0.8, 1],
            'colsample_bytree': [0.5, 0.8, 1],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }

        grid_search = GridSearchCV(estimator=xgb.XGBRegressor(), param_grid=param_grid, cv=5, scoring=rmse_scorer)
        grid_search.fit(self.X_train, self.y_train)

        print("Best parameters:", grid_search.best_params_)
        print("Best score:", grid_search.best_score_)

    def evaluate(self):
        mse = mean_squared_error(self.y_test, self.xgb_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.xgb_pred))
        mae = np.mean(np.abs(np.exp(self.y_test) - np.exp(self.xgb_pred)))
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"RMSE (log scale): {rmse}")
        print(f"MAE: {mae}")

    def submission(self):
        print("Preparing submission...")

        # Preprocess submission data
        test_data_original = pd.read_csv('test.csv')
        test_features = self.test_data.drop(columns=['Id'], errors='ignore')
        test_features = test_features.reindex(columns=self.X.columns, fill_value=0)
        test_features_scaled = self.scaler.transform(test_features)

        # Make predictions
        predicted_SalePrice = self.xgb_regressor.predict(test_features_scaled)
        predicted_SalePrice = np.exp(predicted_SalePrice)  # Convert back from log scale

        # Restore "Id" column from original test data
        submission_df = pd.DataFrame({
            'Id': test_data_original['Id'],
            'SalePrice': predicted_SalePrice
        })

        # Save to CSV
        submission_df.to_csv('submission.csv', index=False)
        print("Submission saved as 'submission.csv'")

if __name__ == "__main__":
    xgb_model = XGBoostModel()
    xgb_model.evaluate()
    xgb_model.plot()
    xgb_model.grid_search()
    xgb_model.submission()
