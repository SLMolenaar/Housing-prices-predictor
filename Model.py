import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from scipy.stats import skew
import warnings

warnings.filterwarnings('ignore')


class XGBoostModel:
    def __init__(self):
        # Load data
        self.train_data_raw = pd.read_csv('train.csv')
        self.test_data_raw = pd.read_csv('test.csv')

        # Store test IDs
        self.test_ids = self.test_data_raw["Id"]

        # Store which features to transform
        self.skewed_features = []

        # Preprocess training data
        self.train_data = self.train_data_raw.copy()
        self.train_data.drop(columns=['Id'], inplace=True, errors='ignore')

        # Handle missing values
        self.train_data = self.handle_missing_values(self.train_data)

        # Feature engineering
        self.train_data = self.engineer_features(self.train_data)

        # Determine and apply skewness transformations (store which features)
        self.train_data = self.handle_skewness_train(self.train_data)

        # One-hot encoding
        self.train_data = pd.get_dummies(self.train_data, drop_first=True)

        # Separate target
        self.y = np.log1p(self.train_data['SalePrice'])
        self.X = self.train_data.drop('SalePrice', axis=1)

        # Remove outliers
        self.remove_outliers(z_threshold=3)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Fit scaler on training data
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.X_scaled = self.scaler.transform(self.X)

        # Step 9: Train models
        self.models = {}
        self.train_models()

    def handle_missing_values(self, df):
        """Handle missing values intelligently"""
        # Numeric features, fill with median
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)

        # Categorical features, fill with 'None' or mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                if col in ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                           'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
                           'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:
                    df[col].fillna('None', inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'None', inplace=True)

        return df

    def engineer_features(self, df):
        """Create new features that capture domain knowledge"""
        # Total square footage
        if 'TotalBsmtSF' in df.columns and '1stFlrSF' in df.columns:
            df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df.get('2ndFlrSF', 0)

        # Total bathrooms
        df['TotalBath'] = (df.get('FullBath', 0) +
                           0.5 * df.get('HalfBath', 0) +
                           df.get('BsmtFullBath', 0) +
                           0.5 * df.get('BsmtHalfBath', 0))

        # Total porch area
        porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        df['TotalPorchSF'] = sum(df.get(col, 0) for col in porch_cols)

        # Age features
        if 'YearBuilt' in df.columns:
            df['HouseAge'] = 2024 - df['YearBuilt']
            df['IsNew'] = (df['HouseAge'] < 5).astype(int)

        if 'YearRemodAdd' in df.columns:
            df['RemodAge'] = 2024 - df['YearRemodAdd']
            df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)

        # Quality interactions
        if 'OverallQual' in df.columns:
            if 'TotalSF' in df.columns:
                df['QualSF'] = df['OverallQual'] * df['TotalSF']
            if 'GrLivArea' in df.columns:
                df['QualGrLivArea'] = df['OverallQual'] * df['GrLivArea']

        # Garage features
        if 'GarageArea' in df.columns and 'GarageCars' in df.columns:
            df['HasGarage'] = (df['GarageArea'] > 0).astype(int)

        # Basement features
        if 'TotalBsmtSF' in df.columns:
            df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)

        # Pool features
        if 'PoolArea' in df.columns:
            df['HasPool'] = (df['PoolArea'] > 0).astype(int)

        # Lot features
        if 'LotArea' in df.columns and 'LotFrontage' in df.columns:
            df['LotAreaPerFrontage'] = df['LotArea'] / (df['LotFrontage'] + 1)

        return df

    def handle_skewness_train(self, df, threshold=0.5):
        """Determine which features are skewed and transform them (TRAINING ONLY)"""
        numeric_feats = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_feats = [f for f in numeric_feats if f != 'SalePrice']

        self.skewed_features = []

        for feat in numeric_feats:
            if df[feat].min() >= 0:  # Only for non-negative features
                skewness = skew(df[feat].dropna())
                if abs(skewness) > threshold:
                    self.skewed_features.append(feat)
                    df[feat] = np.log1p(df[feat])

        print(f"Applied log transformation to {len(self.skewed_features)} skewed features")
        return df

    def handle_skewness_test(self, df):
        """Apply the same skewness transformations determined from training data"""
        for feat in self.skewed_features:
            if feat in df.columns:
                df[feat] = np.log1p(df[feat])
        return df

    def remove_outliers(self, z_threshold=3):
        """Remove extreme outliers from training data"""
        z_scores = np.abs((self.y - self.y.mean()) / self.y.std())

        if 'GrLivArea' in self.X.columns:
            condition = ~((self.X['GrLivArea'] > np.log1p(4000)) & (self.y < np.log1p(300000)))
            self.X = self.X[condition & (z_scores < z_threshold)]
            self.y = self.y[condition & (z_scores < z_threshold)]
        else:
            self.X = self.X[z_scores < z_threshold]
            self.y = self.y[z_scores < z_threshold]

        print(f"Removed {len(z_scores) - len(self.X)} outliers from training data")

    def preprocess_test_data(self):
        """Preprocess test data using the SAME pipeline as training"""
        # Start with raw test data
        test_data = self.test_data_raw.copy()
        test_data.drop(columns=['Id'], inplace=True, errors='ignore')

        # Apply the same pipeline
        test_data = self.handle_missing_values(test_data)
        test_data = self.engineer_features(test_data)
        test_data = self.handle_skewness_test(test_data)
        test_data = pd.get_dummies(test_data, drop_first=True)

        # Align columns with training data
        test_data = test_data.reindex(columns=self.X.columns, fill_value=0)

        return test_data

    def train_models(self):
        """Train multiple models for ensemble"""
        print("Training models...")

        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            gamma=0,
            max_depth=4,
            min_child_weight=3,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1,
            subsample=0.8,
            objective='reg:squarederror',
            random_state=42
        )

        self.models['gbm'] = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )

        self.models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

        self.models['ridge'] = Ridge(alpha=15)

        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train_scaled, self.y_train)

            train_pred = model.predict(self.X_train_scaled)
            test_pred = model.predict(self.X_test_scaled)

            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))

            print(f"{name} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

    def cross_validate(self, model_name='xgb', cv=5):
        """Perform cross-validation on specified model"""
        print(f"\nPerforming {cv}-fold cross-validation on {model_name}...")

        model = self.models[model_name]
        cv_scores = cross_val_score(
            model,
            self.X_scaled,
            self.y,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )

        print(f"CV RMSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        return -cv_scores.mean()

    def plot(self):
        """Plot predictions vs actual for ensemble model"""
        predictions = self.get_ensemble_predictions(self.X_test_scaled)

        y_pred = np.expm1(predictions)
        y_true = np.expm1(self.y_test)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, color="orange", ec='k', s=50)

        plt.plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()], 'k--', lw=2, label="Perfect model")

        std_y = np.std(y_true)
        plt.plot([y_true.min(), y_true.max()],
                 [y_true.min() + std_y, y_true.max() + std_y], 'r--', lw=1, alpha=0.7, label="+/- 1 Std Dev")
        plt.plot([y_true.min(), y_true.max()],
                 [y_true.min() - std_y, y_true.max() - std_y], 'r--', lw=1, alpha=0.7)

        plt.title(f"Ensemble Model Predictions vs Actual\nRMSE: ${rmse:,.0f} | MAE: ${mae:,.0f}", fontsize=14)
        plt.xlabel("Actual Values ($)", fontsize=12)
        plt.ylabel("Predicted Values ($)", fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('predictions_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'predictions_plot.png'")
        plt.show()

    def plot_individual_models(self):
        """Plot predictions for each individual model"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (name, model) in enumerate(self.models.items()):
            pred = model.predict(self.X_test_scaled)
            y_pred = np.expm1(pred)
            y_true = np.expm1(self.y_test)

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            axes[idx].scatter(y_true, y_pred, alpha=0.5, s=30)
            axes[idx].plot([y_true.min(), y_true.max()],
                           [y_true.min(), y_true.max()], 'k--', lw=2)
            axes[idx].set_title(f"{name.upper()} - RMSE: ${rmse:,.0f}")
            axes[idx].set_xlabel("Actual Values ($)")
            axes[idx].set_ylabel("Predicted Values ($)")
            axes[idx].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('individual_models_plot.png', dpi=300, bbox_inches='tight')
        print("Individual models plot saved as 'individual_models_plot.png'")
        plt.show()

    def randomized_search(self, n_iter=30, cv=5):
        """Perform randomized hyperparameter search for XGBoost"""
        print(f"\nPerforming Randomized Search with {n_iter} iterations...")

        param_distributions = {
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
            'n_estimators': [100, 200, 300, 500],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1, 2],
            'reg_lambda': [0, 0.5, 1, 2, 3]
        }

        random_search = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        random_search.fit(self.X_scaled, self.y)

        print("\nBest parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV RMSE: {-random_search.best_score_:.4f}")

        return random_search.best_params_

    def get_ensemble_predictions(self, X):
        """Get weighted ensemble predictions"""
        weights = {
            'xgb': 0.40,
            'gbm': 0.30,
            'rf': 0.20,
            'ridge': 0.10
        }

        predictions = np.zeros(len(X))
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions += weights[name] * pred

        return predictions

    def evaluate(self):
        """Evaluate all models and ensemble"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        print("\nIndividual Model Performance:")
        print("-" * 60)
        for name, model in self.models.items():
            pred = model.predict(self.X_test_scaled)

            rmse_log = np.sqrt(mean_squared_error(self.y_test, pred))

            y_pred = np.expm1(pred)
            y_true = np.expm1(self.y_test)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

            print(f"{name.upper():8} - RMSE (log): {rmse_log:.4f} | RMSE: ${rmse:>10,.0f} | MAE: ${mae:>10,.0f}")

        print("\nEnsemble Model Performance:")
        print("-" * 60)
        ensemble_pred = self.get_ensemble_predictions(self.X_test_scaled)

        rmse_log = np.sqrt(mean_squared_error(self.y_test, ensemble_pred))

        y_pred_ensemble = np.expm1(ensemble_pred)
        y_true = np.expm1(self.y_test)
        rmse_ensemble = np.sqrt(mean_squared_error(y_true, y_pred_ensemble))
        mae_ensemble = mean_absolute_error(y_true, y_pred_ensemble)

        print(f"RMSE (log): {rmse_log:.4f} | RMSE: ${rmse_ensemble:>10,.0f} | MAE: ${mae_ensemble:>10,.0f}")
        print("=" * 60)

    def submission(self, use_ensemble=True):
        """Generate submission file with proper preprocessing"""
        print("\nPreparing submission...")

        # Preprocess test data through the pipeline
        test_data_processed = self.preprocess_test_data()
        test_features_scaled = self.scaler.transform(test_data_processed)

        # Make predictions
        if use_ensemble:
            predicted_log = self.get_ensemble_predictions(test_features_scaled)
            print("Using ensemble predictions")
        else:
            predicted_log = self.models['xgb'].predict(test_features_scaled)
            print("Using XGBoost predictions")

        # Convert back from log scale
        predicted_SalePrice = np.expm1(predicted_log)

        # Sanity check
        print(f"\nPrediction Stats:")
        print(f"   Min:    ${predicted_SalePrice.min():>12,.0f}")
        print(f"   Max:    ${predicted_SalePrice.max():>12,.0f}")
        print(f"   Mean:   ${predicted_SalePrice.mean():>12,.0f}")
        print(f"   Median: ${np.median(predicted_SalePrice):>12,.0f}")

        # Create submission
        submission_df = pd.DataFrame({
            'Id': self.test_ids,
            'SalePrice': predicted_SalePrice
        })

        submission_df.to_csv('submission.csv', index=False)
        print(f"\nSubmission saved as 'submission.csv'")

    def feature_importance(self, top_n=20):
        """Display feature importance from XGBoost model"""
        xgb_model = self.models['xgb']

        importance_dict = xgb_model.get_booster().get_score(importance_type='gain')

        importance_df = pd.DataFrame({
            'Feature': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        }).sort_values('Importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['Importance'], align='center')
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Importance (Gain)')
        plt.title(f'Top {top_n} Most Important Features (XGBoost)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\nFeature importance plot saved as 'feature_importance.png'")
        plt.show()

        return importance_df


if __name__ == "__main__":
    # Initialize and train model
    xgb_model = XGBoostModel()

    # Evaluate models
    xgb_model.evaluate()

    # Cross-validation
    xgb_model.cross_validate(model_name='xgb', cv=5)

    # Plot results
    xgb_model.plot()
    xgb_model.plot_individual_models()

    # Feature importance
    xgb_model.feature_importance(top_n=20)

    # Optional: Hyperparameter tuning
    #best_params = xgb_model.randomized_search(n_iter=30, cv=5)

    # Generate submission
    xgb_model.submission(use_ensemble=True)