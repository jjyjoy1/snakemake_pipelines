import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
import os
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn import linear_model, ensemble, svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, mean_squared_error, r2_score, 
    mean_absolute_error, explained_variance_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy import stats
import joblib

# Import XGBoost conditionally
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. XGBoost models unavailable. Install with 'pip install xgboost'.")

# Import TensorFlow conditionally
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, Dropout, BatchNormalization, Activation, Input, 
        Conv1D, MaxPooling1D, Flatten, LSTM, Bidirectional, 
        MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
    )
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not installed. Deep learning models unavailable. Install with 'pip install tensorflow'.")


class BioinformaticsSupervisedLearner:
    """
    Step 5: Classification & Regression (Supervised Learning)
    
    A class for supervised learning tasks in bioinformatics, supporting:
    - Classification: predicting categorical outcomes, phenotypes, or conditions
    - Regression: predicting continuous biological variables or measurements
    
    This class implements various machine learning algorithms:
    - Classic ML methods: Logistic Regression, Random Forest, SVM, Gradient Boosting, XGBoost
    - Deep Learning: Fully Connected Networks, CNNs, RNNs/LSTM, Transformer-based models
    
    Features include:
    - Model training and evaluation
    - Hyperparameter optimization
    - Cross-validation
    - Performance metrics and visualization
    - Feature importance analysis
    - Model interpretation
    - Model saving and loading
    """
    
    def __init__(self, task_type='classification', model_type='random_forest', random_state=42, logger=None):
        """
        Initialize the supervised learner.
        
        Parameters:
        -----------
        task_type : str
            Type of supervised learning task ('classification' or 'regression')
        model_type : str
            Type of model to use:
            - 'logistic_regression': Logistic Regression (classification only)
            - 'linear_regression': Linear Regression (regression only)
            - 'random_forest': Random Forest
            - 'svm': Support Vector Machine
            - 'gradient_boosting': Gradient Boosting
            - 'xgboost': XGBoost
            - 'mlp': Multi-layer Perceptron (fully connected neural network)
            - 'cnn': Convolutional Neural Network
            - 'rnn': Recurrent Neural Network (LSTM)
            - 'transformer': Transformer-based model
        random_state : int
            Random seed for reproducibility
        logger : logging.Logger
            Logger for tracking the training process
        """
        self.task_type = task_type.lower()
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = None
        self.estimator = None
        self.feature_names_ = None
        self.classes_ = None
        self.label_encoder = None
        self.is_fitted = False
        self.feature_importances_ = None
        self.best_params_ = None
        self.cv_results_ = None
        self.metrics_ = {}
        self.history_ = None  # For deep learning models
        
        # Set up logger
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
            
        # Validate task_type
        valid_tasks = ['classification', 'regression']
        if self.task_type not in valid_tasks:
            raise ValueError(f"Invalid task type '{task_type}'. Choose from: {', '.join(valid_tasks)}")
        
        # Validate model_type and check dependencies
        self._validate_model_type()
    
    def _setup_logger(self):
        """Setup a basic logger if none is provided."""
        logger = logging.getLogger("BioinformaticsSupervisedLearner")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def _validate_model_type(self):
        """Validate the model type and check dependencies."""
        # Define valid model types based on task
        if self.task_type == 'classification':
            valid_ml_models = [
                'logistic_regression', 'random_forest', 'svm', 
                'gradient_boosting', 'xgboost'
            ]
        else:  # regression
            valid_ml_models = [
                'linear_regression', 'random_forest', 'svm', 
                'gradient_boosting', 'xgboost'
            ]
            
        valid_dl_models = ['mlp', 'cnn', 'rnn', 'transformer']
        
        # Validate model type
        valid_models = valid_ml_models + valid_dl_models
        if self.model_type not in valid_models:
            raise ValueError(
                f"Invalid model type '{self.model_type}' for {self.task_type} task. "
                f"Choose from: {', '.join(valid_models)}"
            )
        
        # Check dependencies for specific models
        if self.model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with 'pip install xgboost'")
            
        if self.model_type in valid_dl_models and not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not installed. Install with 'pip install tensorflow'")
    
    def fit(self, X, y, **kwargs):
        """
        Fit the supervised learning model to the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        **kwargs : 
            Additional parameters:
            - hyperparameter_tuning: bool, whether to perform hyperparameter tuning
            - param_grid: dict or list of dicts, grid of parameters to search
            - cv: int or cv splitter, cross-validation strategy
            - scoring: str, metric to optimize in hyperparameter tuning
            - n_jobs: int, number of parallel jobs
            - verbose: int, verbosity level
            - Additional model-specific parameters
            
        Returns:
        --------
        self : BioinformaticsSupervisedLearner
            The fitted learner
        """
        # Process input data
        X_processed, y_processed = self._preprocess_data(X, y)
        
        self.logger.info(f"Starting {self.task_type} with model: {self.model_type}")
        
        # Extract general parameters
        hyperparameter_tuning = kwargs.pop('hyperparameter_tuning', False)
        param_grid = kwargs.pop('param_grid', None)
        cv = kwargs.pop('cv', 5)
        scoring = kwargs.pop('scoring', None)
        n_jobs = kwargs.pop('n_jobs', -1)
        verbose = kwargs.pop('verbose', 1)
        
        # Create the model
        self._initialize_model(**kwargs)
        
        # Perform hyperparameter tuning if requested
        if hyperparameter_tuning and param_grid is not None:
            self._perform_hyperparameter_tuning(
                X_processed, y_processed, param_grid, cv, scoring, n_jobs, verbose
            )
        else:
            # Fit the model directly
            if self.model_type in ['mlp', 'cnn', 'rnn', 'transformer']:
                # For deep learning models, extract training parameters
                batch_size = kwargs.get('batch_size', 32)
                epochs = kwargs.get('epochs', 100)
                validation_split = kwargs.get('validation_split', 0.2)
                callbacks_list = self._get_dl_callbacks(**kwargs)
                
                # Fit the deep learning model
                self.history_ = self.model.fit(
                    X_processed, y_processed,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=validation_split,
                    callbacks=callbacks_list,
                    verbose=verbose
                )
            else:
                # Fit standard ML model
                self.model.fit(X_processed, y_processed)
        
        # Calculate feature importances if applicable
        self._calculate_feature_importances()
        
        self.is_fitted = True
        self.logger.info(f"{self.model_type.capitalize()} model training completed")
        
        return self
    
    def predict(self, X):
        """
        Predict using the trained model.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
            
        Returns:
        --------
        numpy.ndarray
            Predicted values (class labels for classification or 
            continuous values for regression)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Preprocess input data
        X_processed = self._preprocess_input(X)
        
        # Make predictions
        if self.model_type in ['mlp', 'cnn', 'rnn', 'transformer'] and self.task_type == 'classification':
            # For deep learning classification models, get class predictions from probabilities
            y_pred_prob = self.model.predict(X_processed)
            
            if y_pred_prob.shape[1] > 1:  # Multi-class
                y_pred = np.argmax(y_pred_prob, axis=1)
            else:  # Binary
                y_pred = (y_pred_prob > 0.5).astype(int).flatten()
                
            # Decode labels if needed
            if self.label_encoder is not None:
                y_pred = self.label_encoder.inverse_transform(y_pred)
                
            return y_pred
        else:
            # Standard prediction for ML models and DL regression
            return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for classification tasks.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
            
        Returns:
        --------
        numpy.ndarray
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
            
        # Preprocess input data
        X_processed = self._preprocess_input(X)
        
        # Get probability predictions
        if self.model_type in ['mlp', 'cnn', 'rnn', 'transformer']:
            return self.model.predict(X_processed)
        elif hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_processed)
        else:
            raise ValueError(f"{self.model_type} does not support probability predictions")
    
    def evaluate(self, X, y, **kwargs):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The true target values
        **kwargs : 
            Additional parameters for evaluation
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Process input data
        X_processed = self._preprocess_input(X)
        
        # Process target data
        if self.task_type == 'classification':
            if self.label_encoder is not None:
                if isinstance(y, pd.Series):
                    y_true = self.label_encoder.transform(y.values)
                else:
                    y_true = self.label_encoder.transform(y)
            else:
                y_true = y
                
            # Make predictions
            y_pred = self.predict(X)
            
            # Make probability predictions if possible
            try:
                y_pred_proba = self.predict_proba(X)
                has_proba = True
            except:
                has_proba = False
            
            # Calculate classification metrics
            metrics = {}
            
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Handle different classification scenarios
            if len(np.unique(y_true)) > 2:  # Multi-class
                # Use weighted averaging for multi-class
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
                metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
                
                # Calculate AUC using one-vs-rest if probabilities are available
                if has_proba:
                    if y_pred_proba.shape[1] > 2:  # true multi-class
                        try:
                            metrics['roc_auc'] = roc_auc_score(
                                to_categorical(y_true), y_pred_proba, multi_class='ovr'
                            )
                        except:
                            metrics['roc_auc'] = np.nan
                    else:  # binary problem encoded as multi-class
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:  # Binary classification
                metrics['precision'] = precision_score(y_true, y_pred)
                metrics['recall'] = recall_score(y_true, y_pred)
                metrics['f1'] = f1_score(y_true, y_pred)
                
                # Calculate AUC if probabilities are available
                if has_proba:
                    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                        # Use second column for positive class
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        # Use the only column
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # Confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
            
            # Store prediction data for later visualization
            self._evaluation_data = {
                'y_true': y_true,
                'y_pred': y_pred,
                'has_proba': has_proba
            }
            
            if has_proba:
                self._evaluation_data['y_pred_proba'] = y_pred_proba
            
        else:  # Regression
            # Make predictions
            y_pred = self.predict(X)
            
            # Calculate regression metrics
            metrics = {}
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y, y_pred)
            metrics['r2'] = r2_score(y, y_pred)
            metrics['explained_variance'] = explained_variance_score(y, y_pred)
            
            # Add correlation coefficient
            metrics['pearson_r'], metrics['pearson_p'] = stats.pearsonr(y, y_pred)
            metrics['spearman_r'], metrics['spearman_p'] = stats.spearmanr(y, y_pred)
            
            # Store prediction data for later visualization
            self._evaluation_data = {
                'y_true': y,
                'y_pred': y_pred
            }
        
        # Log results
        self.logger.info(f"Evaluation metrics: {metrics}")
        
        # Store metrics
        self.metrics_ = metrics
        
        return metrics
    
    def cross_validate(self, X, y, cv=5, scoring=None, **kwargs):
        """
        Perform cross-validation to estimate model performance.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        cv : int or cross-validation generator
            Cross-validation strategy
        scoring : str or list/dict of str
            Metrics to evaluate
        **kwargs : 
            Additional parameters for cross-validation
            
        Returns:
        --------
        dict
            Dictionary of cross-validation results
        """
        # Process input data
        X_processed, y_processed = self._preprocess_data(X, y)
        
        self.logger.info(f"Performing {cv}-fold cross-validation")
        
        # Set default scoring based on task
        if scoring is None:
            if self.task_type == 'classification':
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'
        
        # For deep learning models, use custom CV implementation
        if self.model_type in ['mlp', 'cnn', 'rnn', 'transformer']:
            cv_results = self._deep_learning_cross_validate(
                X_processed, y_processed, cv, scoring, **kwargs
            )
        else:
            # Use sklearn's cross_val_score for traditional models
            cv_scores = cross_val_score(
                self.model, 
                X_processed, 
                y_processed, 
                cv=cv, 
                scoring=scoring
            )
            
            cv_results = {
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'cv_scores': cv_scores
            }
        
        # Log results
        self.logger.info(f"Cross-validation mean score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        
        # Store results
        self.cv_results_ = cv_results
        
        return cv_results
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Different saving approaches based on model type
        if self.model_type in ['mlp', 'cnn', 'rnn', 'transformer']:
            # Save deep learning model using Keras
            model_path = f"{filepath}_model.h5"
            self.model.save(model_path)
            
            # Save metadata separately using pickle
            metadata = {
                'task_type': self.task_type,
                'model_type': self.model_type,
                'feature_names': self.feature_names_,
                'classes': self.classes_,
                'label_encoder': self.label_encoder,
                'metrics': self.metrics_,
                'feature_importances': self.feature_importances_,
                'best_params': self.best_params_
            }
            
            with open(f"{filepath}_metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
                
            self.logger.info(f"Deep learning model saved to {model_path} with metadata")
        else:
            # Save traditional ML model using joblib
            joblib.dump(self, filepath)
            self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        BioinformaticsSupervisedLearner
            The loaded model
        """
        # Check if it's a deep learning model (has metadata file)
        metadata_path = f"{filepath}_metadata.pkl"
        if os.path.exists(metadata_path):
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                
            # Load Keras model
            model_path = f"{filepath}_model.h5"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found")
                
            # Create an empty instance with appropriate parameters
            instance = cls(
                task_type=metadata['task_type'],
                model_type=metadata['model_type']
            )
            
            # Load the Keras model
            instance.model = tf.keras.models.load_model(model_path)
            
            # Restore metadata
            instance.feature_names_ = metadata['feature_names']
            instance.classes_ = metadata['classes']
            instance.label_encoder = metadata['label_encoder']
            instance.metrics_ = metadata['metrics']
            instance.feature_importances_ = metadata['feature_importances']
            instance.best_params_ = metadata['best_params']
            instance.is_fitted = True
            
            return instance
        else:
            # Load scikit-learn based model using joblib
            return joblib.load(filepath)
    
    def _preprocess_data(self, X, y):
        """
        Preprocess input data and target variable.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
            
        Returns:
        --------
        tuple
            (X_processed, y_processed)
        """
        # Process feature matrix
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_processed = X.values
        else:
            X_processed = X
            if self.feature_names_ is None:
                self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Process target variable
        if self.task_type == 'classification':
            # Encode class labels for classification
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_processed = self.label_encoder.fit_transform(y)
                self.classes_ = self.label_encoder.classes_
            else:
                y_processed = self.label_encoder.transform(y)
                
            # For deep learning classification, convert to categorical if needed
            if self.model_type in ['mlp', 'cnn', 'rnn', 'transformer']:
                if len(self.classes_) > 2:  # Multi-class
                    y_processed = to_categorical(y_processed)
        else:
            # For regression, ensure y is a numpy array
            if isinstance(y, pd.Series):
                y_processed = y.values
            else:
                y_processed = y
                
        return X_processed, y_processed
    
    def _preprocess_input(self, X):
        """
        Preprocess input data for prediction.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
            
        Returns:
        --------
        numpy.ndarray
            Processed input data
        """
        if isinstance(X, pd.DataFrame):
            # Ensure the same features are present
            if set(X.columns) != set(self.feature_names_):
                self.logger.warning("Input features do not match the features used for training")
                
            # Convert to numpy array with the same order of features
            try:
                X_processed = X[self.feature_names_].values
            except KeyError:
                self.logger.warning("Some features are missing, using available features")
                # Use intersection of features
                common_features = list(set(X.columns) & set(self.feature_names_))
                if not common_features:
                    raise ValueError("No common features found between input data and training data")
                    
                X_processed = X[common_features].values
        else:
            X_processed = X
            
        return X_processed
    
    def _initialize_model(self, **kwargs):
        """
        Initialize the machine learning or deep learning model.
        
        Parameters:
        -----------
        **kwargs : 
            Model-specific parameters
        """
        # Dictionary to hold model-specific parameters
        model_params = {}
        
        # Extract common parameters
        model_params['random_state'] = self.random_state
        
        # Extract model-specific parameters from kwargs
        if self.model_type == 'logistic_regression':
            # Extract logistic regression parameters
            model_params['C'] = kwargs.get('C', 1.0)
            model_params['penalty'] = kwargs.get('penalty', 'l2')
            model_params['solver'] = kwargs.get('solver', 'lbfgs')
            model_params['max_iter'] = kwargs.get('max_iter', 1000)
            model_params['class_weight'] = kwargs.get('class_weight', None)
            
            # Create logistic regression model
            self.model = linear_model.LogisticRegression(**model_params)
            
        elif self.model_type == 'linear_regression':
            # Extract linear regression parameters
            model_params['fit_intercept'] = kwargs.get('fit_intercept', True)
            model_params['normalize'] = kwargs.get('normalize', False)
            
            # Create linear regression model
            self.model = linear_model.LinearRegression(**model_params)
            
        elif self.model_type == 'random_forest':
            # Extract random forest parameters
            model_params['n_estimators'] = kwargs.get('n_estimators', 100)
            model_params['max_depth'] = kwargs.get('max_depth', None)
            model_params['min_samples_split'] = kwargs.get('min_samples_split', 2)
            model_params['min_samples_leaf'] = kwargs.get('min_samples_leaf', 1)
            model_params['max_features'] = kwargs.get('max_features', 'auto')
            model_params['bootstrap'] = kwargs.get('bootstrap', True)
            model_params['class_weight'] = kwargs.get('class_weight', None)
            
            # Create random forest model based on task
            if self.task_type == 'classification':
                self.model = ensemble.RandomForestClassifier(**model_params)
            else:
                self.model = ensemble.RandomForestRegressor(**model_params)
                
        elif self.model_type == 'svm':
            # Extract SVM parameters
            model_params['C'] = kwargs.get('C', 1.0)
            model_params['kernel'] = kwargs.get('kernel', 'rbf')
            model_params['gamma'] = kwargs.get('gamma', 'scale')
            model_params['degree'] = kwargs.get('degree', 3)
            model_params['probability'] = kwargs.get('probability', True if self.task_type == 'classification' else False)
            model_params['class_weight'] = kwargs.get('class_weight', None)
            
            # Create SVM model based on task
            if self.task_type == 'classification':
                self.model = svm.SVC(**model_params)
            else:
                self.model = svm.SVR(**model_params)
                
        elif self.model_type == 'gradient_boosting':
            # Extract gradient boosting parameters
            model_params['n_estimators'] = kwargs.get('n_estimators', 100)
            model_params['learning_rate'] = kwargs.get('learning_rate', 0.1)
            model_params['max_depth'] = kwargs.get('max_depth', 3)
            model_params['min_samples_split'] = kwargs.get('min_samples_split', 2)
            model_params['min_samples_leaf'] = kwargs.get('min_samples_leaf', 1)
            model_params['subsample'] = kwargs.get('subsample', 1.0)
            model_params['max_features'] = kwargs.get('max_features', None)
            
            # Create gradient boosting model based on task
            if self.task_type == 'classification':
                self.model = ensemble.GradientBoostingClassifier(**model_params)
            else:
                self.model = ensemble.GradientBoostingRegressor(**model_params)
                
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed. Install with 'pip install xgboost'")
                
            # Extract XGBoost parameters
            model_params['n_estimators'] = kwargs.get('n_estimators', 100)
            model_params['learning_rate'] = kwargs.get('learning_rate', 0.1)
            model_params['max_depth'] = kwargs.get('max_depth', 3)
            model_params['min_child_weight'] = kwargs.get('min_child_weight', 1)
            model_params['gamma'] = kwargs.get('gamma', 0)
            model_params['subsample'] = kwargs.get('subsample', 1.0)
            model_params['colsample_bytree'] = kwargs.get('colsample_bytree', 1.0)
            model_params['reg_alpha'] = kwargs.get('reg_alpha', 0)
            model_params['reg_lambda'] = kwargs.get('reg_lambda', 1)
            
            # Create XGBoost model based on task
            if self.task_type == 'classification':
                model_params['objective'] = kwargs.get('objective', 'binary:logistic')
                model_params['eval_metric'] = kwargs.get('eval_metric', 'logloss')
                self.model = xgb.XGBClassifier(**model_params)
            else:
                model_params['objective'] = kwargs.get('objective', 'reg:squarederror')
                model_params['eval_metric'] = kwargs.get('eval_metric', 'rmse')
                self.model = xgb.XGBRegressor(**model_params)
                
        elif self.model_type in ['mlp', 'cnn', 'rnn', 'transformer']:
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow not installed. Install with 'pip install tensorflow'")
                
            # Initialize deep learning model based on type
            if self.model_type == 'mlp':
                self.model = self._create_mlp_model(**kwargs)
            elif self.model_type == 'cnn':
                self.model = self._create_cnn_model(**kwargs)
            elif self.model_type == 'rnn':
                self.model = self._create_rnn_model(**kwargs)
            elif self.model_type == 'transformer':
                self.model = self._create_transformer_model(**kwargs)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        self.logger.info(f"Initialized {self.model_type} model for {self.task_type}")
    
    def _perform_hyperparameter_tuning(self, X, y, param_grid, cv, scoring, n_jobs, verbose):
        """
        Perform hyperparameter tuning using grid search or random search.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The feature matrix
        y : numpy.ndarray
            The target variable
        param_grid : dict or list of dicts
            Grid of parameters to search
        cv : int or cv splitter
            Cross-validation strategy
        scoring : str
            Metric to optimize
        n_jobs : int
            Number of parallel jobs
        verbose : int
            Verbosity level
        """
        self.logger.info("Performing hyperparameter tuning")
        
        # Set default scoring based on task if not specified
        if scoring is None:
            if self.task_type == 'classification':
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'
                
        # Set up cross-validation strategy
        if isinstance(cv, int):
            if self.task_type == 'classification':
                cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            else:
                cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            cv_strategy = cv
            
        # For deep learning models, use custom tuning with RandomizedSearchCV
        if self.model_type in ['mlp', 'cnn', 'rnn', 'transformer']:
            self._tune_deep_learning(X, y, param_grid, cv_strategy, scoring, n_jobs, verbose)
            return
        
        # For traditional ML models, use GridSearchCV
        grid_search = GridSearchCV(
            self.model, 
            param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True,
            refit=True
        )
        
        # Perform grid search
        grid_search.fit(X, y)
        
        # Get best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        
        # Log results
        self.logger.info(f"Best parameters: {self.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    def _tune_deep_learning(self, X, y, param_grid, cv, scoring, n_jobs, verbose):
        """
        Tune hyperparameters for deep learning models.
        
        Uses a custom approach with either grid search or random search.
        """
        self.logger.info("Tuning deep learning model hyperparameters")
        
        # Get the number of classes for classification tasks
        if self.task_type == 'classification':
            if len(y.shape) > 1:  # One-hot encoded
                n_classes = y.shape[1]
            else:
                n_classes = len(np.unique(y))
        else:
            n_classes = None
            
        # Extract input shape
        input_shape = X.shape[1:]
        
        # Create a wrapper class for the deep learning model
        class KerasWrapper(BaseEstimator):
            def __init__(self, build_fn, **params):
                self.build_fn = build_fn
                self.params = params
                self.model = None
                
            def fit(self, X, y, **kwargs):
                # Build the model
                self.model = self.build_fn(**self.params)
                
                # Compile the model
                self.model.compile(
                    optimizer=self.params.get('optimizer', 'adam'),
                    loss=self.params.get('loss', 'binary_crossentropy' if n_classes == 2 else 'categorical_crossentropy' if n_classes > 2 else 'mse'),
                    metrics=self.params.get('metrics', ['accuracy'] if n_classes is not None else ['mae', 'mse'])
                )
                
                # Set up callbacks
                callbacks_list = []
                
                # Early stopping
                if self.params.get('early_stopping', True):
                    callbacks_list.append(
                        EarlyStopping(
                            monitor=self.params.get('monitor', 'val_loss'),
                            patience=self.params.get('patience', 10),
                            restore_best_weights=True
                        )
                    )
                    
                # Reduce learning rate
                if self.params.get('reduce_lr', False):
                    callbacks_list.append(
                        ReduceLROnPlateau(
                            monitor=self.params.get('monitor', 'val_loss'),
                            factor=self.params.get('lr_factor', 0.1),
                            patience=self.params.get('lr_patience', 5),
                            min_lr=self.params.get('min_lr', 1e-6)
                        )
                    )
                
                # Fit the model
                history = self.model.fit(
                    X, y,
                    batch_size=self.params.get('batch_size', 32),
                    epochs=self.params.get('epochs', 100),
                    validation_split=self.params.get('validation_split', 0.2),
                    callbacks=callbacks_list,
                    verbose=0
                )
                
                return self
                
            def predict(self, X):
                # Make predictions
                if self.task_type == 'classification':
                    y_pred_prob = self.model.predict(X)
                    
                    if n_classes > 2:  # Multi-class
                        return np.argmax(y_pred_prob, axis=1)
                    else:  # Binary
                        return (y_pred_prob > 0.5).astype(int).flatten()
                else:
                    return self.model.predict(X).flatten()
                    
            def predict_proba(self, X):
                # Only for classification
                if self.task_type != 'classification':
                    raise ValueError("predict_proba is only available for classification")
                    
                return self.model.predict(X)
                
            def score(self, X, y):
                # Score based on task
                if self.task_type == 'classification':
                    y_pred = self.predict(X)
                    return accuracy_score(y, y_pred)
                else:
                    y_pred = self.predict(X)
                    return -mean_squared_error(y, y_pred)  # Negative MSE for maximization
        
        # Create a builder function for the appropriate model type
        if self.model_type == 'mlp':
            def build_model(**params):
                return self._create_mlp_model(
                    input_shape=input_shape, 
                    n_classes=n_classes, 
                    **params
                )
        elif self.model_type == 'cnn':
            def build_model(**params):
                return self._create_cnn_model(
                    input_shape=input_shape,
                    n_classes=n_classes,
                    **params
                )
        elif self.model_type == 'rnn':
            def build_model(**params):
                return self._create_rnn_model(
                    input_shape=input_shape,
                    n_classes=n_classes,
                    **params
                )
        elif self.model_type == 'transformer':
            def build_model(**params):
                return self._create_transformer_model(
                    input_shape=input_shape,
                    n_classes=n_classes,
                    **params
                )
        
        # Create the model wrapper
        model_wrapper = KerasWrapper(build_fn=build_model)
        
        # Use RandomizedSearchCV for efficiency with deep learning
        random_search = RandomizedSearchCV(
            model_wrapper,
            param_distributions=param_grid,
            n_iter=10,  # Number of parameter settings sampled
            cv=cv,
            scoring=scoring,
            n_jobs=1,  # Deep learning models are better run in series
            verbose=verbose,
            return_train_score=True,
            refit=True,
            random_state=self.random_state
        )
        
        # Perform random search
        random_search.fit(X, y)
        
        # Get best parameters
        self.best_params_ = random_search.best_params_
        
        # Rebuild the best model
        self.model = build_model(**self.best_params_)
        
        # Compile the model
        self.model.compile(
            optimizer=self.best_params_.get('optimizer', 'adam'),
            loss=self.best_params_.get('loss', 'binary_crossentropy' if n_classes == 2 else 'categorical_crossentropy' if n_classes > 2 else 'mse'),
            metrics=self.best_params_.get('metrics', ['accuracy'] if n_classes is not None else ['mae', 'mse'])
        )
        
        # Set up callbacks
        callbacks_list = self._get_dl_callbacks(**self.best_params_)
        
        # Fit the model with best parameters
        self.history_ = self.model.fit(
            X, y,
            batch_size=self.best_params_.get('batch_size', 32),
            epochs=self.best_params_.get('epochs', 100),
            validation_split=self.best_params_.get('validation_split', 0.2),
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        # Log results
        self.logger.info(f"Best parameters: {self.best_params_}")
        self.logger.info(f"Best CV score: {random_search.best_score_:.4f}")
    
    def _get_dl_callbacks(self, **kwargs):
        """Get callbacks for deep learning models."""
        callbacks_list = []
        
        # Early stopping
        if kwargs.get('early_stopping', True):
            callbacks_list.append(
                EarlyStopping(
                    monitor=kwargs.get('monitor', 'val_loss'),
                    patience=kwargs.get('patience', 10),
                    restore_best_weights=True
                )
            )
            
        # Reduce learning rate
        if kwargs.get('reduce_lr', False):
            callbacks_list.append(
                ReduceLROnPlateau(
                    monitor=kwargs.get('monitor', 'val_loss'),
                    factor=kwargs.get('lr_factor', 0.1),
                    patience=kwargs.get('lr_patience', 5),
                    min_lr=kwargs.get('min_lr', 1e-6)
                )
            )
            
        # Model checkpoint
        if kwargs.get('checkpoint', False):
            checkpoint_path = kwargs.get('checkpoint_path', './model_checkpoint.h5')
            callbacks_list.append(
                ModelCheckpoint(
                    checkpoint_path,
                    monitor=kwargs.get('monitor', 'val_loss'),
                    save_best_only=True,
                    mode='min' if 'loss' in kwargs.get('monitor', 'val_loss') else 'max'
                )
            )
            
        return callbacks_list
    
    def _deep_learning_cross_validate(self, X, y, cv, scoring, **kwargs):
        """
        Perform cross-validation for deep learning models.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The feature matrix
        y : numpy.ndarray
            The target variable
        cv : int or cross-validation generator
            Cross-validation strategy
        scoring : str
            Metric to evaluate
        **kwargs : 
            Additional parameters
            
        Returns:
        --------
        dict
            Dictionary of cross-validation results
        """
        # Set up cross-validation
        if isinstance(cv, int):
            if self.task_type == 'classification':
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            else:
                cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            splits = list(cv_splitter.split(X, y))
        else:
            # Use provided cv splitter
            splits = list(cv.split(X, y))
            
        # Extract parameters
        batch_size = kwargs.get('batch_size', 32)
        epochs = kwargs.get('epochs', 100)
        validation_split = kwargs.get('validation_split', 0.2)
        verbose = kwargs.get('verbose', 0)
        
        # Initialize results
        cv_scores = []
        
        # Perform cross-validation
        for i, (train_idx, test_idx) in enumerate(splits):
            self.logger.info(f"Fold {i+1}/{len(splits)}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Reinitialize model for each fold
            if self.model_type == 'mlp':
                model = self._create_mlp_model(**kwargs)
            elif self.model_type == 'cnn':
                model = self._create_cnn_model(**kwargs)
            elif self.model_type == 'rnn':
                model = self._create_rnn_model(**kwargs)
            elif self.model_type == 'transformer':
                model = self._create_transformer_model(**kwargs)
                
            # Compile model
            if self.task_type == 'classification':
                if len(self.classes_) == 2:
                    loss = kwargs.get('loss', 'binary_crossentropy')
                else:
                    loss = kwargs.get('loss', 'categorical_crossentropy')
            else:
                loss = kwargs.get('loss', 'mse')
                
            model.compile(
                optimizer=kwargs.get('optimizer', 'adam'),
                loss=loss,
                metrics=kwargs.get('metrics', ['accuracy'] if self.task_type == 'classification' else ['mae'])
            )
            
            # Set up callbacks
            callbacks_list = self._get_dl_callbacks(**kwargs)
            
            # Train model
            model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks_list,
                verbose=verbose
            )
            
            # Evaluate model
            if self.task_type == 'classification':
                # Make predictions
                y_pred_prob = model.predict(X_test)
                
                if len(self.classes_) > 2:  # Multi-class
                    y_pred = np.argmax(y_pred_prob, axis=1)
                    y_test_decoded = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
                else:  # Binary
                    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
                    y_test_decoded = y_test.flatten() if len(y_test.shape) > 1 else y_test
                
                # Calculate score based on selected metric
                if scoring == 'accuracy':
                    score = accuracy_score(y_test_decoded, y_pred)
                elif scoring == 'precision':
                    score = precision_score(y_test_decoded, y_pred, average='weighted')
                elif scoring == 'recall':
                    score = recall_score(y_test_decoded, y_pred, average='weighted')
                elif scoring == 'f1':
                    score = f1_score(y_test_decoded, y_pred, average='weighted')
                elif scoring == 'roc_auc':
                    try:
                        if len(self.classes_) > 2:
                            score = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
                        else:
                            score = roc_auc_score(y_test, y_pred_prob)
                    except:
                        score = np.nan
                else:
                    # Default to accuracy
                    score = accuracy_score(y_test_decoded, y_pred)
            else:
                # Regression
                y_pred = model.predict(X_test).flatten()
                
                # Calculate score based on selected metric
                if scoring == 'neg_mean_squared_error':
                    score = -mean_squared_error(y_test, y_pred)
                elif scoring == 'neg_mean_absolute_error':
                    score = -mean_absolute_error(y_test, y_pred)
                elif scoring == 'r2':
                    score = r2_score(y_test, y_pred)
                else:
                    # Default to negative MSE
                    score = -mean_squared_error(y_test, y_pred)
                    
            cv_scores.append(score)
            
        # Calculate mean and std of scores
        cv_results = {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'cv_scores': np.array(cv_scores)
        }
        
        return cv_results
    
    def _calculate_feature_importances(self):
        """
        Calculate feature importances if applicable to the model.
        """
        # Skip if no feature names available
        if self.feature_names_ is None:
            return
            
        # Try to extract feature importances based on model type
        try:
            if self.model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
                # These models have a feature_importances_ attribute
                importances = self.model.feature_importances_
                self.feature_importances_ = pd.Series(
                    importances, index=self.feature_names_
                ).sort_values(ascending=False)
                
            elif self.model_type in ['logistic_regression', 'linear_regression', 'svm']:
                # These models have coefficients
                if hasattr(self.model, 'coef_'):
                    if len(self.model.coef_.shape) > 1:
                        # For multi-class, use the mean absolute coefficient across classes
                        importances = np.mean(np.abs(self.model.coef_), axis=0)
                    else:
                        importances = np.abs(self.model.coef_)
                        
                    self.feature_importances_ = pd.Series(
                        importances, index=self.feature_names_
                    ).sort_values(ascending=False)
                    
            elif self.model_type in ['mlp', 'cnn', 'rnn', 'transformer']:
                # For deep learning, we don't have a straightforward method
                # A simple approach would be to use permutation importance, but that's computationally expensive
                # Instead, we'll use a placeholder message
                self.logger.info(
                    "Feature importances for deep learning models not automatically calculated. "
                    "Use plot_permutation_importance() for a more accurate assessment."
                )
                
        except Exception as e:
            self.logger.warning(f"Error calculating feature importances: {str(e)}")
    
    def _create_mlp_model(self, **kwargs):
        """
        Create a Multi-layer Perceptron (fully connected) neural network.
        
        Parameters:
        -----------
        **kwargs : 
            Model-specific parameters:
            - input_shape: Tuple, shape of input features
            - n_classes: Int, number of classes (for classification)
            - hidden_layers: List, number of units in each hidden layer
            - activation: Str, activation function
            - dropout_rate: Float, dropout rate for regularization
            - batch_norm: Bool, whether to use batch normalization
            - l1_reg: Float, L1 regularization strength
            - l2_reg: Float, L2 regularization strength
            
        Returns:
        --------
        tensorflow.keras.Model
            The compiled MLP model
        """
        # Extract parameters
        input_shape = kwargs.get('input_shape', (self.feature_names_.__len__(),))
        hidden_layers = kwargs.get('hidden_layers', [128, 64])
        activation = kwargs.get('activation', 'relu')
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        batch_norm = kwargs.get('batch_norm', True)
        l1_reg = kwargs.get('l1_reg', 0.0)
        l2_reg = kwargs.get('l2_reg', 0.001)
        
        # Set up regularization
        kernel_regularizer = None
        if l1_reg > 0 or l2_reg > 0:
            kernel_regularizer = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
            
        # Create model
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=input_shape))
        
        # Hidden layers
        for units in hidden_layers:
            model.add(Dense(
                units,
                kernel_regularizer=kernel_regularizer
            ))
            
            if batch_norm:
                model.add(BatchNormalization())
                
            model.add(Activation(activation))
            
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Output layer
        if self.task_type == 'classification':
            n_classes = kwargs.get('n_classes', len(self.classes_) if self.classes_ is not None else 2)
            
            if n_classes == 2:
                model.add(Dense(1, activation='sigmoid'))
            else:
                model.add(Dense(n_classes, activation='softmax'))
        else:
            model.add(Dense(1, activation='linear'))
        
        return model
    
    def _create_cnn_model(self, **kwargs):
        """
        Create a Convolutional Neural Network (1D CNN).
        
        Parameters:
        -----------
        **kwargs : 
            Model-specific parameters:
            - input_shape: Tuple, shape of input features
            - n_classes: Int, number of classes (for classification)
            - conv_layers: List of tuples, each containing (filters, kernel_size, pool_size)
            - hidden_layers: List, number of units in each dense layer after convolution
            - activation: Str, activation function
            - dropout_rate: Float, dropout rate for regularization
            - batch_norm: Bool, whether to use batch normalization
            
        Returns:
        --------
        tensorflow.keras.Model
            The compiled CNN model
        """
        # Extract parameters
        input_shape = kwargs.get('input_shape', (self.feature_names_.__len__(), 1))
        conv_layers = kwargs.get('conv_layers', [(64, 3, 2), (128, 3, 2)])
        hidden_layers = kwargs.get('hidden_layers', [128, 64])
        activation = kwargs.get('activation', 'relu')
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        batch_norm = kwargs.get('batch_norm', True)
        
        # Ensure input shape is compatible with 1D CNN
        if len(input_shape) == 1:
            # Add a channel dimension for 1D convolution
            input_shape = (input_shape[0], 1)
            
        # Create model
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=input_shape))
        
        # Convolutional layers
        for filters, kernel_size, pool_size in conv_layers:
            model.add(Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same'
            ))
            
            if batch_norm:
                model.add(BatchNormalization())
                
            model.add(Activation(activation))
            model.add(MaxPooling1D(pool_size=pool_size))
            
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Flatten before dense layers
        model.add(Flatten())
        
        # Dense layers
        for units in hidden_layers:
            model.add(Dense(units))
            
            if batch_norm:
                model.add(BatchNormalization())
                
            model.add(Activation(activation))
            
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Output layer
        if self.task_type == 'classification':
            n_classes = kwargs.get('n_classes', len(self.classes_) if self.classes_ is not None else 2)
            
            if n_classes == 2:
                model.add(Dense(1, activation='sigmoid'))
            else:
                model.add(Dense(n_classes, activation='softmax'))
        else:
            model.add(Dense(1, activation='linear'))
        
        return model
    
    def _create_rnn_model(self, **kwargs):
        """
        Create a Recurrent Neural Network (LSTM/Bi-LSTM).
        
        Parameters:
        -----------
        **kwargs : 
            Model-specific parameters:
            - input_shape: Tuple, shape of input features
            - n_classes: Int, number of classes (for classification)
            - rnn_layers: List of tuples, each containing (units, return_sequences)
            - bidirectional: Bool, whether to use bidirectional LSTM
            - hidden_layers: List, number of units in each dense layer after RNN
            - activation: Str, activation function
            - dropout_rate: Float, dropout rate for regularization
            
        Returns:
        --------
        tensorflow.keras.Model
            The compiled RNN model
        """
        # Extract parameters
        input_shape = kwargs.get('input_shape', (self.feature_names_.__len__(), 1))
        rnn_layers = kwargs.get('rnn_layers', [(64, True), (64, False)])
        bidirectional = kwargs.get('bidirectional', True)
        hidden_layers = kwargs.get('hidden_layers', [64])
        activation = kwargs.get('activation', 'relu')
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        
        # Ensure input shape is compatible with RNN
        if len(input_shape) == 1:
            # Add a time step dimension for RNN
            input_shape = (input_shape[0], 1)
            
        # Create model
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=input_shape))
        
        # RNN layers
        for i, (units, return_sequences) in enumerate(rnn_layers):
            if bidirectional:
                model.add(Bidirectional(LSTM(
                    units=units,
                    return_sequences=return_sequences
                )))
            else:
                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences
                ))
                
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Dense layers
        for units in hidden_layers:
            model.add(Dense(units, activation=activation))
            
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Output layer
        if self.task_type == 'classification':
            n_classes = kwargs.get('n_classes', len(self.classes_) if self.classes_ is not None else 2)
            
            if n_classes == 2:
                model.add(Dense(1, activation='sigmoid'))
            else:
                model.add(Dense(n_classes, activation='softmax'))
        else:
            model.add(Dense(1, activation='linear'))
        
        return model
    
    def _create_transformer_model(self, **kwargs):
        """
        Create a Transformer-based model.
        
        Parameters:
        -----------
        **kwargs : 
            Model-specific parameters:
            - input_shape: Tuple, shape of input features
            - n_classes: Int, number of classes (for classification)
            - num_heads: Int, number of attention heads
            - ff_dim: Int, feed-forward dimension
            - num_transformer_blocks: Int, number of transformer blocks
            - hidden_layers: List, number of units in each dense layer after transformer
            - dropout_rate: Float, dropout rate for regularization
            
        Returns:
        --------
        tensorflow.keras.Model
            The compiled Transformer model
        """
        # Extract parameters
        input_shape = kwargs.get('input_shape', (self.feature_names_.__len__(), 1))
        num_heads = kwargs.get('num_heads', 4)
        ff_dim = kwargs.get('ff_dim', 64)
        num_transformer_blocks = kwargs.get('num_transformer_blocks', 2)
        hidden_layers = kwargs.get('hidden_layers', [64])
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        
        # Ensure input shape is compatible with Transformer
        if len(input_shape) == 1:
            # Add a feature dimension for transformer
            input_shape = (input_shape[0], 1)
            
        # Create model using Functional API
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Create multiple transformer blocks
        for _ in range(num_transformer_blocks):
            # Self-attention
            attention_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=input_shape[-1]
            )(x, x)
            
            # Skip connection and normalization
            x = LayerNormalization(epsilon=1e-6)(attention_output + x)
            
            # Feed Forward Network
            ffn = Dense(ff_dim, activation="relu")(x)
            ffn = Dense(input_shape[-1])(ffn)
            
            # Skip connection and normalization
            x = LayerNormalization(epsilon=1e-6)(ffn + x)
            
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)
        
        # Global pooling to reduce sequence dimension
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        for units in hidden_layers:
            x = Dense(units, activation="relu")(x)
            
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)
        
        # Output layer
        if self.task_type == 'classification':
            n_classes = kwargs.get('n_classes', len(self.classes_) if self.classes_ is not None else 2)
            
            if n_classes == 2:
                outputs = Dense(1, activation='sigmoid')(x)
            else:
                outputs = Dense(n_classes, activation='softmax')(x)
        else:
            outputs = Dense(1, activation='linear')(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    # Visualization methods
    def plot_feature_importances(self, n_features=20, **kwargs):
        """
        Plot feature importances for models that support them.
        
        Parameters:
        -----------
        n_features : int
            Number of top features to display
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size
            - title: Plot title
            - color: Bar color
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available for this model")
            
        # Extract parameters
        figsize = kwargs.get('figsize', (10, max(5, min(n_features/2, 10))))
        title = kwargs.get('title', f"Feature Importances ({self.model_type})")
        color = kwargs.get('color', 'skyblue')
        
        # Get top features
        top_features = self.feature_importances_.head(n_features)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        top_features.sort_values().plot(
            kind='barh',
            ax=ax,
            color=color
        )
        
        # Add labels and title
        ax.set_title(title)
        ax.set_xlabel('Importance')
        
        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curve(self, **kwargs):
        """
        Plot learning curve for deep learning models.
        
        Parameters:
        -----------
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size
            - title: Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure or None if history not available
        """
        if self.history_ is None:
            raise ValueError("Training history not available")
            
        # Extract parameters
        figsize = kwargs.get('figsize', (12, 5))
        title = kwargs.get('title', f"Learning Curves ({self.model_type})")
        
        # Get history dictionary
        history = self.history_.history
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        axes[0].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss')
            
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot metrics
        metric_keys = [k for k in history.keys() if ('acc' in k.lower() or 'mae' in k.lower() or 'mse' in k.lower()) and not k.startswith('val_')]
        
        if metric_keys:
            metric_key = metric_keys[0]
            axes[1].plot(history[metric_key], label=f'Training {metric_key}')
            val_metric_key = f'val_{metric_key}'
            
            if val_metric_key in history:
                axes[1].plot(history[val_metric_key], label=f'Validation {metric_key}')
                
            axes[1].set_title(f'Model {metric_key}')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel(metric_key)
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        
        # Add overall title
        fig.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def plot_confusion_matrix(self, **kwargs):
        """
        Plot confusion matrix for classification models.
        
        Parameters:
        -----------
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size
            - title: Plot title
            - normalize: Whether to normalize the confusion matrix
            - cmap: Colormap
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.task_type != 'classification':
            raise ValueError("Confusion matrix only available for classification tasks")
            
        if not hasattr(self, '_evaluation_data'):
            raise ValueError("No evaluation data available. Call evaluate() first")
            
        # Extract parameters
        figsize = kwargs.get('figsize', (8, 6))
        title = kwargs.get('title', 'Confusion Matrix')
        normalize = kwargs.get('normalize', True)
        cmap = kwargs.get('cmap', 'Blues')
        
        # Get confusion matrix
        y_true = self._evaluation_data['y_true']
        y_pred = self._evaluation_data['y_pred']
        
        # Convert to labels if needed
        if self.label_encoder is not None:
            try:
                y_true_labels = self.label_encoder.inverse_transform(y_true)
                y_pred_labels = self.label_encoder.inverse_transform(y_pred)
                class_names = self.classes_
            except:
                y_true_labels = y_true
                y_pred_labels = y_pred
                class_names = np.unique(np.concatenate([y_true, y_pred]))
        else:
            y_true_labels = y_true
            y_pred_labels = y_pred
            class_names = np.unique(np.concatenate([y_true, y_pred]))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_names)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        
        # Show all ticks and label them
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            ylabel='True label',
            xlabel='Predicted label'
        )
        
        # Rotate x tick labels if there are many classes
        if len(class_names) > 4:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )
                
        fig.tight_layout()
        return fig
    
    def plot_roc_curve(self, **kwargs):
        """
        Plot ROC curve for binary classification.
        
        Parameters:
        -----------
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size
            - title: Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.task_type != 'classification':
            raise ValueError("ROC curve only available for classification tasks")
            
        if not hasattr(self, '_evaluation_data') or not self._evaluation_data.get('has_proba', False):
            raise ValueError("No evaluation data with probabilities available. Call evaluate() first")
            
        # Extract parameters
        figsize = kwargs.get('figsize', (8, 6))
        title = kwargs.get('title', 'ROC Curve')
        
        # Get data
        y_true = self._evaluation_data['y_true']
        y_pred_proba = self._evaluation_data['y_pred_proba']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # For multi-class classification
        if len(self.classes_) > 2:
            # One-hot encode true labels if needed
            if len(y_true.shape) == 1:
                y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(self.classes_))
            else:
                y_true_onehot = y_true
                
            # Plot ROC curve for each class
            for i, class_name in enumerate(self.classes_):
                fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
                roc_auc = roc_auc_score(y_true_onehot[:, i], y_pred_proba[:, i])
                ax.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
        else:
            # Binary classification
            # Get positive class probabilities
            if y_pred_proba.shape[1] > 1:
                pos_probs = y_pred_proba[:, 1]
            else:
                pos_probs = y_pred_proba.flatten()
                
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, pos_probs)
            roc_auc = roc_auc_score(y_true, pos_probs)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        
        # Add diagonal reference line
        ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
        
        # Set labels and title
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        
        # Add legend and grid
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Set axis limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        fig.tight_layout()
        return fig
    
    def plot_regression_results(self, **kwargs):
        """
        Plot regression results.
        
        Parameters:
        -----------
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size
            - title: Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.task_type != 'regression':
            raise ValueError("Regression results only available for regression tasks")
            
        if not hasattr(self, '_evaluation_data'):
            raise ValueError("No evaluation data available. Call evaluate() first")
            
        # Extract parameters
        figsize = kwargs.get('figsize', (12, 5))
        title = kwargs.get('title', f'Regression Results ({self.model_type})')
        
        # Get data
        y_true = self._evaluation_data['y_true']
        y_pred = self._evaluation_data['y_pred']
        
        # Calculate regression metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot: Predicted vs True
        ax1.scatter(y_true, y_pred, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add labels and title
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predictions')
        ax1.set_title(f'Predicted vs True Values\nR² = {r2:.3f}, RMSE = {rmse:.3f}')
        
        # Add grid
        ax1.grid(alpha=0.3)
        
        # Residual plot
        residuals = y_pred - y_true
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        
        # Add labels and title
        ax2.set_xlabel('Predictions')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        
        # Add grid
        ax2.grid(alpha=0.3)
        
        # Add overall title
        fig.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        return fig
    
    def plot_permutation_importance(self, X, y, **kwargs):
        """
        Plot permutation feature importance.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size
            - title: Plot title
            - n_repeats: Number of times to permute each feature
            - n_jobs: Number of parallel jobs
            - n_features: Number of top features to display
            - scoring: Scoring metric for importance calculation
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        from sklearn.inspection import permutation_importance
        
        # Process input data
        X_processed = self._preprocess_input(X)
        
        # Process target data
        if self.task_type == 'classification':
            if self.label_encoder is not None:
                if isinstance(y, pd.Series):
                    y_processed = self.label_encoder.transform(y.values)
                else:
                    y_processed = self.label_encoder.transform(y)
            else:
                y_processed = y
        else:
            # For regression, ensure y is a numpy array
            if isinstance(y, pd.Series):
                y_processed = y.values
            else:
                y_processed = y
                
        # Extract parameters
        figsize = kwargs.get('figsize', (10, 8))
        title = kwargs.get('title', 'Permutation Feature Importance')
        n_repeats = kwargs.get('n_repeats', 10)
        n_jobs = kwargs.get('n_jobs', -1)
        n_features = kwargs.get('n_features', 20)
        scoring = kwargs.get('scoring', None)  # Use default scorer
        
        # Calculate permutation importance
        self.logger.info("Calculating permutation feature importance...")
        
        # For deep learning models, create a wrapper to use with permutation_importance
        if self.model_type in ['mlp', 'cnn', 'rnn', 'transformer']:
            # Create a scikit-learn compatible wrapper
            class KerasWrapper:
                def __init__(self, model, task_type, label_encoder=None):
                    self.model = model
                    self.task_type = task_type
                    self.label_encoder = label_encoder
                    
                def predict(self, X):
                    if self.task_type == 'classification':
                        # Get probability predictions
                        y_pred_prob = self.model.predict(X)
                        
                        if y_pred_prob.shape[1] > 1:  # Multi-class
                            return np.argmax(y_pred_prob, axis=1)
                        else:  # Binary
                            return (y_pred_prob > 0.5).astype(int).flatten()
                    else:
                        return self.model.predict(X).flatten()
                    
                def predict_proba(self, X):
                    if self.task_type != 'classification':
                        raise ValueError("predict_proba only available for classification")
                        
                    return self.model.predict(X)
                    
                def score(self, X, y):
                    if self.task_type == 'classification':
                        # Get predictions
                        y_pred = self.predict(X)
                        
                        # Calculate accuracy
                        return accuracy_score(y, y_pred)
                    else:
                        # Get predictions
                        y_pred = self.predict(X)
                        
                        # Calculate R-squared
                        return r2_score(y, y_pred)
            
            # Create wrapper
            model_wrapper = KerasWrapper(self.model, self.task_type, self.label_encoder)
            
            # Calculate permutation importance
            result = permutation_importance(
                model_wrapper,
                X_processed,
                y_processed,
                n_repeats=n_repeats,
                random_state=self.random_state,
                n_jobs=n_jobs,
                scoring=scoring
            )
        else:
            # For scikit-learn models
            result = permutation_importance(
                self.model,
                X_processed,
                y_processed,
                n_repeats=n_repeats,
                random_state=self.random_state,
                n_jobs=n_jobs,
                scoring=scoring
            )
            
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        # Select top features
        top_features = importance_df.head(n_features)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot feature importances
        y_pos = np.arange(len(top_features))
        ax.barh(
            y_pos,
            top_features['importance_mean'],
            xerr=top_features['importance_std'],
            align='center',
            alpha=0.8
        )
        
        # Add feature names
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        
        # Add labels and title
        ax.set_xlabel('Permutation Importance')
        ax.set_title(title)
        
        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Invert y-axis to show highest importances at the top
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig, importance_df
