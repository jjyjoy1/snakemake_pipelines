import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import logging
import warnings
import os
import pickle
import time
from datetime import datetime
import itertools
from functools import partial
from collections import defaultdict
import joblib
from sklearn.model_selection import (
    cross_validate, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold,
    LeaveOneOut, ShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold,
    cross_val_score, cross_val_predict, train_test_split, learning_curve,
    validation_curve, permutation_test_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    confusion_matrix, precision_recall_curve, roc_curve, average_precision_score,
    classification_report, silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    make_scorer
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.utils import resample
from sklearn.exceptions import NotFittedError
from scipy import stats

# Try importing optional dependencies
try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not installed. Bayesian optimization unavailable. "
                 "Install with 'pip install optuna'.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Model explanation with SHAP unavailable. "
                 "Install with 'pip install shap'.")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class BioinformaticsModelEvaluator:
    """
    Step 6: Model Evaluation & Optimization
    
    A class for evaluating and optimizing ML/DL models in bioinformatics,
    providing robust performance assessment and hyperparameter tuning.
    
    This class implements:
    - Various cross-validation strategies (k-fold, nested CV, etc.)
    - Hyperparameter optimization techniques (Grid Search, Random Search, Bayesian Optimization)
    - Comprehensive performance metrics for classification, regression, and clustering tasks
    - Visualization tools for model evaluation and comparison
    - Model calibration and statistical significance testing
    - Ensemble methods for combining multiple models
    
    Can work with models from previous steps in the bioinformatics pipeline.
    """
    
    def __init__(self, task_type='classification', random_state=42, logger=None):
        """
        Initialize the model evaluator.
        
        Parameters:
        -----------
        task_type : str
            Type of machine learning task ('classification', 'regression', or 'clustering')
        random_state : int
            Random seed for reproducibility
        logger : logging.Logger
            Logger for tracking the evaluation process
        """
        self.task_type = task_type.lower()
        self.random_state = random_state
        self.cv_results_ = {}
        self.optimization_results_ = {}
        self.evaluation_metrics_ = {}
        self.best_params_ = {}
        self.best_estimator_ = None
        self.best_score_ = None
        self.feature_importances_ = None
        
        # Set up logger
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
            
        # Validate task_type
        valid_tasks = ['classification', 'regression', 'clustering']
        if self.task_type not in valid_tasks:
            raise ValueError(f"Invalid task type '{self.task_type}'. Choose from: {', '.join(valid_tasks)}")
    
    def _setup_logger(self):
        """Setup a basic logger if none is provided."""
        logger = logging.getLogger("BioinformaticsModelEvaluator")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def cross_validate_model(self, model, X, y, cv_strategy='kfold', n_splits=5, scoring=None, **kwargs):
        """
        Perform cross-validation for model evaluation.
        
        Parameters:
        -----------
        model : estimator object
            Model to evaluate
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        cv_strategy : str or cv splitter
            Cross-validation strategy to use:
            - 'kfold': K-Fold cross-validation
            - 'stratified': Stratified K-Fold cross-validation
            - 'repeated_kfold': Repeated K-Fold cross-validation
            - 'repeated_stratified': Repeated Stratified K-Fold cross-validation
            - 'leave_one_out': Leave-One-Out cross-validation
            - 'shuffle_split': Random permutation cross-validation
            - Or any sklearn-compatible CV splitter object
        n_splits : int
            Number of splits for cross-validation
        scoring : str, list, or dict
            Scoring metrics to evaluate (if None, uses default for task_type)
        **kwargs :
            Additional parameters for cross-validation:
            - n_repeats : int
                Number of repeats for repeated cross-validation
            - test_size : float
                Test size for shuffle_split
            - return_estimator : bool
                Whether to return estimator objects
            - return_train_score : bool
                Whether to calculate score on training set
            - n_jobs : int
                Number of parallel jobs
            - verbose : int
                Verbosity level
            
        Returns:
        --------
        dict
            Dictionary of cross-validation results
        """
        # Process input data
        X_processed, y_processed = self._preprocess_data(X, y)
        
        # Set default scoring based on task type if not specified
        if scoring is None:
            if self.task_type == 'classification':
                scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr_weighted']
            elif self.task_type == 'regression':
                scoring = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error', 'explained_variance']
            else:  # clustering
                # For clustering, we typically don't use CV this way
                scoring = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
                # We'll register custom scorer for clustering
                scoring = self._register_clustering_scorer(scoring)
                
        # Extract additional parameters
        n_repeats = kwargs.get('n_repeats', 3)
        test_size = kwargs.get('test_size', 0.2)
        return_estimator = kwargs.get('return_estimator', False)
        return_train_score = kwargs.get('return_train_score', True)
        n_jobs = kwargs.get('n_jobs', -1)
        verbose = kwargs.get('verbose', 0)
        
        # Set up cross-validation strategy
        if isinstance(cv_strategy, str):
            cv_strategy = cv_strategy.lower()
            
            if cv_strategy == 'kfold':
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            elif cv_strategy == 'stratified':
                if self.task_type == 'classification':
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                else:
                    self.logger.warning("Stratified K-Fold is for classification only. Using regular K-Fold instead.")
                    cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            elif cv_strategy == 'repeated_kfold':
                cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state)
            elif cv_strategy == 'repeated_stratified':
                if self.task_type == 'classification':
                    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state)
                else:
                    self.logger.warning("Stratified K-Fold is for classification only. Using regular Repeated K-Fold instead.")
                    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state)
            elif cv_strategy == 'leave_one_out':
                cv = LeaveOneOut()
            elif cv_strategy == 'shuffle_split':
                cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=self.random_state)
            else:
                raise ValueError(f"Unknown CV strategy: {cv_strategy}")
        else:
            # Use provided CV splitter object
            cv = cv_strategy
            
        # Check if model is a deep learning model
        is_dl_model = self._is_deep_learning_model(model)
        
        # For deep learning models, use custom CV implementation
        if is_dl_model:
            cv_results = self._deep_learning_cross_validate(
                model, X_processed, y_processed, cv, scoring, **kwargs
            )
        else:
            # Use scikit-learn's cross_validate
            try:
                cv_results = cross_validate(
                    model, 
                    X_processed, 
                    y_processed,
                    cv=cv,
                    scoring=scoring,
                    return_estimator=return_estimator,
                    return_train_score=return_train_score,
                    n_jobs=n_jobs,
                    verbose=verbose
                )
            except Exception as e:
                self.logger.error(f"Error in cross-validation: {str(e)}")
                # Try a simplified approach
                self.logger.info("Trying simplified cross-validation...")
                if isinstance(scoring, (list, dict)):
                    primary_scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
                    self.logger.info(f"Using {primary_scoring} as primary metric")
                else:
                    primary_scoring = scoring
                    
                try:
                    scores = cross_val_score(
                        model, X_processed, y_processed, 
                        cv=cv, scoring=primary_scoring, n_jobs=n_jobs
                    )
                    cv_results = {
                        f'test_{primary_scoring}': scores,
                        'test_score': scores
                    }
                except Exception as e2:
                    self.logger.error(f"Error in simplified cross-validation: {str(e2)}")
                    raise
        
        # Calculate summary statistics
        summary = self._summarize_cv_results(cv_results)
        
        # Log results
        self.logger.info(f"Cross-validation completed with {cv_strategy} strategy ({n_splits} splits)")
        for metric, values in summary.items():
            if isinstance(values, dict):
                self.logger.info(f"{metric}: mean={values['mean']:.4f}, std={values['std']:.4f}")
            
        # Store results
        self.cv_results_ = {'raw': cv_results, 'summary': summary}
        
        return self.cv_results_
    
    def _deep_learning_cross_validate(self, model, X, y, cv, scoring, **kwargs):
        """
        Custom cross-validation for deep learning models.
        
        Parameters:
        -----------
        model : keras.Model or similar
            Deep learning model to evaluate
        X : numpy.ndarray
            The feature matrix
        y : numpy.ndarray
            The target variable
        cv : cv splitter
            Cross-validation strategy
        scoring : str, list, or dict
            Scoring metrics to evaluate
        **kwargs :
            Additional parameters
            
        Returns:
        --------
        dict
            Dictionary of cross-validation results
        """
        # Extract deep learning specific parameters
        batch_size = kwargs.get('batch_size', 32)
        epochs = kwargs.get('epochs', 100)
        verbose = kwargs.get('verbose', 0)
        validation_split = kwargs.get('validation_split', 0.2)
        callbacks = kwargs.get('callbacks', None)
        
        # Initialize results dictionary
        cv_results = {
            'test_score': [],
            'train_score': [] if kwargs.get('return_train_score', True) else None,
            'fit_time': [],
            'score_time': []
        }
        
        # Add keys for each scoring metric
        if isinstance(scoring, list):
            for metric in scoring:
                cv_results[f'test_{metric}'] = []
                if kwargs.get('return_train_score', True):
                    cv_results[f'train_{metric}'] = []
        elif isinstance(scoring, dict):
            for metric_name in scoring:
                cv_results[f'test_{metric_name}'] = []
                if kwargs.get('return_train_score', True):
                    cv_results[f'train_{metric_name}'] = []
        elif isinstance(scoring, str):
            cv_results[f'test_{scoring}'] = []
            if kwargs.get('return_train_score', True):
                cv_results[f'train_{scoring}'] = []
                
        # Add estimator storage if requested
        if kwargs.get('return_estimator', False):
            cv_results['estimator'] = []
        
        # Get the fold splitter
        splits = list(cv.split(X, y))
        
        # Run cross-validation
        for i, (train_idx, test_idx) in enumerate(splits):
            self.logger.info(f"Fold {i+1}/{len(splits)}")
            
            # Split the data for this fold
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Clone the model - for deep learning, this means recreating it
            try:
                # If model has a 'clone' method or property
                if hasattr(model, 'clone') and callable(getattr(model, 'clone')):
                    clone_model = model.clone()
                else:
                    # Try to use Keras clone_model function
                    from tensorflow.keras.models import clone_model
                    clone_config = model.get_config()
                    clone_model = clone_model(model)
                    clone_model.set_weights(model.get_weights())
                    clone_model.compile(
                        optimizer=model.optimizer, 
                        loss=model.loss,
                        metrics=model.metrics
                    )
            except:
                self.logger.warning("Could not clone model. Using original model reference.")
                clone_model = model
            
            # Fit the model
            start_time = time.time()
            history = clone_model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            fit_time = time.time() - start_time
            
            # Score the model
            start_time = time.time()
            
            # Get predictions for test set
            if self.task_type == 'classification':
                y_pred_prob = clone_model.predict(X_test)
                
                if y_pred_prob.shape[1] > 1:  # Multi-class
                    y_pred = np.argmax(y_pred_prob, axis=1)
                else:  # Binary
                    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
                    
                # Convert predictions to proper format for multi-class
                if len(np.unique(y)) > 2:
                    if len(y_test.shape) == 1 or y_test.shape[1] == 1:
                        # One-hot encode predictions for multi-class
                        n_classes = len(np.unique(y))
                        y_test_onehot = np.eye(n_classes)[y_test.astype(int)]
                    else:
                        y_test_onehot = y_test
                else:
                    y_test_onehot = y_test
                    
            else:  # Regression
                y_pred = clone_model.predict(X_test)
                if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                    y_pred = y_pred.flatten()
                    
            score_time = time.time() - start_time
            
            # Calculate scores for test set
            if self.task_type == 'classification':
                # Calculate classification metrics
                test_scores = {}
                
                if isinstance(scoring, list):
                    for metric in scoring:
                        test_scores[metric] = self._calculate_classification_metric(
                            metric, y_test, y_pred, y_pred_prob
                        )
                elif isinstance(scoring, dict):
                    for metric_name, metric_func in scoring.items():
                        test_scores[metric_name] = self._calculate_classification_metric(
                            metric_func, y_test, y_pred, y_pred_prob
                        )
                elif isinstance(scoring, str):
                    test_scores[scoring] = self._calculate_classification_metric(
                        scoring, y_test, y_pred, y_pred_prob
                    )
                
                # Add primary score
                if isinstance(scoring, list) and 'accuracy' in scoring:
                    test_scores['score'] = test_scores['accuracy']
                elif isinstance(scoring, list) and len(scoring) > 0:
                    test_scores['score'] = test_scores[scoring[0]]
                elif isinstance(scoring, dict) and len(scoring) > 0:
                    test_scores['score'] = test_scores[list(scoring.keys())[0]]
                elif isinstance(scoring, str):
                    test_scores['score'] = test_scores[scoring]
                else:
                    # Default to accuracy
                    test_scores['score'] = accuracy_score(y_test, y_pred)
                
            else:  # Regression
                # Calculate regression metrics
                test_scores = {}
                
                if isinstance(scoring, list):
                    for metric in scoring:
                        test_scores[metric] = self._calculate_regression_metric(
                            metric, y_test, y_pred
                        )
                elif isinstance(scoring, dict):
                    for metric_name, metric_func in scoring.items():
                        test_scores[metric_name] = self._calculate_regression_metric(
                            metric_func, y_test, y_pred
                        )
                elif isinstance(scoring, str):
                    test_scores[scoring] = self._calculate_regression_metric(
                        scoring, y_test, y_pred
                    )
                
                # Add primary score
                if isinstance(scoring, list) and 'r2' in scoring:
                    test_scores['score'] = test_scores['r2']
                elif isinstance(scoring, list) and len(scoring) > 0:
                    test_scores['score'] = test_scores[scoring[0]]
                elif isinstance(scoring, dict) and len(scoring) > 0:
                    test_scores['score'] = test_scores[list(scoring.keys())[0]]
                elif isinstance(scoring, str):
                    test_scores['score'] = test_scores[scoring]
                else:
                    # Default to r2
                    test_scores['score'] = r2_score(y_test, y_pred)
            
            # Store test scores in results
            for metric, value in test_scores.items():
                if metric == 'score':
                    cv_results['test_score'].append(value)
                else:
                    cv_results[f'test_{metric}'].append(value)
            
            # Calculate train scores if requested
            if kwargs.get('return_train_score', True):
                # Get predictions for train set
                if self.task_type == 'classification':
                    y_train_pred_prob = clone_model.predict(X_train)
                    
                    if y_train_pred_prob.shape[1] > 1:  # Multi-class
                        y_train_pred = np.argmax(y_train_pred_prob, axis=1)
                    else:  # Binary
                        y_train_pred = (y_train_pred_prob > 0.5).astype(int).flatten()
                        
                else:  # Regression
                    y_train_pred = clone_model.predict(X_train)
                    if len(y_train_pred.shape) > 1 and y_train_pred.shape[1] == 1:
                        y_train_pred = y_train_pred.flatten()
                
                # Calculate scores for train set
                if self.task_type == 'classification':
                    # Calculate classification metrics
                    train_scores = {}
                    
                    if isinstance(scoring, list):
                        for metric in scoring:
                            train_scores[metric] = self._calculate_classification_metric(
                                metric, y_train, y_train_pred, y_train_pred_prob
                            )
                    elif isinstance(scoring, dict):
                        for metric_name, metric_func in scoring.items():
                            train_scores[metric_name] = self._calculate_classification_metric(
                                metric_func, y_train, y_train_pred, y_train_pred_prob
                            )
                    elif isinstance(scoring, str):
                        train_scores[scoring] = self._calculate_classification_metric(
                            scoring, y_train, y_train_pred, y_train_pred_prob
                        )
                    
                    # Add primary score
                    if isinstance(scoring, list) and 'accuracy' in scoring:
                        train_scores['score'] = train_scores['accuracy']
                    elif isinstance(scoring, list) and len(scoring) > 0:
                        train_scores['score'] = train_scores[scoring[0]]
                    elif isinstance(scoring, dict) and len(scoring) > 0:
                        train_scores['score'] = train_scores[list(scoring.keys())[0]]
                    elif isinstance(scoring, str):
                        train_scores['score'] = train_scores[scoring]
                    else:
                        # Default to accuracy
                        train_scores['score'] = accuracy_score(y_train, y_train_pred)
                    
                else:  # Regression
                    # Calculate regression metrics
                    train_scores = {}
                    
                    if isinstance(scoring, list):
                        for metric in scoring:
                            train_scores[metric] = self._calculate_regression_metric(
                                metric, y_train, y_train_pred
                            )
                    elif isinstance(scoring, dict):
                        for metric_name, metric_func in scoring.items():
                            train_scores[metric_name] = self._calculate_regression_metric(
                                metric_func, y_train, y_train_pred
                            )
                    elif isinstance(scoring, str):
                        train_scores[scoring] = self._calculate_regression_metric(
                            scoring, y_train, y_train_pred
                        )
                    
                    # Add primary score
                    if isinstance(scoring, list) and 'r2' in scoring:
                        train_scores['score'] = train_scores['r2']
                    elif isinstance(scoring, list) and len(scoring) > 0:
                        train_scores['score'] = train_scores[scoring[0]]
                    elif isinstance(scoring, dict) and len(scoring) > 0:
                        train_scores['score'] = train_scores[list(scoring.keys())[0]]
                    elif isinstance(scoring, str):
                        train_scores['score'] = train_scores[scoring]
                    else:
                        # Default to r2
                        train_scores['score'] = r2_score(y_train, y_train_pred)
                
                # Store train scores in results
                for metric, value in train_scores.items():
                    if metric == 'score':
                        cv_results['train_score'].append(value)
                    else:
                        cv_results[f'train_{metric}'].append(value)
            
            # Store timings
            cv_results['fit_time'].append(fit_time)
            cv_results['score_time'].append(score_time)
            
            # Store estimator if requested
            if kwargs.get('return_estimator', False):
                cv_results['estimator'].append(clone_model)
        
        # Convert result lists to numpy arrays
        for key in cv_results:
            if cv_results[key] is not None:
                cv_results[key] = np.array(cv_results[key])
        
        return cv_results
    
    def _calculate_classification_metric(self, metric, y_true, y_pred, y_pred_proba=None):
        """Calculate a specific classification metric."""
        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            return precision_score(y_true, y_pred, average='binary')
        elif metric == 'precision_weighted':
            return precision_score(y_true, y_pred, average='weighted')
        elif metric == 'recall':
            return recall_score(y_true, y_pred, average='binary')
        elif metric == 'recall_weighted':
            return recall_score(y_true, y_pred, average='weighted')
        elif metric == 'f1':
            return f1_score(y_true, y_pred, average='binary')
        elif metric == 'f1_weighted':
            return f1_score(y_true, y_pred, average='weighted')
        elif metric == 'roc_auc' or metric == 'roc_auc_ovr_weighted':
            # Requires probability predictions
            if y_pred_proba is None:
                return np.nan
                
            try:
                if len(np.unique(y_true)) > 2:  # Multi-class
                    # Check shape of y_true and convert if needed
                    if len(y_true.shape) == 1 or y_true.shape[1] == 1:
                        # One-hot encode true labels for multi-class
                        n_classes = y_pred_proba.shape[1]
                        y_true_onehot = np.eye(n_classes)[y_true.astype(int)]
                    else:
                        y_true_onehot = y_true
                        
                    return roc_auc_score(y_true_onehot, y_pred_proba, multi_class='ovr', average='weighted')
                else:  # Binary
                    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                        # Use second column for positive class probability
                        return roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        return roc_auc_score(y_true, y_pred_proba)
            except:
                return np.nan
        elif callable(metric):
            # Custom metric function
            return metric(y_true, y_pred)
        else:
            # Unknown metric
            return np.nan
    
    def _calculate_regression_metric(self, metric, y_true, y_pred):
        """Calculate a specific regression metric."""
        if metric == 'mse' or metric == 'mean_squared_error':
            return mean_squared_error(y_true, y_pred)
        elif metric == 'neg_mean_squared_error':
            return -mean_squared_error(y_true, y_pred)
        elif metric == 'rmse' or metric == 'root_mean_squared_error':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'neg_root_mean_squared_error':
            return -np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'mae' or metric == 'mean_absolute_error':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'neg_mean_absolute_error':
            return -mean_absolute_error(y_true, y_pred)
        elif metric == 'r2':
            return r2_score(y_true, y_pred)
        elif metric == 'explained_variance':
            return explained_variance_score(y_true, y_pred)
        elif callable(metric):
            # Custom metric function
            return metric(y_true, y_pred)
        else:
            # Unknown metric
            return np.nan
    
    def _summarize_cv_results(self, cv_results):
        """Summarize cross-validation results with descriptive statistics."""
        summary = {}
        
        # Process each result key
        for key, values in cv_results.items():
            if key == 'estimator' or values is None:
                continue
                
            # Calculate summary statistics
            try:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
            except:
                # Skip metrics that can't be summarized
                pass
                
        return summary
    
    def nested_cross_validate(self, model, X, y, param_grid, outer_cv=5, inner_cv=3, 
                             search_method='grid', scoring=None, **kwargs):
        """
        Perform nested cross-validation for hyperparameter tuning and model evaluation.
        
        Parameters:
        -----------
        model : estimator object
            Model to evaluate
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        param_grid : dict or list of dicts
            Grid of parameters to search
        outer_cv : int or cv splitter
            Cross-validation strategy for outer loop
        inner_cv : int or cv splitter
            Cross-validation strategy for inner loop
        search_method : str
            Method for hyperparameter search ('grid', 'random', or 'bayesian')
        scoring : str or callable
            Scoring metric to optimize
        **kwargs :
            Additional parameters:
            - n_iter : int
                Number of iterations for random/Bayesian search
            - refit : bool
                Whether to refit the model with best parameters
            - n_jobs : int
                Number of parallel jobs
            - verbose : int
                Verbosity level
            
        Returns:
        --------
        dict
            Dictionary of nested cross-validation results
        """
        # Process input data
        X_processed, y_processed = self._preprocess_data(X, y)
        
        # Set default scoring based on task type if not specified
        if scoring is None:
            if self.task_type == 'classification':
                scoring = 'accuracy'
            elif self.task_type == 'regression':
                scoring = 'neg_mean_squared_error'
            else:  # clustering
                scoring = 'silhouette'
                # Register custom scorers for clustering
                scoring = self._register_clustering_scorer(scoring)
                
        # Extract additional parameters
        n_iter = kwargs.get('n_iter', 10)
        refit = kwargs.get('refit', True)
        n_jobs = kwargs.get('n_jobs', -1)
        verbose = kwargs.get('verbose', 0)
        
        # Set up outer CV
        if isinstance(outer_cv, int):
            if self.task_type == 'classification':
                outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=self.random_state)
            else:
                outer_cv_splitter = KFold(n_splits=outer_cv, shuffle=True, random_state=self.random_state)
        else:
            outer_cv_splitter = outer_cv
            
        # Set up inner CV
        if isinstance(inner_cv, int):
            if self.task_type == 'classification':
                inner_cv_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=self.random_state)
            else:
                inner_cv_splitter = KFold(n_splits=inner_cv, shuffle=True, random_state=self.random_state)
        else:
            inner_cv_splitter = inner_cv
            
        # Check if model is a deep learning model
        is_dl_model = self._is_deep_learning_model(model)
        
        # For deep learning models, use custom nested CV implementation
        if is_dl_model:
            return self._deep_learning_nested_cv(
                model, X_processed, y_processed, param_grid, 
                outer_cv_splitter, inner_cv_splitter, 
                search_method, scoring, **kwargs
            )
        
        # Initialize results
        outer_scores = []
        best_params_list = []
        best_estimators = []
        
        # Get the outer fold splits
        outer_splits = list(outer_cv_splitter.split(X_processed, y_processed))
        
        # Perform nested CV
        for i, (train_idx, test_idx) in enumerate(outer_splits):
            self.logger.info(f"Outer fold {i+1}/{len(outer_splits)}")
            
            # Split data for this outer fold
            X_train, X_test = X_processed[train_idx], X_processed[test_idx]
            y_train, y_test = y_processed[train_idx], y_processed[test_idx]
            
            # Create the hyperparameter search object
            if search_method.lower() == 'grid':
                search = GridSearchCV(
                    model, param_grid, cv=inner_cv_splitter, scoring=scoring,
                    refit=refit, n_jobs=n_jobs, verbose=verbose
                )
            elif search_method.lower() == 'random':
                search = RandomizedSearchCV(
                    model, param_grid, cv=inner_cv_splitter, scoring=scoring,
                    n_iter=n_iter, refit=refit, n_jobs=n_jobs, verbose=verbose,
                    random_state=self.random_state
                )
            elif search_method.lower() == 'bayesian':
                if not OPTUNA_AVAILABLE:
                    raise ImportError("Optuna not installed. Bayesian optimization unavailable.")
                    
                # For Bayesian optimization, we'll use Optuna
                # This requires a different approach
                best_params, best_score, study = self._bayesian_optimization(
                    model, X_train, y_train, param_grid, 
                    cv=inner_cv_splitter, scoring=scoring, 
                    n_trials=n_iter, verbose=verbose
                )
                
                # Create a clone of the model with best parameters
                from sklearn.base import clone
                best_model = clone(model)
                best_model.set_params(**best_params)
                
                # Fit the model with best parameters
                if refit:
                    best_model.fit(X_train, y_train)
                    
                # Create a mock search object to maintain consistent interface
                class MockSearch:
                    def __init__(self, best_model, best_params, best_score):
                        self.best_estimator_ = best_model
                        self.best_params_ = best_params
                        self.best_score_ = best_score
                        
                search = MockSearch(best_model, best_params, best_score)
            else:
                raise ValueError(f"Unknown search method: {search_method}")
                
            # Run the hyperparameter search if not Bayesian
            if search_method.lower() != 'bayesian':
                search.fit(X_train, y_train)
                
            # Evaluate the best model on the test set
            best_model = search.best_estimator_
            
            # For classification, calculate scores differently
            if self.task_type == 'classification':
                try:
                    # Try to get probability predictions
                    y_pred_proba = best_model.predict_proba(X_test)
                    has_proba = True
                except:
                    has_proba = False
                    
                # Get class predictions
                y_pred = best_model.predict(X_test)
                
                # Calculate appropriate metric
                if scoring == 'accuracy':
                    test_score = accuracy_score(y_test, y_pred)
                elif scoring == 'precision' or scoring == 'precision_weighted':
                    test_score = precision_score(y_test, y_pred, average='weighted')
                elif scoring == 'recall' or scoring == 'recall_weighted':
                    test_score = recall_score(y_test, y_pred, average='weighted')
                elif scoring == 'f1' or scoring == 'f1_weighted':
                    test_score = f1_score(y_test, y_pred, average='weighted')
                elif scoring == 'roc_auc' or scoring == 'roc_auc_ovr_weighted':
                    if has_proba:
                        # Check if multi-class
                        if len(np.unique(y_test)) > 2:
                            # One-hot encode for multi-class
                            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                            test_score = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='weighted')
                        else:
                            # Binary classification
                            test_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        # Fall back to accuracy if probabilities not available
                        test_score = accuracy_score(y_test, y_pred)
                elif callable(scoring):
                    # Custom scorer
                    test_score = scoring(best_model, X_test, y_test)
                else:
                    # Default to accuracy
                    test_score = accuracy_score(y_test, y_pred)
            elif self.task_type == 'regression':
                # Get predictions
                y_pred = best_model.predict(X_test)
                
                # Calculate appropriate metric
                if scoring == 'neg_mean_squared_error':
                    test_score = -mean_squared_error(y_test, y_pred)
                elif scoring == 'neg_root_mean_squared_error':
                    test_score = -np.sqrt(mean_squared_error(y_test, y_pred))
                elif scoring == 'neg_mean_absolute_error':
                    test_score = -mean_absolute_error(y_test, y_pred)
                elif scoring == 'r2':
                    test_score = r2_score(y_test, y_pred)
                elif scoring == 'explained_variance':
                    test_score = explained_variance_score(y_test, y_pred)
                elif callable(scoring):
                    # Custom scorer
                    test_score = scoring(best_model, X_test, y_test)
                else:
                    # Default to R²
                    test_score = r2_score(y_test, y_pred)
            else:  # clustering
                # For clustering, we typically evaluate on some internal metric
                # rather than a held-out test set
                # We'll calculate silhouette score as a default
                labels = best_model.predict(X_test)
                test_score = silhouette_score(X_test, labels)
                
            # Store results for this fold
            outer_scores.append(test_score)
            best_params_list.append(search.best_params_)
            best_estimators.append(best_model)
            
            self.logger.info(f"  Best parameters: {search.best_params_}")
            self.logger.info(f"  Test score: {test_score:.4f}")
            
        # Summarize results
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        
        self.logger.info(f"Nested CV completed. Mean score: {mean_score:.4f} ± {std_score:.4f}")
        
        # Find the best overall model
        best_idx = np.argmax(outer_scores)
        best_estimator = best_estimators[best_idx]
        best_params = best_params_list[best_idx]
        best_score = outer_scores[best_idx]
        
        # Store results
        results = {
            'outer_scores': outer_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'best_score': best_score,
            'best_params': best_params,
            'best_estimator': best_estimator,
            'all_best_params': best_params_list
        }
        
        # Store results in instance variables
        self.best_estimator_ = best_estimator
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        return results
    
    def _deep_learning_nested_cv(self, model, X, y, param_grid, outer_cv, inner_cv, 
                                search_method, scoring, **kwargs):
        """
        Custom nested cross-validation for deep learning models.
        
        Much of the code is similar to the standard nested CV, but handles
        deep learning models specifically.
        """
        # Extract additional parameters
        n_iter = kwargs.get('n_iter', 10)
        batch_size = kwargs.get('batch_size', 32)
        epochs = kwargs.get('epochs', 100)
        verbose = kwargs.get('verbose', 0)
        
        # Initialize results
        outer_scores = []
        best_params_list = []
        best_estimators = []
        
        # Get the outer fold splits
        outer_splits = list(outer_cv.split(X, y))
        
        # Perform nested CV
        for i, (train_idx, test_idx) in enumerate(outer_splits):
            self.logger.info(f"Outer fold {i+1}/{len(outer_splits)}")
            
            # Split data for this outer fold
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # For inner loop, we'll implement the search manually
            if search_method.lower() == 'grid':
                # Create a grid of parameters
                param_combinations = list(self._param_grid_to_dict(param_grid))
                self.logger.info(f"  Grid search with {len(param_combinations)} combinations")
            elif search_method.lower() == 'random':
                # Sample random combinations
                param_combinations = list(self._random_param_grid(param_grid, n_iter))
                self.logger.info(f"  Random search with {len(param_combinations)} combinations")
            elif search_method.lower() == 'bayesian':
                if not OPTUNA_AVAILABLE:
                    raise ImportError("Optuna not installed. Bayesian optimization unavailable.")
                    
                # For Bayesian optimization, we'll use Optuna
                best_params, best_score, study = self._bayesian_optimization_dl(
                    model, X_train, y_train, param_grid, 
                    cv=inner_cv, scoring=scoring, 
                    n_trials=n_iter, epochs=epochs, batch_size=batch_size, verbose=verbose
                )
                
                # Create a new model with best parameters
                best_model = self._create_dl_model_with_params(model, best_params)
                
                # Fit the model with best parameters
                self.logger.info(f"  Training best model on outer fold {i+1}")
                best_model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose
                )
                
                # Store results
                outer_scores.append(best_score)
                best_params_list.append(best_params)
                best_estimators.append(best_model)
                
                # Skip to next fold
                continue
                
            # Run inner cross-validation for grid/random search
            inner_scores = []
            inner_splits = list(inner_cv.split(X_train, y_train))
            
            # Evaluate each parameter combination
            for params in param_combinations:
                fold_scores = []
                
                # Create a new model with these parameters
                fold_model = self._create_dl_model_with_params(model, params)
                
                # Evaluate on each inner fold
                for j, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
                    # Get the inner train/val split
                    X_inner_train = X_train[inner_train_idx]
                    y_inner_train = y_train[inner_train_idx]
                    X_inner_val = X_train[inner_val_idx]
                    y_inner_val = y_train[inner_val_idx]
                    
                    # Train the model
                    fold_model.fit(
                        X_inner_train, y_inner_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_inner_val, y_inner_val),
                        verbose=0
                    )
                    
                    # Evaluate on validation data
                    if self.task_type == 'classification':
                        # Get predictions
                        y_val_pred_proba = fold_model.predict(X_inner_val)
                        
                        if y_val_pred_proba.shape[1] > 1:  # Multi-class
                            y_val_pred = np.argmax(y_val_pred_proba, axis=1)
                        else:  # Binary
                            y_val_pred = (y_val_pred_proba > 0.5).astype(int).flatten()
                            
                        # Calculate score based on metric
                        if scoring == 'accuracy':
                            score = accuracy_score(y_inner_val, y_val_pred)
                        elif scoring == 'precision' or scoring == 'precision_weighted':
                            score = precision_score(y_inner_val, y_val_pred, average='weighted')
                        elif scoring == 'recall' or scoring == 'recall_weighted':
                            score = recall_score(y_inner_val, y_val_pred, average='weighted')
                        elif scoring == 'f1' or scoring == 'f1_weighted':
                            score = f1_score(y_inner_val, y_val_pred, average='weighted')
                        elif scoring == 'roc_auc' or scoring == 'roc_auc_ovr_weighted':
                            # For multi-class, need to binarize labels
                            if len(np.unique(y_inner_val)) > 2:
                                y_inner_val_bin = label_binarize(y_inner_val, classes=np.unique(y_inner_val))
                                score = roc_auc_score(y_inner_val_bin, y_val_pred_proba, multi_class='ovr', average='weighted')
                            else:
                                if len(y_val_pred_proba.shape) > 1 and y_val_pred_proba.shape[1] > 1:
                                    score = roc_auc_score(y_inner_val, y_val_pred_proba[:, 1])
                                else:
                                    score = roc_auc_score(y_inner_val, y_val_pred_proba)
                        else:
                            # Default to accuracy
                            score = accuracy_score(y_inner_val, y_val_pred)
                    else:  # Regression
                        # Get predictions
                        y_val_pred = fold_model.predict(X_inner_val)
                        if len(y_val_pred.shape) > 1:
                            y_val_pred = y_val_pred.flatten()
                            
                        # Calculate score based on metric
                        if scoring == 'neg_mean_squared_error':
                            score = -mean_squared_error(y_inner_val, y_val_pred)
                        elif scoring == 'neg_root_mean_squared_error':
                            score = -np.sqrt(mean_squared_error(y_inner_val, y_val_pred))
                        elif scoring == 'neg_mean_absolute_error':
                            score = -mean_absolute_error(y_inner_val, y_val_pred)
                        elif scoring == 'r2':
                            score = r2_score(y_inner_val, y_val_pred)
                        elif scoring == 'explained_variance':
                            score = explained_variance_score(y_inner_val, y_val_pred)
                        else:
                            # Default to R²
                            score = r2_score(y_inner_val, y_val_pred)
                            
                    # Store the score
                    fold_scores.append(score)
                
                # Calculate mean score across inner folds
                mean_fold_score = np.mean(fold_scores)
                inner_scores.append((mean_fold_score, params))
                
                self.logger.info(f"  Params: {params}, Mean score: {mean_fold_score:.4f}")
                
            # Find best parameters from inner CV
            best_inner_score, best_params = max(inner_scores, key=lambda x: x[0])
            self.logger.info(f"  Best inner parameters: {best_params}, Score: {best_inner_score:.4f}")
            
            # Train final model with best parameters on all training data
            best_model = self._create_dl_model_with_params(model, best_params)
            best_model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose
            )
            
            # Evaluate on test set
            if self.task_type == 'classification':
                # Get predictions
                y_test_pred_proba = best_model.predict(X_test)
                
                if y_test_pred_proba.shape[1] > 1:  # Multi-class
                    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
                else:  # Binary
                    y_test_pred = (y_test_pred_proba > 0.5).astype(int).flatten()
                    
                # Calculate score based on metric
                if scoring == 'accuracy':
                    test_score = accuracy_score(y_test, y_test_pred)
                elif scoring == 'precision' or scoring == 'precision_weighted':
                    test_score = precision_score(y_test, y_test_pred, average='weighted')
                elif scoring == 'recall' or scoring == 'recall_weighted':
                    test_score = recall_score(y_test, y_test_pred, average='weighted')
                elif scoring == 'f1' or scoring == 'f1_weighted':
                    test_score = f1_score(y_test, y_test_pred, average='weighted')
                elif scoring == 'roc_auc' or scoring == 'roc_auc_ovr_weighted':
                    # For multi-class, need to binarize labels
                    if len(np.unique(y_test)) > 2:
                        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                        test_score = roc_auc_score(y_test_bin, y_test_pred_proba, multi_class='ovr', average='weighted')
                    else:
                        if len(y_test_pred_proba.shape) > 1 and y_test_pred_proba.shape[1] > 1:
                            test_score = roc_auc_score(y_test, y_test_pred_proba[:, 1])
                        else:
                            test_score = roc_auc_score(y_test, y_test_pred_proba)
                else:
                    # Default to accuracy
                    test_score = accuracy_score(y_test, y_test_pred)
            else:  # Regression
                # Get predictions
                y_test_pred = best_model.predict(X_test)
                if len(y_test_pred.shape) > 1:
                    y_test_pred = y_test_pred.flatten()
                    
                # Calculate score based on metric
                if scoring == 'neg_mean_squared_error':
                    test_score = -mean_squared_error(y_test, y_test_pred)
                elif scoring == 'neg_root_mean_squared_error':
                    test_score = -np.sqrt(mean_squared_error(y_test, y_test_pred))
                elif scoring == 'neg_mean_absolute_error':
                    test_score = -mean_absolute_error(y_test, y_test_pred)
                elif scoring == 'r2':
                    test_score = r2_score(y_test, y_test_pred)
                elif scoring == 'explained_variance':
                    test_score = explained_variance_score(y_test, y_test_pred)
                else:
                    # Default to R²
                    test_score = r2_score(y_test, y_test_pred)
                    
            # Store results for this fold
            outer_scores.append(test_score)
            best_params_list.append(best_params)
            best_estimators.append(best_model)
            
            self.logger.info(f"  Outer test score: {test_score:.4f}")
            
        # Summarize results
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        
        self.logger.info(f"Nested CV completed. Mean score: {mean_score:.4f} ± {std_score:.4f}")
        
        # Find the best overall model
        best_idx = np.argmax(outer_scores)
        best_estimator = best_estimators[best_idx]
        best_params = best_params_list[best_idx]
        best_score = outer_scores[best_idx]
        
        # Store results
        results = {
            'outer_scores': outer_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'best_score': best_score,
            'best_params': best_params,
            'best_estimator': best_estimator,
            'all_best_params': best_params_list
        }
        
        # Store results in instance variables
        self.best_estimator_ = best_estimator
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        return results
    
    def _create_dl_model_with_params(self, base_model, params):
        """
        Create a deep learning model with specific parameters.
        
        Parameters:
        -----------
        base_model : keras.Model or similar
            Base model to use as a template
        params : dict
            Parameters to apply to the model
            
        Returns:
        --------
        keras.Model
            A new model with the specified parameters
        """
        # Try different approaches depending on the model type
        try:
            # If model has a clone method
            if hasattr(base_model, 'clone') and callable(getattr(base_model, 'clone')):
                new_model = base_model.clone()
                # Apply parameters
                for param, value in params.items():
                    if hasattr(new_model, param):
                        setattr(new_model, param, value)
                return new_model
            
            # Try to use Keras approach
            try:
                from tensorflow.keras.models import clone_model, model_from_json
                
                # If it's a keras model
                if hasattr(base_model, 'to_json'):
                    # Clone the architecture
                    model_json = base_model.to_json()
                    new_model = model_from_json(model_json)
                    
                    # Compile with parameters
                    optimizer = params.get('optimizer', 'adam')
                    loss = params.get('loss', 'binary_crossentropy' if base_model.output_shape[-1] == 1 else 'categorical_crossentropy')
                    metrics = params.get('metrics', ['accuracy'])
                    
                    new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                    
                    return new_model
            except:
                pass
                
            # If it has a build method, assume it's a custom model builder
            if hasattr(base_model, 'build') and callable(getattr(base_model, 'build')):
                return base_model.build(**params)
                
            # Last resort: try to recreate from scratch assuming base_model is a function
            if callable(base_model):
                return base_model(**params)
                
            # If all else fails, return the base model (not ideal)
            self.logger.warning("Could not create a new model with parameters. Using base model.")
            return base_model
            
        except Exception as e:
            self.logger.error(f"Error creating model with parameters: {str(e)}")
            self.logger.warning("Using base model as fallback.")
            return base_model
    
    def optimize_hyperparameters(self, model, X, y, param_grid, search_method='grid', cv=5,
                               scoring=None, **kwargs):
        """
        Perform hyperparameter optimization.
        
        Parameters:
        -----------
        model : estimator object
            Model to optimize
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        param_grid : dict or list of dicts
            Grid of parameters to search
        search_method : str
            Method for hyperparameter search ('grid', 'random', or 'bayesian')
        cv : int or cv splitter
            Cross-validation strategy
        scoring : str or callable
            Scoring metric to optimize
        **kwargs :
            Additional parameters:
            - n_iter : int
                Number of iterations for random/Bayesian search
            - refit : bool
                Whether to refit the model with best parameters
            - n_jobs : int
                Number of parallel jobs
            - verbose : int
                Verbosity level
            
        Returns:
        --------
        dict
            Dictionary of optimization results
        """
        # Process input data
        X_processed, y_processed = self._preprocess_data(X, y)
        
        # Set default scoring based on task type if not specified
        if scoring is None:
            if self.task_type == 'classification':
                scoring = 'accuracy'
            elif self.task_type == 'regression':
                scoring = 'neg_mean_squared_error'
            else:  # clustering
                scoring = 'silhouette'
                scoring = self._register_clustering_scorer(scoring)
                
        # Extract additional parameters
        n_iter = kwargs.get('n_iter', 10)
        refit = kwargs.get('refit', True)
        n_jobs = kwargs.get('n_jobs', -1)
        verbose = kwargs.get('verbose', 0)
        
        # Set up CV strategy if it's an integer
        if isinstance(cv, int):
            if self.task_type == 'classification':
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            else:
                cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            cv_splitter = cv
            
        # Check if model is a deep learning model
        is_dl_model = self._is_deep_learning_model(model)
        
        # For deep learning models, use custom optimization
        if is_dl_model:
            if search_method.lower() == 'grid':
                result = self._grid_search_dl(
                    model, X_processed, y_processed, param_grid, 
                    cv=cv_splitter, scoring=scoring, **kwargs
                )
            elif search_method.lower() == 'random':
                result = self._random_search_dl(
                    model, X_processed, y_processed, param_grid, 
                    cv=cv_splitter, scoring=scoring, n_iter=n_iter, **kwargs
                )
            elif search_method.lower() == 'bayesian':
                if not OPTUNA_AVAILABLE:
                    raise ImportError("Optuna not installed. Bayesian optimization unavailable.")
                    
                result = self._bayesian_optimization_dl(
                    model, X_processed, y_processed, param_grid, 
                    cv=cv_splitter, scoring=scoring, n_trials=n_iter, **kwargs
                )
            else:
                raise ValueError(f"Unknown search method: {search_method}")
                
            # Store results
            self.best_params_ = result[0]
            self.best_score_ = result[1]
            
            if len(result) > 2:
                # For Bayesian optimization, we also have the study
                self.optimization_results_ = {
                    'best_params': result[0],
                    'best_score': result[1],
                    'study': result[2] if len(result) > 2 else None,
                    'search_method': search_method
                }
            else:
                self.optimization_results_ = {
                    'best_params': result[0],
                    'best_score': result[1],
                    'all_params': result[2] if len(result) > 2 else None,
                    'all_scores': result[3] if len(result) > 3 else None,
                    'search_method': search_method
                }
                
            return self.optimization_results_
        
        # For traditional models, use sklearn's search methods
        try:
            if search_method.lower() == 'grid':
                search = GridSearchCV(
                    model, param_grid, cv=cv_splitter, scoring=scoring,
                    refit=refit, n_jobs=n_jobs, verbose=verbose,
                    return_train_score=True
                )
            elif search_method.lower() == 'random':
                search = RandomizedSearchCV(
                    model, param_grid, cv=cv_splitter, scoring=scoring,
                    n_iter=n_iter, refit=refit, n_jobs=n_jobs, verbose=verbose,
                    random_state=self.random_state, return_train_score=True
                )
            elif search_method.lower() == 'bayesian':
                if not OPTUNA_AVAILABLE:
                    raise ImportError("Optuna not installed. Bayesian optimization unavailable.")
                    
                # For Bayesian optimization, we'll use Optuna
                best_params, best_score, study = self._bayesian_optimization(
                    model, X_processed, y_processed, param_grid, 
                    cv=cv_splitter, scoring=scoring, 
                    n_trials=n_iter, verbose=verbose
                )
                
                # Create a basic search results dictionary to maintain consistent interface
                search_results = {
                    'best_params_': best_params,
                    'best_score_': best_score,
                    'best_estimator_': None,  # Will set this later if refit=True
                }
                
                # If refit is requested, fit a model with the best parameters
                if refit:
                    from sklearn.base import clone
                    best_model = clone(model)
                    best_model.set_params(**best_params)
                    best_model.fit(X_processed, y_processed)
                    search_results['best_estimator_'] = best_model
                    
                # Create a mock search object to maintain consistent interface
                class MockSearch:
                    def __init__(self, results):
                        self.__dict__.update(results)
                        
                search = MockSearch(search_results)
                
                # Store Optuna study for later visualization
                self.optimization_results_ = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'study': study,
                    'search_method': search_method
                }
                
                # Store best params and score
                self.best_params_ = best_params
                self.best_score_ = best_score
                self.best_estimator_ = search.best_estimator_ if hasattr(search, 'best_estimator_') else None
                
                return self.optimization_results_
            else:
                raise ValueError(f"Unknown search method: {search_method}")
                
            # Run the search
            search.fit(X_processed, y_processed)
            
            # Create results dictionary
            results = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_,
                'search_method': search_method
            }
            
            # Store best estimator if available
            if refit:
                results['best_estimator'] = search.best_estimator_
                self.best_estimator_ = search.best_estimator_
                
            # Log results
            self.logger.info(f"Best parameters: {search.best_params_}")
            self.logger.info(f"Best score: {search.best_score_:.4f}")
            
            # Store results in instance variables
            self.best_params_ = search.best_params_
            self.best_score_ = search.best_score_
            self.optimization_results_ = results
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            raise
    
    def _bayesian_optimization(self, model, X, y, param_space, cv, scoring, n_trials=100, verbose=0):
        """
        Perform Bayesian optimization using Optuna.
        
        Parameters:
        -----------
        model : estimator object
            Model to optimize
        X : numpy.ndarray
            The feature matrix
        y : numpy.ndarray
            The target variable
        param_space : dict
            Dictionary of parameter spaces
        cv : cv splitter
            Cross-validation strategy
        scoring : str or callable
            Scoring metric to optimize
        n_trials : int
            Number of optimization trials
        verbose : int
            Verbosity level
            
        Returns:
        --------
        tuple
            (best_params, best_score, study)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Install with 'pip install optuna'")
            
        # Create a study
        study = optuna.create_study(direction='maximize')
        
        # Define the objective function
        def objective(trial):
            # Convert parameter space to Optuna parameters
            params = {}
            for param_name, param_specs in param_space.items():
                if isinstance(param_specs, list):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(param_name, param_specs)
                elif isinstance(param_specs, dict):
                    # Complex parameter specification
                    param_type = param_specs.get('type', 'categorical')
                    
                    if param_type == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_specs['values']
                        )
                    elif param_type == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, 
                            param_specs['low'], 
                            param_specs['high'],
                            step=param_specs.get('step', 1)
                        )
                    elif param_type == 'float':
                        if param_specs.get('log', False):
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_specs['low'],
                                param_specs['high'],
                                log=True
                            )
                        else:
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_specs['low'],
                                param_specs['high'],
                                step=param_specs.get('step', None)
                            )
                elif isinstance(param_specs, tuple) and len(param_specs) == 2:
                    # Simple range specification (low, high)
                    if isinstance(param_specs[0], int) and isinstance(param_specs[1], int):
                        params[param_name] = trial.suggest_int(
                            param_name, param_specs[0], param_specs[1]
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_specs[0], param_specs[1]
                        )
                else:
                    raise ValueError(f"Unsupported parameter specification for {param_name}")
            
            # Create a clone of the model with the sampled parameters
            from sklearn.base import clone
            model_clone = clone(model)
            model_clone.set_params(**params)
            
            # Perform cross-validation
            try:
                cv_scores = cross_val_score(
                    model_clone, X, y, cv=cv, scoring=scoring
                )
                avg_score = np.mean(cv_scores)
            except Exception as e:
                self.logger.warning(f"Error in trial {trial.number}: {str(e)}")
                return float('-inf')
                
            return avg_score
            
        # Optimize
        if verbose > 0:
            self.logger.info(f"Starting Bayesian optimization with {n_trials} trials")
            
        study.optimize(objective, n_trials=n_trials, show_progress_bar=(verbose > 0))
        
        # Get best parameters and score
        best_params = study.best_params
        best_score = study.best_value
        
        if verbose > 0:
            self.logger.info(f"Best parameters: {best_params}")
            self.logger.info(f"Best score: {best_score:.4f}")
            
        return best_params, best_score, study
    
    def _grid_search_dl(self, model, X, y, param_grid, cv, scoring, **kwargs):
        """
        Perform grid search for deep learning models.
        
        Parameters:
        -----------
        model : deep learning model
            Model to optimize
        X : numpy.ndarray
            The feature matrix
        y : numpy.ndarray
            The target variable
        param_grid : dict
            Dictionary of parameter grid
        cv : cv splitter
            Cross-validation strategy
        scoring : str or callable
            Scoring metric to optimize
        **kwargs :
            Additional parameters
            
        Returns:
        --------
        tuple
            (best_params, best_score, all_params, all_scores)
        """
        # Extract deep learning specific parameters
        batch_size = kwargs.get('batch_size', 32)
        epochs = kwargs.get('epochs', 100)
        verbose = kwargs.get('verbose', 0)
        
        # Generate all parameter combinations
        param_combinations = list(self._param_grid_to_dict(param_grid))
        self.logger.info(f"Grid search with {len(param_combinations)} combinations")
        
        # Store all results
        all_params = []
        all_scores = []
        
        # Get CV splits
        cv_splits = list(cv.split(X, y))
        
        # Evaluate each parameter combination
        for i, params in enumerate(param_combinations):
            self.logger.info(f"Evaluating combination {i+1}/{len(param_combinations)}: {params}")
            fold_scores = []
            
            # Evaluate on each fold
            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create model with these parameters
                model_clone = self._create_dl_model_with_params(model, params)
                
                # Train the model
                model_clone.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0
                )
                
                # Evaluate on validation set
                if self.task_type == 'classification':
                    y_pred_proba = model_clone.predict(X_val)
                    
                    if y_pred_proba.shape[1] > 1:  # Multi-class
                        y_pred = np.argmax(y_pred_proba, axis=1)
                    else:  # Binary
                        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                        
                    # Calculate appropriate metric
                    if scoring == 'accuracy':
                        score = accuracy_score(y_val, y_pred)
                    elif scoring == 'precision' or scoring == 'precision_weighted':
                        score = precision_score(y_val, y_pred, average='weighted')
                    elif scoring == 'recall' or scoring == 'recall_weighted':
                        score = recall_score(y_val, y_pred, average='weighted')
                    elif scoring == 'f1' or scoring == 'f1_weighted':
                        score = f1_score(y_val, y_pred, average='weighted')
                    elif scoring == 'roc_auc' or scoring == 'roc_auc_ovr_weighted':
                        # For multi-class, need to binarize labels
                        if len(np.unique(y_val)) > 2:
                            y_val_bin = label_binarize(y_val, classes=np.unique(y_val))
                            score = roc_auc_score(y_val_bin, y_pred_proba, multi_class='ovr', average='weighted')
                        else:
                            # Binary classification
                            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                                score = roc_auc_score(y_val, y_pred_proba[:, 1])
                            else:
                                score = roc_auc_score(y_val, y_pred_proba)
                    else:
                        # Default to accuracy
                        score = accuracy_score(y_val, y_pred)
                else:
                    # Regression
                    y_pred = model_clone.predict(X_val)
                    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                        y_pred = y_pred.flatten()
                        
                    # Calculate appropriate metric
                    if scoring == 'neg_mean_squared_error':
                        score = -mean_squared_error(y_val, y_pred)
                    elif scoring == 'neg_root_mean_squared_error':
                        score = -np.sqrt(mean_squared_error(y_val, y_pred))
                    elif scoring == 'neg_mean_absolute_error':
                        score = -mean_absolute_error(y_val, y_pred)
                    elif scoring == 'r2':
                        score = r2_score(y_val, y_pred)
                    elif scoring == 'explained_variance':
                        score = explained_variance_score(y_val, y_pred)
                    else:
                        # Default to R²
                        score = r2_score(y_val, y_pred)
                
                fold_scores.append(score)
                
            # Calculate mean score across folds
            mean_score = np.mean(fold_scores)
            
            # Store results
            all_params.append(params)
            all_scores.append(mean_score)
            
            self.logger.info(f"  Mean score: {mean_score:.4f}")
            
        # Find best parameters
        best_idx = np.argmax(all_scores)
        best_params = all_params[best_idx]
        best_score = all_scores[best_idx]
        
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best score: {best_score:.4f}")
        
        return best_params, best_score, all_params, all_scores
    
    def _random_search_dl(self, model, X, y, param_grid, cv, scoring, n_iter=10, **kwargs):
        """
        Perform random search for deep learning models.
        
        Parameters:
        -----------
        model : deep learning model
            Model to optimize
        X : numpy.ndarray
            The feature matrix
        y : numpy.ndarray
            The target variable
        param_grid : dict
            Dictionary of parameter grid
        cv : cv splitter
            Cross-validation strategy
        scoring : str or callable
            Scoring metric to optimize
        n_iter : int
            Number of random combinations to try
        **kwargs :
            Additional parameters
            
        Returns:
        --------
        tuple
            (best_params, best_score, all_params, all_scores)
        """
        # Extract deep learning specific parameters
        batch_size = kwargs.get('batch_size', 32)
        epochs = kwargs.get('epochs', 100)
        verbose = kwargs.get('verbose', 0)
        
        # Generate random parameter combinations
        param_combinations = list(self._random_param_grid(param_grid, n_iter))
        self.logger.info(f"Random search with {len(param_combinations)} combinations")
        
        # Store all results
        all_params = []
        all_scores = []
        
        # Get CV splits
        cv_splits = list(cv.split(X, y))
        
        # Evaluate each parameter combination
        for i, params in enumerate(param_combinations):
            self.logger.info(f"Evaluating combination {i+1}/{len(param_combinations)}: {params}")
            fold_scores = []
            
            # Evaluate on each fold
            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create model with these parameters
                model_clone = self._create_dl_model_with_params(model, params)
                
                # Train the model
                model_clone.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0
                )
                
                # Evaluate on validation set
                if self.task_type == 'classification':
                    y_pred_proba = model_clone.predict(X_val)
                    
                    if y_pred_proba.shape[1] > 1:  # Multi-class
                        y_pred = np.argmax(y_pred_proba, axis=1)
                    else:  # Binary
                        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                        
                    # Calculate appropriate metric
                    if scoring == 'accuracy':
                        score = accuracy_score(y_val, y_pred)
                    elif scoring == 'precision' or scoring == 'precision_weighted':
                        score = precision_score(y_val, y_pred, average='weighted')
                    elif scoring == 'recall' or scoring == 'recall_weighted':
                        score = recall_score(y_val, y_pred, average='weighted')
                    elif scoring == 'f1' or scoring == 'f1_weighted':
                        score = f1_score(y_val, y_pred, average='weighted')
                    elif scoring == 'roc_auc' or scoring == 'roc_auc_ovr_weighted':
                        # For multi-class, need to binarize labels
                        if len(np.unique(y_val)) > 2:
                            y_val_bin = label_binarize(y_val, classes=np.unique(y_val))
                            score = roc_auc_score(y_val_bin, y_pred_proba, multi_class='ovr', average='weighted')
                        else:
                            # Binary classification
                            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                                score = roc_auc_score(y_val, y_pred_proba[:, 1])
                            else:
                                score = roc_auc_score(y_val, y_pred_proba)
                    else:
                        # Default to accuracy
                        score = accuracy_score(y_val, y_pred)
                else:
                    # Regression
                    y_pred = model_clone.predict(X_val)
                    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                        y_pred = y_pred.flatten()
                        
                    # Calculate appropriate metric
                    if scoring == 'neg_mean_squared_error':
                        score = -mean_squared_error(y_val, y_pred)
                    elif scoring == 'neg_root_mean_squared_error':
                        score = -np.sqrt(mean_squared_error(y_val, y_pred))
                    elif scoring == 'neg_mean_absolute_error':
                        score = -mean_absolute_error(y_val, y_pred)
                    elif scoring == 'r2':
                        score = r2_score(y_val, y_pred)
                    elif scoring == 'explained_variance':
                        score = explained_variance_score(y_val, y_pred)
                    else:
                        # Default to R²
                        score = r2_score(y_val, y_pred)
                
                fold_scores.append(score)
                
            # Calculate mean score across folds
            mean_score = np.mean(fold_scores)
            
            # Store results
            all_params.append(params)
            all_scores.append(mean_score)
            
            self.logger.info(f"  Mean score: {mean_score:.4f}")
            
        # Find best parameters
        best_idx = np.argmax(all_scores)
        best_params = all_params[best_idx]
        best_score = all_scores[best_idx]
        
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best score: {best_score:.4f}")
        
        return best_params, best_score, all_params, all_scores
    
    def _bayesian_optimization_dl(self, model, X, y, param_space, cv, scoring, n_trials=100, **kwargs):
        """
        Perform Bayesian optimization for deep learning models using Optuna.
        
        Parameters:
        -----------
        model : deep learning model
            Model to optimize
        X : numpy.ndarray
            The feature matrix
        y : numpy.ndarray
            The target variable
        param_space : dict
            Dictionary of parameter spaces
        cv : cv splitter
            Cross-validation strategy
        scoring : str or callable
            Scoring metric to optimize
        n_trials : int
            Number of optimization trials
        **kwargs :
            Additional parameters
            
        Returns:
        --------
        tuple
            (best_params, best_score, study)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Install with 'pip install optuna'")
            
        # Extract deep learning specific parameters
        batch_size = kwargs.get('batch_size', 32)
        epochs = kwargs.get('epochs', 100)
        verbose = kwargs.get('verbose', 0)
        
        # Create a study
        study = optuna.create_study(direction='maximize')
        
        # Get CV splits
        cv_splits = list(cv.split(X, y))
        
        # Define the objective function
        def objective(trial):
            # Convert parameter space to Optuna parameters
            params = {}
            for param_name, param_specs in param_space.items():
                if isinstance(param_specs, list):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(param_name, param_specs)
                elif isinstance(param_specs, dict):
                    # Complex parameter specification
                    param_type = param_specs.get('type', 'categorical')
                    
                    if param_type == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_specs['values']
                        )
                    elif param_type == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, 
                            param_specs['low'], 
                            param_specs['high'],
                            step=param_specs.get('step', 1)
                        )
                    elif param_type == 'float':
                        if param_specs.get('log', False):
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_specs['low'],
                                param_specs['high'],
                                log=True
                            )
                        else:
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_specs['low'],
                                param_specs['high'],
                                step=param_specs.get('step', None)
                            )
                elif isinstance(param_specs, tuple) and len(param_specs) == 2:
                    # Simple range specification (low, high)
                    if isinstance(param_specs[0], int) and isinstance(param_specs[1], int):
                        params[param_name] = trial.suggest_int(
                            param_name, param_specs[0], param_specs[1]
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_specs[0], param_specs[1]
                        )
                else:
                    raise ValueError(f"Unsupported parameter specification for {param_name}")
            
            # Evaluate the parameter combination using CV
            fold_scores = []
            
            # Evaluate on each fold
            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                try:
                    # Create model with these parameters
                    model_clone = self._create_dl_model_with_params(model, params)
                    
                    # Train the model
                    model_clone.fit(
                        X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0
                    )
                    
                    # Evaluate on validation set
                    if self.task_type == 'classification':
                        y_pred_proba = model_clone.predict(X_val)
                        
                        if y_pred_proba.shape[1] > 1:  # Multi-class
                            y_pred = np.argmax(y_pred_proba, axis=1)
                        else:  # Binary
                            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                            
                        # Calculate appropriate metric
                        if scoring == 'accuracy':
                            score = accuracy_score(y_val, y_pred)
                        elif scoring == 'precision' or scoring == 'precision_weighted':
                            score = precision_score(y_val, y_pred, average='weighted')
                        elif scoring == 'recall' or scoring == 'recall_weighted':
                            score = recall_score(y_val, y_pred, average='weighted')
                        elif scoring == 'f1' or scoring == 'f1_weighted':
                            score = f1_score(y_val, y_pred, average='weighted')
                        elif scoring == 'roc_auc' or scoring == 'roc_auc_ovr_weighted':
                            # For multi-class, need to binarize labels
                            if len(np.unique(y_val)) > 2:
                                y_val_bin = label_binarize(y_val, classes=np.unique(y_val))
                                score = roc_auc_score(y_val_bin, y_pred_proba, multi_class='ovr', average='weighted')
                            else:
                                # Binary classification
                                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                                    score = roc_auc_score(y_val, y_pred_proba[:, 1])
                                else:
                                    score = roc_auc_score(y_val, y_pred_proba)
                        else:
                            # Default to accuracy
                            score = accuracy_score(y_val, y_pred)
                    else:
                        # Regression
                        y_pred = model_clone.predict(X_val)
                        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                            y_pred = y_pred.flatten()
                            
                        # Calculate appropriate metric
                        if scoring == 'neg_mean_squared_error':
                            score = -mean_squared_error(y_val, y_pred)
                        elif scoring == 'neg_root_mean_squared_error':
                            score = -np.sqrt(mean_squared_error(y_val, y_pred))
                        elif scoring == 'neg_mean_absolute_error':
                            score = -mean_absolute_error(y_val, y_pred)
                        elif scoring == 'r2':
                            score = r2_score(y_val, y_pred)
                        elif scoring == 'explained_variance':
                            score = explained_variance_score(y_val, y_pred)
                        else:
                            # Default to R²
                            score = r2_score(y_val, y_pred)
                except Exception as e:
                    self.logger.warning(f"Error in trial {trial.number}, fold {fold}: {str(e)}")
                    return float('-inf')
                
                fold_scores.append(score)
                
            # Calculate mean score across folds
            mean_score = np.mean(fold_scores)
            
            return mean_score
            
        # Optimize
        if verbose > 0:
            self.logger.info(f"Starting Bayesian optimization with {n_trials} trials")
            
        study.optimize(objective, n_trials=n_trials, show_progress_bar=(verbose > 0))
        
        # Get best parameters and score
        best_params = study.best_params
        best_score = study.best_value
        
        if verbose > 0:
            self.logger.info(f"Best parameters: {best_params}")
            self.logger.info(f"Best score: {best_score:.4f}")
            
        return best_params, best_score, study
    
    def _param_grid_to_dict(self, param_grid):
        """Convert parameter grid to list of dictionaries with all combinations."""
        # If param_grid is already a list of dicts, return it
        if isinstance(param_grid, list):
            return param_grid
            
        # Get all parameter names and possible values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all combinations
        for combination in itertools.product(*param_values):
            yield dict(zip(param_names, combination))
    
    def _random_param_grid(self, param_grid, n_iter):
        """Generate random combinations from parameter grid."""
        import random

        # Get all parameter names and possible values
        param_names = list(param_grid.keys())

        for _ in range(n_iter):
            # Generate a random combination
            params = {}
            for param_name in param_names:
                param_values = param_grid[param_name]

                if isinstance(param_values, list):
                    # Categorical parameter
                    params[param_name] = random.choice(param_values)
                elif isinstance(param_values, dict):
                    # Complex parameter specification
                    param_type = param_values.get('type', 'categorical')

                    if param_type == 'categorical':
                        params[param_name] = random.choice(param_values['values'])
                    elif param_type == 'int':
                        low = param_values['low']
                        high = param_values['high']
                        step = param_values.get('step', 1)

                        # Generate a random integer
                        if step == 1:
                            params[param_name] = random.randint(low, high)
                        else:
                            # Generate with step
                            values = list(range(low, high + 1, step))
                            params[param_name] = random.choice(values)
                    elif param_type == 'float':
                        low = param_values['low']
                        high = param_values['high']
                        step = param_values.get('step', None)

                        if param_values.get('log', False):
                            # Generate from log-uniform distribution
                            params[param_name] = np.exp(random.uniform(np.log(low), np.log(high)))
                        else:
                            # Generate from uniform distribution
                            if step is None:
                                params[param_name] = random.uniform(low, high)
                            else:
                                # Generate with step
                                n_steps = int((high - low) / step) + 1
                                values = [low + i * step for i in range(n_steps)]
                                params[param_name] = random.choice(values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    # Simple range specification (low, high)
                    low, high = param_values

                    if isinstance(low, int) and isinstance(high, int):
                        params[param_name] = random.randint(low, high)
                    else:
                        params[param_name] = random.uniform(low, high)
                else:
                    raise ValueError(f"Unsupported parameter specification for {param_name}")

            yield params

    def _register_clustering_scorer(self, scoring):
        """Register custom scoring functions for clustering evaluation."""
        from sklearn.metrics import make_scorer

        if isinstance(scoring, str):
            # Single scorer
            if scoring == 'silhouette':
                return make_scorer(silhouette_score, greater_is_better=True)
            elif scoring == 'davies_bouldin':
                # Davies-Bouldin is better when lower, so negate it
                return make_scorer(davies_bouldin_score, greater_is_better=False)
            elif scoring == 'calinski_harabasz':
                return make_scorer(calinski_harabasz_score, greater_is_better=True)
            else:
                return scoring
        elif isinstance(scoring, list):
            # List of scorers
            result = {}
            for scorer in scoring:
                if scorer == 'silhouette':
                    result[scorer] = make_scorer(silhouette_score, greater_is_better=True)
                elif scorer == 'davies_bouldin':
                    result[scorer] = make_scorer(davies_bouldin_score, greater_is_better=False)
                elif scorer == 'calinski_harabasz':
                    result[scorer] = make_scorer(calinski_harabasz_score, greater_is_better=True)
                else:
                    result[scorer] = scorer
            return result
        else:
            return scoring

    def _is_deep_learning_model(self, model):
        """Check if a model is a deep learning model."""
        # Check for TensorFlow/Keras model
        if TENSORFLOW_AVAILABLE:
            if isinstance(model, tf.keras.Model):
                return True

        # Check for class/type name containing keywords
        model_type = type(model).__name__.lower()
        dl_keywords = ['keras', 'tensorflow', 'sequential', 'neural', 'deep', 'nn', 'mlp', 'cnn', 'rnn', 'lstm']

        for keyword in dl_keywords:
            if keyword in model_type:
                return True

        # Check for common deep learning methods/attributes
        dl_attributes = ['fit', 'predict', 'layers', 'compile']
        attribute_count = sum(1 for attr in dl_attributes if hasattr(model, attr))

        if attribute_count >= 3:
            # If it has most of the common DL attributes, it's likely a DL model
            return True

        return False

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
            X_processed = X.values
        else:
            X_processed = X

        # Process target variable
        if y is not None:
            if isinstance(y, pd.Series):
                y_processed = y.values
            else:
                y_processed = y

            return X_processed, y_processed
        else:
            return X_processed

    def evaluate_model(self, model, X, y, test_size=0.2, scoring=None, **kwargs):
        """
        Evaluate a model on a holdout test set.

        Parameters:
        -----------
        model : estimator object
            Model to evaluate
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        test_size : float
            Size of the test set
        scoring : str or list
            Metrics to calculate
        **kwargs :
            Additional parameters

        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Process input data
        X_processed, y_processed = self._preprocess_data(X, y)

        # Set default scoring based on task type if not specified
        if scoring is None:
            if self.task_type == 'classification':
                scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr_weighted']
            elif self.task_type == 'regression':
                scoring = ['mse', 'rmse', 'mae', 'r2', 'explained_variance']
            else:  # clustering
                scoring = ['silhouette', 'davies_bouldin', 'calinski_harabasz']

        # Split data into train/test sets
        stratify = y_processed if self.task_type == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=test_size,
            random_state=self.random_state, stratify=stratify
        )

        # Check if model is already fitted
        try:
            # Check if the model is already fitted
            if hasattr(model, 'predict'):
                model.predict(X_train[:1])
                is_fitted = True
            else:
                # If predict method doesn't exist, assume not fitted
                is_fitted = False
        except (NotFittedError, Exception):
            is_fitted = False

        # Fit the model if not already fitted
        if not is_fitted:
            self.logger.info("Fitting model...")

            # Check if model is a deep learning model
            is_dl_model = self._is_deep_learning_model(model)

            if is_dl_model:
                # Extract deep learning specific parameters
                batch_size = kwargs.get('batch_size', 32)
                epochs = kwargs.get('epochs', 100)
                verbose = kwargs.get('verbose', 0)
                validation_split = kwargs.get('validation_split', 0.1)
                callbacks = kwargs.get('callbacks', None)

                # Train the deep learning model
                model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=verbose
                )
            else:
                # Train a traditional ML model
                model.fit(X_train, y_train)

        # Evaluate the model
        metrics = {}

        if self.task_type == 'classification':
            # Make predictions
            y_pred = model.predict(X_test)

            # Try to get probability predictions if available
            try:
                y_pred_proba = model.predict_proba(X_test)
                has_proba = True
            except:
                has_proba = False

            # Calculate classification metrics
            if 'accuracy' in scoring:
                metrics['accuracy'] = accuracy_score(y_test, y_pred)

            if 'precision' in scoring or 'precision_weighted' in scoring:
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted')

            if 'recall' in scoring or 'recall_weighted' in scoring:
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted')

            if 'f1' in scoring or 'f1_weighted' in scoring:
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted')

            if ('roc_auc' in scoring or 'roc_auc_ovr_weighted' in scoring) and has_proba:
                # Check if multi-class
                if len(np.unique(y_test)) > 2:
                    # One-hot encode for multi-class
                    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                    metrics['roc_auc'] = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='weighted')
                else:
                    # Binary classification
                    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

            # Calculate confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)

            # Store prediction data for later visualization
            self._evaluation_data = {
                'y_true': y_test,
                'y_pred': y_pred,
                'has_proba': has_proba
            }

            if has_proba:
                self._evaluation_data['y_pred_proba'] = y_pred_proba

        elif self.task_type == 'regression':
            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate regression metrics
            if 'mse' in scoring or 'mean_squared_error' in scoring:
                metrics['mse'] = mean_squared_error(y_test, y_pred)

            if 'rmse' in scoring or 'root_mean_squared_error' in scoring:
                metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))

            if 'mae' in scoring or 'mean_absolute_error' in scoring:
                metrics['mae'] = mean_absolute_error(y_test, y_pred)

            if 'r2' in scoring:
                metrics['r2'] = r2_score(y_test, y_pred)

            if 'explained_variance' in scoring:
                metrics['explained_variance'] = explained_variance_score(y_test, y_pred)

            # Add correlation coefficient
            metrics['pearson_r'], metrics['pearson_p'] = stats.pearsonr(y_test, y_pred)
            metrics['spearman_r'], metrics['spearman_p'] = stats.spearmanr(y_test, y_pred)

            # Store prediction data for later visualization
            self._evaluation_data = {
                'y_true': y_test,
                'y_pred': y_pred
            }

        else:  # clustering
            # Make predictions (cluster assignments)
            labels = model.predict(X_test)

            # Calculate clustering metrics
            if 'silhouette' in scoring:
                metrics['silhouette'] = silhouette_score(X_test, labels)

            if 'davies_bouldin' in scoring:
                metrics['davies_bouldin'] = davies_bouldin_score(X_test, labels)

            if 'calinski_harabasz' in scoring:
                metrics['calinski_harabasz'] = calinski_harabasz_score(X_test, labels)

            # Store data for later visualization
            self._evaluation_data = {
                'X': X_test,
                'labels': labels
            }

        # Log results
        self.logger.info("Evaluation completed")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                self.logger.info(f"{metric}: {value:.4f}")

        # Store metrics
        self.evaluation_metrics_ = metrics

        return metrics

    def bootstrap_evaluate(self, model, X, y, n_iterations=1000, confidence_level=0.95, **kwargs):
        """
        Perform bootstrap evaluation to get confidence intervals for metrics.

        Parameters:
        -----------
        model : estimator object
            Fitted model to evaluate
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        n_iterations : int
            Number of bootstrap iterations
        confidence_level : float
            Confidence level for intervals (0.0-1.0)
        **kwargs :
            Additional parameters

        Returns:
        --------
        dict
            Dictionary of bootstrap evaluation results
        """
        # Process input data
        X_processed, y_processed = self._preprocess_data(X, y)

        # Check if model is fitted
        try:
            # Check if the model is already fitted
            if hasattr(model, 'predict'):
                model.predict(X_processed[:1])
            else:
                raise ValueError("Model must be fitted before bootstrap evaluation")
        except (NotFittedError, Exception):
            raise ValueError("Model must be fitted before bootstrap evaluation")

        # Set up bootstrap metrics
        bootstrap_metrics = defaultdict(list)

        self.logger.info(f"Performing bootstrap evaluation with {n_iterations} iterations...")

        # Perform bootstrap iterations
        for i in range(n_iterations):
            if i % 100 == 0 and i > 0:
                self.logger.info(f"  Completed {i} iterations")

            # Create bootstrap sample
            indices = np.random.choice(len(X_processed), len(X_processed), replace=True)
            X_boot = X_processed[indices]
            y_boot = y_processed[indices]

            # Evaluate on bootstrap sample
            if self.task_type == 'classification':
                # Make predictions
                y_pred = model.predict(X_boot)

                # Try to get probability predictions if available
                try:
                    y_pred_proba = model.predict_proba(X_boot)
                    has_proba = True
                except:
                    has_proba = False

                # Calculate classification metrics
                bootstrap_metrics['accuracy'].append(accuracy_score(y_boot, y_pred))
                bootstrap_metrics['precision'].append(precision_score(y_boot, y_pred, average='weighted'))
                bootstrap_metrics['recall'].append(recall_score(y_boot, y_pred, average='weighted'))
                bootstrap_metrics['f1'].append(f1_score(y_boot, y_pred, average='weighted'))

                if has_proba:
                    # Check if multi-class
                    if len(np.unique(y_boot)) > 2:
                        # One-hot encode for multi-class
                        y_boot_bin = label_binarize(y_boot, classes=np.unique(y_boot))
                        bootstrap_metrics['roc_auc'].append(
                            roc_auc_score(y_boot_bin, y_pred_proba, multi_class='ovr', average='weighted')
                        )
                    else:
                        # Binary classification
                        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                            bootstrap_metrics['roc_auc'].append(roc_auc_score(y_boot, y_pred_proba[:, 1]))
                        else:
                            bootstrap_metrics['roc_auc'].append(roc_auc_score(y_boot, y_pred_proba))

            elif self.task_type == 'regression':
                # Make predictions
                y_pred = model.predict(X_boot)

                # Calculate regression metrics
                bootstrap_metrics['mse'].append(mean_squared_error(y_boot, y_pred))
                bootstrap_metrics['rmse'].append(np.sqrt(mean_squared_error(y_boot, y_pred)))
                bootstrap_metrics['mae'].append(mean_absolute_error(y_boot, y_pred))
                bootstrap_metrics['r2'].append(r2_score(y_boot, y_pred))
                bootstrap_metrics['explained_variance'].append(explained_variance_score(y_boot, y_pred))

            else:  # clustering
                # Make predictions (cluster assignments)
                labels = model.predict(X_boot)

                # Calculate clustering metrics
                bootstrap_metrics['silhouette'].append(silhouette_score(X_boot, labels))
                bootstrap_metrics['davies_bouldin'].append(davies_bouldin_score(X_boot, labels))
                bootstrap_metrics['calinski_harabasz'].append(calinski_harabasz_score(X_boot, labels))

        # Calculate confidence intervals
        alpha = 1.0 - confidence_level
        lower_percentile = alpha / 2.0 * 100
        upper_percentile = (1.0 - alpha / 2.0) * 100

        bootstrap_results = {}

        for metric, values in bootstrap_metrics.items():
            # Calculate mean and standard deviation
            mean_value = np.mean(values)
            std_value = np.std(values)

            # Calculate confidence interval
            lower_bound = np.percentile(values, lower_percentile)
            upper_bound = np.percentile(values, upper_percentile)

            bootstrap_results[metric] = {
                'mean': mean_value,
                'std': std_value,
                'confidence_interval': (lower_bound, upper_bound),
                'values': values
            }

            self.logger.info(f"{metric}: {mean_value:.4f} ± {std_value:.4f} ({confidence_level*100:.1f}% CI: {lower_bound:.4f} - {upper_bound:.4f})")

        return bootstrap_results

    def permutation_importance(self, model, X, y, n_repeats=10, **kwargs):
        """
        Compute permutation feature importance.

        Parameters:
        -----------
        model : estimator object
            Fitted model
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        n_repeats : int
            Number of times to permute each feature
        **kwargs :
            Additional parameters

        Returns:
        --------
        dict
            Dictionary of permutation importance results
        """
        from sklearn.inspection import permutation_importance

        # Process input data
        X_processed, y_processed = self._preprocess_data(X, y)

        # Store feature names if available
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

        # Set up scoring based on task type
        scoring = kwargs.get('scoring', None)
        if scoring is None:
            if self.task_type == 'classification':
                scoring = 'accuracy'
            elif self.task_type == 'regression':
                scoring = 'r2'
            else:  # clustering
                # For clustering, we'll use silhouette score
                scoring = make_scorer(silhouette_score)

        # Check if model is a deep learning model
        is_dl_model = self._is_deep_learning_model(model)

        if is_dl_model:
            # For deep learning models, we'll need to create a wrapper
            self.logger.info("Creating wrapper for deep learning model...")

            # Create a scikit-learn compatible wrapper
            class KerasWrapper:
                def __init__(self, model, task_type):
                    self.model = model
                    self.task_type = task_type

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
            model_wrapper = KerasWrapper(model, self.task_type)

            # Compute permutation importance
            self.logger.info(f"Computing permutation importance with {n_repeats} repeats...")
            result = permutation_importance(
                model_wrapper, X_processed, y_processed,
                n_repeats=n_repeats,
                random_state=self.random_state,
                scoring=scoring
            )
        else:
            # Compute permutation importance
            self.logger.info(f"Computing permutation importance with {n_repeats} repeats...")
            result = permutation_importance(
                model, X_processed, y_processed,
                n_repeats=n_repeats,
                random_state=self.random_state,
                scoring=scoring
            )

        # Create results dictionary with feature names
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)

        # Store feature importances
        self.feature_importances_ = importance_df

        self.logger.info("Permutation importance completed")
        self.logger.info("Top 5 features:")
        for i, row in importance_df.head(5).iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

        return {
            'importances_mean': result.importances_mean,
            'importances_std': result.importances_std,
            'importances': result.importances,
            'feature_names': feature_names,
            'importance_df': importance_df
        }

    def compute_shap_values(self, model, X, **kwargs):
        """
        Compute SHAP values for model explanation.

        Parameters:
        -----------
        model : estimator object
            Fitted model
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        **kwargs :
            Additional parameters

        Returns:
        --------
        object
            SHAP values object
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with 'pip install shap'")

        # Process input data
        if isinstance(X, pd.DataFrame):
            X_processed = X.copy()
            feature_names = X.columns.tolist()
        else:
            X_processed = X
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

        # Check if model is a deep learning model
        is_dl_model = self._is_deep_learning_model(model)

        self.logger.info("Computing SHAP values...")

        if is_dl_model:
            # For deep learning models, use the DeepExplainer
            # We need a background dataset (subset of X)
            background_size = min(100, len(X_processed))
            background = shap.sample(X_processed, background_size)

            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(X_processed)
        else:
            # Try to detect the model type and use the appropriate explainer
            model_type = type(model).__name__.lower()

            if 'tree' in model_type or 'forest' in model_type or 'boost' in model_type or 'xgb' in model_type:
                # Tree-based model
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_processed)
            elif 'linear' in model_type or 'logistic' in model_type or 'regression' in model_type:
                # Linear model
                explainer = shap.LinearExplainer(model, X_processed)
                shap_values = explainer.shap_values(X_processed)
            else:
                # Fallback to Kernel explainer
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_processed, 100))
                shap_values = explainer.shap_values(X_processed)

        self.logger.info("SHAP values computed successfully")

        # Return SHAP values along with feature names
        return {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'explainer': explainer
        }

    def create_ensemble(self, models, ensemble_type='voting', weights=None, **kwargs):
        """
        Create an ensemble of models.

        Parameters:
        -----------
        models : list
            List of (name, model) tuples
        ensemble_type : str
            Type of ensemble ('voting' or 'stacking')
        weights : list, optional
            List of weights for voting ensemble
        **kwargs :
            Additional parameters

        Returns:
        --------
        estimator
            Ensemble model
        """
        # Check if models is a list of (name, model) tuples
        if not all(isinstance(item, tuple) and len(item) == 2 for item in models):
            raise ValueError("models must be a list of (name, model) tuples")

        if self.task_type == 'classification':
            # Create a classification ensemble
            if ensemble_type.lower() == 'voting':
                # Determine voting type (hard or soft)
                voting = kwargs.get('voting', 'soft')

                ensemble = VotingClassifier(
                    estimators=models,
                    voting=voting,
                    weights=weights
                )
            elif ensemble_type.lower() == 'stacking':
                # Get final estimator
                final_estimator = kwargs.get('final_estimator', None)

                ensemble = StackingClassifier(
                    estimators=models,
                    final_estimator=final_estimator,
                    cv=kwargs.get('cv', 5),
                    stack_method=kwargs.get('stack_method', 'auto'),
                    n_jobs=kwargs.get('n_jobs', -1)
                )
            else:
                raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        elif self.task_type == 'regression':
            # Create a regression ensemble
            if ensemble_type.lower() == 'voting':
                ensemble = VotingRegressor(
                    estimators=models,
                    weights=weights
                )
            elif ensemble_type.lower() == 'stacking':
                # Get final estimator
                final_estimator = kwargs.get('final_estimator', None)

                ensemble = StackingRegressor(
                    estimators=models,
                    final_estimator=final_estimator,
                    cv=kwargs.get('cv', 5),
                    n_jobs=kwargs.get('n_jobs', -1)
                )
            else:
                raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        else:
            raise ValueError(f"Ensemble models not supported for {self.task_type} task")

        self.logger.info(f"Created {ensemble_type} ensemble with {len(models)} models")

        return ensemble

    def calibrate_model(self, model, X, y, method='isotonic', cv=5):
        """
        Calibrate a classification model's probability predictions.

        Parameters:
        -----------
        model : estimator object
            Fitted classification model
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        method : str
            Calibration method ('sigmoid' or 'isotonic')
        cv : int
            Number of cross-validation folds

        Returns:
        --------
        tuple
            (calibrated_model, calibration_scores)
        """
        if self.task_type != 'classification':
            raise ValueError("Model calibration is only for classification tasks")

        from sklearn.calibration import CalibratedClassifierCV

        # Process input data
        X_processed, y_processed = self._preprocess_data(X, y)

        # Create calibrated model
        calibrated_model = CalibratedClassifierCV(
            base_estimator=model,
            method=method,
            cv=cv
        )

        # Fit the calibrated model
        calibrated_model.fit(X_processed, y_processed)

        # Evaluate calibration
        prob_true, prob_pred = calibration_curve(
            y_processed,
            calibrated_model.predict_proba(X_processed)[:, 1],
            n_bins=10
        )

        # Calculate Brier score
        from sklearn.metrics import brier_score_loss
        brier_score = brier_score_loss(y_processed, calibrated_model.predict_proba(X_processed)[:, 1])

        self.logger.info(f"Model calibrated using {method} method")
        self.logger.info(f"Brier score: {brier_score:.4f}")

        return calibrated_model, {
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'brier_score': brier_score
        }

    def learning_curve_analysis(self, model, X, y, train_sizes=None, cv=5, scoring=None, **kwargs):
        """
        Generate learning curves to analyze model performance vs. training set size.

        Parameters:
        -----------
        model : estimator object
            Model to analyze
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        train_sizes : array-like
            Array of training set sizes
        cv : int or cv splitter
            Cross-validation strategy
        scoring : str
            Scoring metric
        **kwargs :
            Additional parameters

        Returns:
        --------
        dict
            Dictionary of learning curve results
        """
        # Process input data
        X_processed, y_processed = self._preprocess_data(X, y)

        # Set default train sizes if not specified
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        # Set default scoring
        if scoring is None:
            if self.task_type == 'classification':
                scoring = 'accuracy'
            elif self.task_type == 'regression':
                scoring = 'r2'
            else:
                scoring = 'silhouette'

        # Set up CV strategy
        if isinstance(cv, int):
            if self.task_type == 'classification':
                cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            else:
                cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            cv_strategy = cv

        # Check if model is a deep learning model
        is_dl_model = self._is_deep_learning_model(model)

        if is_dl_model:
            # For deep learning models, we'll implement a custom learning curve
            return self._deep_learning_learning_curve(
                model, X_processed, y_processed,
                train_sizes, cv_strategy, scoring, **kwargs
            )

        # For traditional models, use sklearn's learning_curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_processed, y_processed,
            train_sizes=train_sizes,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=kwargs.get('n_jobs', -1),
            verbose=kwargs.get('verbose', 0),
            shuffle=True,
            random_state=self.random_state
        )

        # Calculate mean and std of scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Create results dictionary
        results = {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'train_mean': train_mean,
            'train_std': train_std,
            'test_mean': test_mean,
            'test_std': test_std
        }

        self.logger.info("Learning curve analysis completed")

        return results

    def _deep_learning_learning_curve(self, model, X, y, train_sizes, cv, scoring, **kwargs):
        """Custom learning curve implementation for deep learning models."""
        # Extract deep learning specific parameters
        batch_size = kwargs.get('batch_size', 32)
        epochs = kwargs.get('epochs', 100)
        verbose = kwargs.get('verbose', 0)

        # Convert train_sizes to absolute values
        n_samples = X.shape[0]
        train_sizes_abs = np.array([int(n * n_samples) for n in train_sizes])

        # Get CV splits
        cv_splits = list(cv.split(X, y))
        n_splits = len(cv_splits)

        # Initialize arrays for scores
        train_scores = np.zeros((len(train_sizes_abs), n_splits))
        test_scores = np.zeros((len(train_sizes_abs), n_splits))

        # For each CV split
        for i, (train_idx, test_idx) in enumerate(cv_splits):
            self.logger.info(f"CV split {i+1}/{n_splits}")

            # Get the full training and test sets for this split
            X_train_full, X_test = X[train_idx], X[test_idx]
            y_train_full, y_test = y[train_idx], y[test_idx]

            # For each training set size
            for j, train_size in enumerate(train_sizes_abs):
                self.logger.info(f"  Training size: {train_size} samples")

                # Subsample the training data
                train_indices = np.random.choice(len(X_train_full), train_size, replace=False)
                X_train = X_train_full[train_indices]
                y_train = y_train_full[train_indices]

                # Create and train a new model
                try:
                    # Clone the model by recreating it
                    model_clone = self._create_dl_model_with_params(model, {})

                    # Train the model
                    model_clone.fit(
                        X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0
                    )

                    # Calculate scores
                    if self.task_type == 'classification':
                        # Make predictions
                        y_train_pred = model_clone.predict(X_train)
                        y_test_pred = model_clone.predict(X_test)

                        if y_train_pred.shape[1] > 1:  # Multi-class
                            y_train_pred_class = np.argmax(y_train_pred, axis=1)
                            y_test_pred_class = np.argmax(y_test_pred, axis=1)
                        else:  # Binary
                            y_train_pred_class = (y_train_pred > 0.5).astype(int).flatten()
                            y_test_pred_class = (y_test_pred > 0.5).astype(int).flatten()

                        # Calculate appropriate metric
                        if scoring == 'accuracy':
                            train_score = accuracy_score(y_train, y_train_pred_class)
                            test_score = accuracy_score(y_test, y_test_pred_class)
                        elif scoring == 'precision' or scoring == 'precision_weighted':
                            train_score = precision_score(y_train, y_train_pred_class, average='weighted')
                            test_score = precision_score(y_test, y_test_pred_class, average='weighted')
                        elif scoring == 'recall' or scoring == 'recall_weighted':
                            train_score = recall_score(y_train, y_train_pred_class, average='weighted')
                            test_score = recall_score(y_test, y_test_pred_class, average='weighted')
                        elif scoring == 'f1' or scoring == 'f1_weighted':
                            train_score = f1_score(y_train, y_train_pred_class, average='weighted')
                            test_score = f1_score(y_test, y_test_pred_class, average='weighted')
                        elif scoring == 'roc_auc' or scoring == 'roc_auc_ovr_weighted':
                            # For multi-class, need to handle differently
                            if len(np.unique(y_train)) > 2:
                                # One-hot encode labels
                                y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
                                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

                                train_score = roc_auc_score(y_train_bin, y_train_pred, multi_class='ovr', average='weighted')
                                test_score = roc_auc_score(y_test_bin, y_test_pred, multi_class='ovr', average='weighted')
                            else:
                                # Binary classification
                                train_score = roc_auc_score(y_train, y_train_pred)
                                test_score = roc_auc_score(y_test, y_test_pred)
                        else:
                            # Default to accuracy
                            train_score = accuracy_score(y_train, y_train_pred_class)
                            test_score = accuracy_score(y_test, y_test_pred_class)
                    else:  # regression
                        # Make predictions
                        y_train_pred = model_clone.predict(X_train).flatten()
                        y_test_pred = model_clone.predict(X_test).flatten()

                        # Calculate appropriate metric
                        if scoring == 'neg_mean_squared_error':
                            train_score = -mean_squared_error(y_train, y_train_pred)
                            test_score = -mean_squared_error(y_test, y_test_pred)
                        elif scoring == 'neg_root_mean_squared_error':
                            train_score = -np.sqrt(mean_squared_error(y_train, y_train_pred))
                            test_score = -np.sqrt(mean_squared_error(y_test, y_test_pred))
                        elif scoring == 'neg_mean_absolute_error':
                            train_score = -mean_absolute_error(y_train, y_train_pred)
                            test_score = -mean_absolute_error(y_test, y_test_pred)
                        elif scoring == 'r2':
                            train_score = r2_score(y_train, y_train_pred)
                            test_score = r2_score(y_test, y_test_pred)
                        elif scoring == 'explained_variance':
                            train_score = explained_variance_score(y_train, y_train_pred)
                            test_score = explained_variance_score(y_test, y_test_pred)
                        else:
                            # Default to R²
                            train_score = r2_score(y_train, y_train_pred)
                            test_score = r2_score(y_test, y_test_pred)

                    # Store scores
                    train_scores[j, i] = train_score
                    test_scores[j, i] = test_score

                    self.logger.info(f"    Train score: {train_score:.4f}, Test score: {test_score:.4f}")

                except Exception as e:
                    self.logger.error(f"Error in learning curve calculation: {str(e)}")
                    train_scores[j, i] = np.nan
                    test_scores[j, i] = np.nan

        # Calculate mean and std of scores
        train_mean = np.nanmean(train_scores, axis=1)
        train_std = np.nanstd(train_scores, axis=1)
        test_mean = np.nanmean(test_scores, axis=1)
        test_std = np.nanstd(test_scores, axis=1)

        # Create results dictionary
        results = {
            'train_sizes': train_sizes,
            'train_sizes_abs': train_sizes_abs,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'train_mean': train_mean,
            'train_std': train_std,
            'test_mean': test_mean,
            'test_std': test_std
        }

        self.logger.info("Learning curve analysis completed")

        return results

    def plot_learning_curve(self, learning_curve_results, **kwargs):
        """
        Plot learning curve results.

        Parameters:
        -----------
        learning_curve_results : dict
            Results from learning_curve_analysis
        **kwargs :
            Additional plotting parameters

        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        # Extract results
        train_sizes = learning_curve_results['train_sizes']
        train_mean = learning_curve_results['train_mean']
        train_std = learning_curve_results['train_std']
        test_mean = learning_curve_results['test_mean']
        test_std = learning_curve_results['test_std']

        # Extract plotting parameters
        figsize = kwargs.get('figsize', (10, 6))
        title = kwargs.get('title', 'Learning Curve')
        ylabel = kwargs.get('ylabel', 'Score')
        ylim = kwargs.get('ylim', None)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot learning curve
        ax.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color='blue')
        ax.fill_between(train_sizes, test_mean - test_std,
                        test_mean + test_std, alpha=0.1, color='orange')
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label="Training score")
        ax.plot(train_sizes, test_mean, 'o-', color='orange', label="Cross-validation score")

        # Set axis labels and title
        ax.set_xlabel("Training set size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Set y-axis limits if provided
        if ylim is not None:
            ax.set_ylim(ylim)

        # Add grid and legend
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

        return fig

    def plot_confusion_matrix(self, **kwargs):
        """
        Plot confusion matrix for classification models.

        Parameters:
        -----------
        **kwargs :
            Additional plotting parameters

        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.task_type != 'classification':
            raise ValueError("Confusion matrix only available for classification tasks")

        if not hasattr(self, '_evaluation_data') or self._evaluation_data is None:
            raise ValueError("No evaluation data available. Call evaluate_model() first")

        # Get confusion matrix
        y_true = self._evaluation_data['y_true']
        y_pred = self._evaluation_data['y_pred']

        # If class labels are not strings, try to convert them
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        if not isinstance(unique_labels[0], str):
            labels = [str(label) for label in unique_labels]
        else:
            labels = unique_labels

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        # Extract plotting parameters
        figsize = kwargs.get('figsize', (8, 6))
        title = kwargs.get('title', 'Confusion Matrix')
        cmap = kwargs.get('cmap', 'Blues')
        normalize = kwargs.get('normalize', False)

        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        # Set axis labels and title
        ax.set(xticks=np.arange(cm.shape[1]),
              yticks=np.arange(cm.shape[0]),
              xticklabels=labels, yticklabels=labels,
              title=title,
              ylabel='True label',
              xlabel='Predicted label')

        # Rotate tick labels if necessary
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        # Adjust layout
        fig.tight_layout()

        return fig

    def plot_roc_curve(self, **kwargs):
        """
        Plot ROC curve for classification models.

        Parameters:
        -----------
        **kwargs :
            Additional plotting parameters

        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.task_type != 'classification':
            raise ValueError("ROC curve only available for classification tasks")

        if not hasattr(self, '_evaluation_data') or self._evaluation_data is None:
            raise ValueError("No evaluation data available. Call evaluate_model() first")

        if not self._evaluation_data.get('has_proba', False):
            raise ValueError("ROC curve requires probability predictions")

        # Get data
        y_true = self._evaluation_data['y_true']
        y_pred_proba = self._evaluation_data['y_pred_proba']

        # Extract plotting parameters
        figsize = kwargs.get('figsize', (8, 6))
        title = kwargs.get('title', 'ROC Curve')

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Check if multi-class or binary classification
        unique_classes = np.unique(y_true)

        if len(unique_classes) > 2:
            # Multi-class ROC
            # One-hot encode true labels
            y_true_bin = label_binarize(y_true, classes=unique_classes)

            # Compute ROC curve and ROC area for each class
            fpr = {}
            tpr = {}
            roc_auc = {}

            for i, cls in enumerate(unique_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

                # Plot ROC curve for each class
                ax.plot(fpr[i], tpr[i], lw=2,
                       label=f'ROC curve (class {cls}, area = {roc_auc[i]:.2f})')
        else:
            # Binary classification
            # Get positive class probabilities
            if y_pred_proba.shape[1] > 1:
                pos_probs = y_pred_proba[:, 1]
            else:
                pos_probs = y_pred_proba.flatten()

            # Compute ROC curve and ROC area
            fpr, tpr, _ = roc_curve(y_true, pos_probs)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=2)

        # Set axis labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)

        # Add legend and grid
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        # Adjust layout
        fig.tight_layout()

        return fig

    def plot_precision_recall_curve(self, **kwargs):
        """
        Plot precision-recall curve for classification models.

        Parameters:
        -----------
        **kwargs :
            Additional plotting parameters

        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.task_type != 'classification':
            raise ValueError("Precision-recall curve only available for classification tasks")

        if not hasattr(self, '_evaluation_data') or self._evaluation_data is None:
            raise ValueError("No evaluation data available. Call evaluate_model() first")

        if not self._evaluation_data.get('has_proba', False):
            raise ValueError("Precision-recall curve requires probability predictions")

        # Get data
        y_true = self._evaluation_data['y_true']
        y_pred_proba = self._evaluation_data['y_pred_proba']

        # Extract plotting parameters
        figsize = kwargs.get('figsize', (8, 6))
        title = kwargs.get('title', 'Precision-Recall Curve')

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Check if multi-class or binary classification
        unique_classes = np.unique(y_true)

        if len(unique_classes) > 2:
            # Multi-class precision-recall
            # One-hot encode true labels
            y_true_bin = label_binarize(y_true, classes=unique_classes)

            # Compute precision-recall curve for each class
            precision = {}
            recall = {}
            avg_precision = {}

            for i, cls in enumerate(unique_classes):
                precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
                avg_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])

                # Plot precision-recall curve for each class
                ax.plot(recall[i], precision[i], lw=2,
                       label=f'Class {cls} (AP = {avg_precision[i]:.2f})')
        else:
            # Binary classification
            # Get positive class probabilities
            if y_pred_proba.shape[1] > 1:
                pos_probs = y_pred_proba[:, 1]
            else:
                pos_probs = y_pred_proba.flatten()

            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, pos_probs)
            avg_precision = average_precision_score(y_true, pos_probs)

            # Plot precision-recall curve
            ax.plot(recall, precision, lw=2, label=f'AP = {avg_precision:.2f}')

        # Set axis labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)

        # Add legend and grid
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        # Adjust layout
        fig.tight_layout()

        return fig

    def plot_residuals(self, **kwargs):
        """
        Plot residuals for regression models.

        Parameters:
        -----------
        **kwargs :
            Additional plotting parameters

        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.task_type != 'regression':
            raise ValueError("Residual plots only available for regression tasks")

        if not hasattr(self, '_evaluation_data') or self._evaluation_data is None:
            raise ValueError("No evaluation data available. Call evaluate_model() first")

        # Get data
        y_true = self._evaluation_data['y_true']
        y_pred = self._evaluation_data['y_pred']

        # Calculate residuals
        residuals = y_true - y_pred

        # Extract plotting parameters
        figsize = kwargs.get('figsize', (12, 8))
        title = kwargs.get('title', 'Regression Residuals')

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Predicted vs. Actual
        axes[0, 0].scatter(y_pred, y_true, alpha=0.5)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_title('Predicted vs. Actual')
        axes[0, 0].grid(alpha=0.3)

        # Calculate R²
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=axes[0, 0].transAxes,
                      bbox=dict(facecolor='white', alpha=0.8))

        # Plot 2: Residuals vs. Predicted
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs. Predicted')
        axes[0, 1].grid(alpha=0.3)

        # Plot 3: Histogram of Residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(x=0, color='k', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Histogram of Residuals')
        axes[1, 0].grid(alpha=0.3)

        # Calculate mean and std of residuals
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        axes[1, 0].text(0.05, 0.95, f'Mean = {mean_residual:.3f}\nStd = {std_residual:.3f}',
                      transform=axes[1, 0].transAxes,
                      bbox=dict(facecolor='white', alpha=0.8))

        # Plot 4: Q-Q Plot
        from scipy import stats

        # Calculate quantiles for normal Q-Q plot
        sorted_residuals = np.sort(residuals)
        norm_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))

        # Create Q-Q plot
        axes[1, 1].scatter(norm_quantiles, sorted_residuals, alpha=0.5)
        axes[1, 1].plot([np.min(norm_quantiles), np.max(norm_quantiles)],
                      [np.min(norm_quantiles) * std_residual + mean_residual,
                       np.max(norm_quantiles) * std_residual + mean_residual],
                      'k--', lw=2)
        axes[1, 1].set_xlabel('Theoretical Quantiles')
        axes[1, 1].set_ylabel('Sample Quantiles')
        axes[1, 1].set_title('Normal Q-Q Plot')
        axes[1, 1].grid(alpha=0.3)

        # Overall title
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        return fig

    def plot_permutation_importance(self, n_features=20, **kwargs):
        """
        Plot permutation feature importance.

        Parameters:
        -----------
        n_features : int
            Number of top features to display
        **kwargs :
            Additional plotting parameters

        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available. Run permutation_importance() first")

        # Get the top features
        top_features = self.feature_importances_.head(n_features)

        # Extract plotting parameters
        figsize = kwargs.get('figsize', (10, 0.5 * n_features))
        title = kwargs.get('title', 'Permutation Feature Importance')

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot feature importances
        top_features = top_features.sort_values('importance_mean')
        y_pos = np.arange(len(top_features))

        ax.barh(y_pos, top_features['importance_mean'], xerr=top_features['importance_std'],
               align='center', alpha=0.8)

        # Set axis labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(title)

        # Add grid
        ax.grid(alpha=0.3, axis='x')

        # Adjust layout
        fig.tight_layout()

        return fig

    def plot_shap_summary(self, shap_values, **kwargs):
        """
        Plot SHAP summary plot.

        Parameters:
        -----------
        shap_values : object
            SHAP values from compute_shap_values()
        **kwargs :
            Additional plotting parameters

        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with 'pip install shap'")

        # Extract plotting parameters
        figsize = kwargs.get('figsize', (10, 8))
        max_display = kwargs.get('max_display', 20)

        # Create figure
        plt.figure(figsize=figsize)

        # Create SHAP summary plot
        if self.task_type == 'classification' and isinstance(shap_values['shap_values'], list):
            # For multi-class classification, we have a list of SHAP values
            # Plot for a specific class or the mean absolute value
            if 'class_idx' in kwargs:
                class_idx = kwargs['class_idx']
                shap.summary_plot(shap_values['shap_values'][class_idx],
                                feature_names=shap_values['feature_names'],
                                max_display=max_display, show=False)
            else:
                # Calculate mean absolute SHAP values across classes
                mean_shap = np.abs(np.array(shap_values['shap_values'])).mean(axis=0)
                shap.summary_plot(mean_shap, feature_names=shap_values['feature_names'],
                                max_display=max_display, show=False)
        else:
            # For binary classification or regression
            shap.summary_plot(shap_values['shap_values'], feature_names=shap_values['feature_names'],
                            max_display=max_display, show=False)

        return plt.gcf()

    def save_results(self, filepath, **kwargs):
        """
        Save evaluation results to a file.

        Parameters:
        -----------
        filepath : str
            Path to save the results
        **kwargs :
            Additional parameters
        """
        # Create a results dictionary
        results = {
            'task_type': self.task_type,
            'cv_results': self.cv_results_,
            'optimization_results': self.optimization_results_,
            'evaluation_metrics': self.evaluation_metrics_,
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'feature_importances': self.feature_importances_,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Add metadata if provided
        if 'metadata' in kwargs:
            results['metadata'] = kwargs['metadata']

        # Save the results
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)

        self.logger.info(f"Results saved to {filepath}")

    @classmethod
    def load_results(cls, filepath):
        """
        Load evaluation results from a file.

        Parameters:
        -----------
        filepath : str
            Path to the saved results

        Returns:
        --------
        dict
            Dictionary of evaluation results
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)

        return results
