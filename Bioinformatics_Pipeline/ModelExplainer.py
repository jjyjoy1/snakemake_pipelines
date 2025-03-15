import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
import os
import pickle
from collections import defaultdict
import json
from datetime import datetime
from tqdm import tqdm
import joblib

# Import scikit-learn related modules
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

# Try importing optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. SHAP explanations unavailable. "
                 "Install with 'pip install shap'.")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not installed. LIME explanations unavailable. "
                 "Install with 'pip install lime'.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not installed. Deep learning model explanations limited. "
                  "Install with 'pip install tensorflow'.")

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. PyTorch model explanations unavailable. "
                  "Install with 'pip install torch'.")


class BioinformaticsModelExplainer:
    """
    Step 7: Explainable AI (XAI)
    
    A class for explaining and interpreting machine learning models in bioinformatics,
    identifying key features and mechanisms driving predictions.
    
    This class implements various explanation methods:
    - SHAP (SHapley Additive exPlanations) for global and local feature attribution
    - LIME (Local Interpretable Model-agnostic Explanations) for local explanations
    - Feature importance extraction from various model types
    - Attention weights visualization for transformer/LSTM models
    
    Supports various model types:
    - Traditional ML: Random Forest, XGBoost, etc.
    - Deep Learning: TensorFlow/Keras models, PyTorch models
    """
    
    def __init__(self, model, task_type='classification', model_type=None, logger=None):
        """
        Initialize the model explainer.
        
        Parameters:
        -----------
        model : estimator object
            The trained model to explain
        task_type : str
            Type of machine learning task ('classification' or 'regression')
        model_type : str, optional
            Type of model (e.g., 'random_forest', 'xgboost', 'cnn', 'lstm', 'transformer')
            If None, will attempt to detect automatically
        logger : logging.Logger
            Logger for tracking the explanation process
        """
        self.model = model
        self.task_type = task_type.lower()
        self.model_type = model_type
        self.explanation_results = {}
        self.feature_names = None
        self.feature_importances = None
        self.shap_values = None
        self.lime_explanations = {}
        self.attention_weights = None
        
        # Set up logger
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
            
        # Validate task_type
        valid_tasks = ['classification', 'regression', 'clustering']
        if self.task_type not in valid_tasks:
            raise ValueError(f"Invalid task type '{self.task_type}'. Choose from: {', '.join(valid_tasks)}")
            
        # Detect model type if not provided
        if self.model_type is None:
            self.model_type = self._detect_model_type()
            self.logger.info(f"Detected model type: {self.model_type}")
    
    def _setup_logger(self):
        """Setup a basic logger if none is provided."""
        logger = logging.getLogger("BioinformaticsModelExplainer")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def _detect_model_type(self):
        """
        Detect the type of model automatically.
        
        Returns:
        --------
        str
            Detected model type
        """
        model_str = str(type(self.model)).lower()
        
        # Check for scikit-learn models
        if "randomforest" in model_str:
            return "random_forest"
        elif "xgb" in model_str or "xgboost" in model_str:
            return "xgboost"
        elif "gradient" in model_str and "boost" in model_str:
            return "gradient_boosting"
        elif "linearsvc" in model_str or "svc" in model_str:
            return "svm"
        elif "logistic" in model_str:
            return "logistic_regression"
        elif "linear" in model_str and "regression" in model_str:
            return "linear_regression"
            
        # Check for deep learning models
        if TENSORFLOW_AVAILABLE:
            if isinstance(self.model, tf.keras.Model):
                # Try to determine the type of deep learning model
                for layer in self.model.layers:
                    layer_type = layer.__class__.__name__.lower()
                    if "lstm" in layer_type or "gru" in layer_type or "rnn" in layer_type:
                        return "lstm"
                    elif "conv" in layer_type:
                        return "cnn"
                    elif "attention" in layer_type or "transformer" in layer_type:
                        return "transformer"
                # Default to generic neural network
                return "neural_network"
                
        if PYTORCH_AVAILABLE and isinstance(self.model, torch.nn.Module):
            # Try to determine the type of PyTorch model
            model_str = str(self.model).lower()
            if "lstm" in model_str or "gru" in model_str or "rnn" in model_str:
                return "lstm"
            elif "conv" in model_str:
                return "cnn"
            elif "attention" in model_str or "transformer" in model_str:
                return "transformer"
            # Default to generic neural network
            return "neural_network"
            
        # Default to unknown
        return "unknown"
    
    def compute_feature_importance(self, X, y=None, method='native', **kwargs):
        """
        Compute feature importance using the model's native method or permutation importance.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray, optional
            The target variable (required for some methods)
        method : str
            Method to compute importance:
            - 'native': Use the model's built-in feature importance
            - 'permutation': Use permutation importance
            - 'shap': Use SHAP values
        **kwargs :
            Additional parameters for the method
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature importances
        """
        # Process input data
        X_processed, feature_names = self._preprocess_features(X)
        self.feature_names = feature_names
        
        self.logger.info(f"Computing feature importance using {method} method")
        
        if method == 'native':
            importances = self._compute_native_importance(**kwargs)
        elif method == 'permutation':
            importances = self._compute_permutation_importance(X_processed, y, **kwargs)
        elif method == 'shap':
            if not SHAP_AVAILABLE:
                raise ImportError("SHAP not installed. Install with 'pip install shap'")
            # If SHAP values already computed, use them; otherwise compute them
            if self.shap_values is None:
                self.compute_shap_values(X, **kwargs)
            importances = self._compute_shap_importance(**kwargs)
        else:
            raise ValueError(f"Unknown importance method: {method}")
            
        # Store feature importances
        self.feature_importances = importances
        self.explanation_results['feature_importance'] = {
            'method': method,
            'importance': importances
        }
        
        # Log top features
        self.logger.info("Top 10 features by importance:")
        for i, (feature, importance) in enumerate(importances.head(10).itertuples(index=False)):
            self.logger.info(f"  {i+1}. {feature}: {importance:.4f}")
            
        return importances
    
    def _compute_native_importance(self, **kwargs):
        """
        Compute feature importance using the model's built-in method.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature importances
        """
        # Try to extract feature importance based on model type
        try:
            if self.model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
                # These models have a feature_importances_ attribute
                importances = self.model.feature_importances_
                
            elif self.model_type in ['logistic_regression', 'linear_regression', 'svm']:
                # These models have coefficients
                if hasattr(self.model, 'coef_'):
                    if len(self.model.coef_.shape) > 1:
                        # For multi-class, use the mean absolute coefficient across classes
                        importances = np.mean(np.abs(self.model.coef_), axis=0)
                    else:
                        importances = np.abs(self.model.coef_)
                elif hasattr(self.model, 'feature_importances_'):
                    importances = self.model.feature_importances_
                else:
                    raise ValueError(f"Model {self.model_type} does not have accessible feature importances or coefficients")
                    
            elif self.model_type in ['cnn', 'lstm', 'transformer', 'neural_network']:
                # For deep learning, no straightforward method
                raise ValueError("Deep learning models don't have native feature importance. Use permutation or SHAP methods.")
                
            else:
                # Try a generic approach
                if hasattr(self.model, 'feature_importances_'):
                    importances = self.model.feature_importances_
                elif hasattr(self.model, 'coef_'):
                    if len(self.model.coef_.shape) > 1:
                        importances = np.mean(np.abs(self.model.coef_), axis=0)
                    else:
                        importances = np.abs(self.model.coef_)
                else:
                    raise ValueError(f"Model {self.model_type} does not have accessible feature importances")
                    
            # Create DataFrame with feature names
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error computing native feature importance: {str(e)}")
            raise
    
    def _compute_permutation_importance(self, X, y, **kwargs):
        """
        Compute permutation feature importance.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The feature matrix
        y : numpy.ndarray
            The target variable
        **kwargs :
            Additional parameters for permutation importance
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature importances
        """
        if y is None:
            raise ValueError("Target variable (y) is required for permutation importance")
            
        # Extract parameters
        n_repeats = kwargs.get('n_repeats', 10)
        random_state = kwargs.get('random_state', 42)
        n_jobs = kwargs.get('n_jobs', -1)
        
        # Check if model is a deep learning model
        is_dl = self.model_type in ['cnn', 'lstm', 'transformer', 'neural_network']
        
        if is_dl and TENSORFLOW_AVAILABLE:
            # Create a wrapper for the deep learning model
            wrapper = self._create_dl_model_wrapper()
            
            # Compute permutation importance
            result = permutation_importance(
                wrapper, X, y,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=n_jobs
            )
        else:
            # Compute permutation importance directly
            result = permutation_importance(
                self.model, X, y,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=n_jobs
            )
            
        # Create DataFrame with feature names
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': result.importances_mean,
            'std': result.importances_std
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def _create_dl_model_wrapper(self):
        """
        Create a scikit-learn compatible wrapper for deep learning models.
        
        Returns:
        --------
        object
            A wrapper object with predict and score methods
        """
        model = self.model
        task_type = self.task_type
        
        # Create a wrapper class
        class DeepLearningWrapper:
            def __init__(self, model, task_type):
                self.model = model
                self.task_type = task_type
                
            def predict(self, X):
                if self.task_type == 'classification':
                    # Get probability predictions
                    y_pred_prob = self.model.predict(X)
                    
                    if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] > 1:
                        # Multi-class
                        return np.argmax(y_pred_prob, axis=1)
                    else:
                        # Binary
                        return (y_pred_prob > 0.5).astype(int).flatten()
                else:
                    # Regression
                    preds = self.model.predict(X)
                    if len(preds.shape) > 1:
                        return preds.flatten()
                    return preds
                    
            def predict_proba(self, X):
                if self.task_type != 'classification':
                    raise ValueError("predict_proba only available for classification")
                return self.model.predict(X)
                
            def score(self, X, y):
                from sklearn.metrics import accuracy_score, r2_score
                
                y_pred = self.predict(X)
                
                if self.task_type == 'classification':
                    return accuracy_score(y, y_pred)
                else:
                    return r2_score(y, y_pred)
                    
        return DeepLearningWrapper(model, task_type)
    
    def compute_shap_values(self, X, y=None, **kwargs):
        """
        Compute SHAP values to explain model predictions.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray, optional
            The target variable (helpful for some models)
        **kwargs :
            Additional parameters for SHAP:
            - background_samples: int, number of background samples for explainers that need it
            - n_samples: int, number of samples to use for explanation (for KernelExplainer)
            - sample_indices: array-like, specific indices to explain
            - explainer_type: str, force a specific explainer type
            
        Returns:
        --------
        object
            SHAP values object
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with 'pip install shap'")
            
        # Process input data
        X_processed, feature_names = self._preprocess_features(X)
        self.feature_names = feature_names
        
        # Extract parameters
        background_samples = kwargs.get('background_samples', min(100, X_processed.shape[0]))
        n_samples = kwargs.get('n_samples', None)
        sample_indices = kwargs.get('sample_indices', None)
        explainer_type = kwargs.get('explainer_type', None)
        
        # Prepare data for explanation
        if sample_indices is not None:
            X_to_explain = X_processed[sample_indices]
        elif n_samples is not None:
            # Randomly sample n_samples data points
            sample_indices = np.random.choice(X_processed.shape[0], size=n_samples, replace=False)
            X_to_explain = X_processed[sample_indices]
        else:
            X_to_explain = X_processed
            
        # Log the explanation process
        self.logger.info(f"Computing SHAP values for {X_to_explain.shape[0]} samples")
        
        # Create appropriate explainer based on model type
        if explainer_type is not None:
            # Use the forced explainer type
            explainer, shap_values = self._create_shap_explainer(
                explainer_type, X_processed, X_to_explain, background_samples, **kwargs
            )
        else:
            # Automatically determine the best explainer
            explainer_map = {
                'random_forest': 'tree',
                'gradient_boosting': 'tree',
                'xgboost': 'tree',
                'logistic_regression': 'linear',
                'linear_regression': 'linear',
                'svm': 'kernel',
                'cnn': 'deep',
                'lstm': 'deep',
                'transformer': 'deep',
                'neural_network': 'deep'
            }
            
            explainer_type = explainer_map.get(self.model_type, 'kernel')
            explainer, shap_values = self._create_shap_explainer(
                explainer_type, X_processed, X_to_explain, background_samples, **kwargs
            )
            
        # Store SHAP values and explainer
        self.shap_values = {
            'values': shap_values,
            'explainer': explainer,
            'X_to_explain': X_to_explain,
            'feature_names': feature_names,
            'sample_indices': sample_indices
        }
        
        self.explanation_results['shap'] = {
            'explainer_type': explainer_type,
            'n_samples': X_to_explain.shape[0]
        }
        
        self.logger.info(f"SHAP values computed using {explainer_type} explainer")
        
        return self.shap_values
    
    def _create_shap_explainer(self, explainer_type, X_all, X_to_explain, background_samples, **kwargs):
        """
        Create the appropriate SHAP explainer based on the model type.
        
        Parameters:
        -----------
        explainer_type : str
            Type of explainer to create
        X_all : numpy.ndarray
            Complete feature matrix
        X_to_explain : numpy.ndarray
            Data points to explain
        background_samples : int
            Number of background samples to use
        **kwargs :
            Additional parameters
            
        Returns:
        --------
        tuple
            (explainer, shap_values)
        """
        # Try to create the explainer based on the type
        try:
            if explainer_type == 'tree':
                # For tree-based models like Random Forest, GBM, XGBoost
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_to_explain)
                
            elif explainer_type == 'linear':
                # For linear models
                explainer = shap.LinearExplainer(self.model, X_all)
                shap_values = explainer.shap_values(X_to_explain)
                
            elif explainer_type == 'deep':
                # For deep learning models
                if self.task_type == 'classification':
                    # Create a background dataset
                    if background_samples < X_all.shape[0]:
                        background = shap.sample(X_all, background_samples)
                    else:
                        background = X_all
                        
                    explainer = shap.DeepExplainer(self.model, background)
                    shap_values = explainer.shap_values(X_to_explain)
                else:
                    # For regression models, GradientExplainer sometimes works better
                    if background_samples < X_all.shape[0]:
                        background = shap.sample(X_all, background_samples)
                    else:
                        background = X_all
                        
                    explainer = shap.GradientExplainer(self.model, background)
                    shap_values = explainer.shap_values(X_to_explain)
                    
            elif explainer_type == 'gradient':
                # Specifically for deep learning, using gradients
                if background_samples < X_all.shape[0]:
                    background = shap.sample(X_all, background_samples)
                else:
                    background = X_all
                    
                explainer = shap.GradientExplainer(self.model, background)
                shap_values = explainer.shap_values(X_to_explain)
                
            elif explainer_type == 'kernel':
                # Fallback for any model type, but computationally expensive
                # For KernelExplainer, we need a function that predicts
                if self.task_type == 'classification':
                    # For classification, we want probability outputs
                    def predict_fn(x):
                        return self.model.predict_proba(x)
                else:
                    # For regression, just the prediction
                    def predict_fn(x):
                        return self.model.predict(x)
                        
                # Create a background dataset
                if background_samples < X_all.shape[0]:
                    background = shap.sample(X_all, background_samples)
                else:
                    background = X_all
                    
                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values = explainer.shap_values(X_to_explain)
                
            else:
                raise ValueError(f"Unknown explainer type: {explainer_type}")
                
            return explainer, shap_values
            
        except Exception as e:
            self.logger.error(f"Error creating SHAP explainer: {str(e)}")
            self.logger.info("Falling back to KernelExplainer")
            
            # Fallback to KernelExplainer
            try:
                # Create predict function
                if self.task_type == 'classification':
                    # Try to get probability outputs
                    try:
                        def predict_fn(x):
                            return self.model.predict_proba(x)
                    except:
                        # If predict_proba is not available
                        def predict_fn(x):
                            return self.model.predict(x)
                else:
                    def predict_fn(x):
                        return self.model.predict(x)
                        
                # Create a background dataset
                if background_samples < X_all.shape[0]:
                    background = shap.sample(X_all, background_samples)
                else:
                    background = X_all
                    
                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values = explainer.shap_values(X_to_explain)
                
                return explainer, shap_values
                
            except Exception as e2:
                self.logger.error(f"Error with fallback KernelExplainer: {str(e2)}")
                raise
    
    def _compute_shap_importance(self, **kwargs):
        """
        Compute feature importance from SHAP values.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature importances based on SHAP values
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
            
        shap_values = self.shap_values['values']
        feature_names = self.shap_values['feature_names']
        
        # Different handling based on shape of SHAP values
        if isinstance(shap_values, list):
            # For multi-class classification, SHAP returns a list of arrays
            # We'll use the mean absolute value across all classes
            mean_abs_shap = np.mean([np.abs(shap_arr).mean(axis=0) for shap_arr in shap_values], axis=0)
            importances = mean_abs_shap
        else:
            # For binary classification or regression
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importances = mean_abs_shap
            
        # Create DataFrame with feature names
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def compute_lime_explanation(self, X, instances=None, n_samples=5, **kwargs):
        """
        Compute LIME explanations for specific instances.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix (used for training the LIME explainer)
        instances : pandas.DataFrame, numpy.ndarray, or list of indices
            Specific instances to explain
        n_samples : int
            Number of random instances to explain if instances is None
        **kwargs :
            Additional parameters for LIME
            
        Returns:
        --------
        dict
            Dictionary of LIME explanations
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME not installed. Install with 'pip install lime'")
            
        # Process input data
        X_processed, feature_names = self._preprocess_features(X)
        self.feature_names = feature_names
        
        # Select instances to explain
        if instances is None:
            # Randomly select n_samples instances
            indices = np.random.choice(X_processed.shape[0], size=n_samples, replace=False)
            X_to_explain = X_processed[indices]
        elif isinstance(instances, (list, np.ndarray)) and all(isinstance(i, (int, np.integer)) for i in instances):
            # instances is a list of indices
            indices = instances
            X_to_explain = X_processed[indices]
        else:
            # instances is actual data
            X_to_explain = instances
            indices = kwargs.get('indices', list(range(len(X_to_explain))))
            
        # Extract LIME parameters
        num_features = kwargs.get('num_features', min(10, X_processed.shape[1]))
        kernel_width = kwargs.get('kernel_width', 0.75)
        
        self.logger.info(f"Computing LIME explanations for {len(X_to_explain)} instances")
        
        # Initialize LIME explainer based on task type
        if self.task_type == 'classification':
            mode = 'classification'
            
            # Try to get class names if available
            if hasattr(self.model, 'classes_'):
                class_names = [str(c) for c in self.model.classes_]
            else:
                # Use default class names
                if hasattr(self, 'n_classes'):
                    class_names = [str(i) for i in range(self.n_classes)]
                else:
                    class_names = ['0', '1']  # Default for binary classification
        else:
            mode = 'regression'
            class_names = None
            
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_processed,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode,
            kernel_width=kernel_width,
            random_state=kwargs.get('random_state', 42)
        )
        
        # For deep learning models, create a prediction function
        if self.model_type in ['cnn', 'lstm', 'transformer', 'neural_network']:
            if self.task_type == 'classification':
                def predict_fn(x):
                    # Get probability predictions
                    probs = self.model.predict(x)
                    # Make sure they're 2D
                    if len(probs.shape) < 2:
                        probs = probs.reshape(-1, 1)
                        # Add the opposite class probability for binary
                        return np.hstack([1 - probs, probs])
                    return probs
            else:
                def predict_fn(x):
                    return self.model.predict(x)
        else:
            # Use model's predict method directly
            if self.task_type == 'classification':
                def predict_fn(x):
                    return self.model.predict_proba(x)
            else:
                def predict_fn(x):
                    return self.model.predict(x)
                    
        # Compute explanations for each instance
        explanations = {}
        
        for i, instance in enumerate(X_to_explain):
            try:
                idx = indices[i] if i < len(indices) else i
                
                self.logger.info(f"  Computing LIME explanation for instance {idx}")
                
                # Generate explanation
                exp = explainer.explain_instance(
                    instance, 
                    predict_fn,
                    num_features=num_features,
                    **{k: v for k, v in kwargs.items() if k not in ['num_features', 'random_state', 'indices']}
                )
                
                # Extract and store the explanation
                if self.task_type == 'classification':
                    # For classification, get explanation for each class
                    class_explanations = {}
                    
                    # Get the prediction from the model
                    if len(instance.shape) == 1:
                        instance_reshaped = instance.reshape(1, -1)
                    else:
                        instance_reshaped = instance
                        
                    prediction = predict_fn(instance_reshaped)
                    predicted_class = np.argmax(prediction[0])
                    
                    # Get explanation for the predicted class
                    explanation_data = exp.as_list(label=predicted_class)
                    feature_weights = {feature: weight for feature, weight in explanation_data}
                    
                    class_explanations[str(predicted_class)] = {
                        'feature_weights': feature_weights,
                        'explanation_data': explanation_data,
                        'probability': prediction[0][predicted_class]
                    }
                    
                    explanations[idx] = {
                        'class_explanations': class_explanations,
                        'predicted_class': str(predicted_class),
                        'explainer': exp
                    }
                else:
                    # For regression, simpler explanation
                    explanation_data = exp.as_list()
                    feature_weights = {feature: weight for feature, weight in explanation_data}
                    
                    explanations[idx] = {
                        'feature_weights': feature_weights,
                        'explanation_data': explanation_data,
                        'predicted_value': float(predict_fn(instance.reshape(1, -1))[0]),
                        'explainer': exp
                    }
            except Exception as e:
                self.logger.error(f"Error computing LIME explanation for instance {idx}: {str(e)}")
                explanations[idx] = {'error': str(e)}
                
        # Store LIME explanations
        self.lime_explanations = {
            'explanations': explanations,
            'explainer': explainer,
            'feature_names': feature_names,
            'mode': mode
        }
        
        self.explanation_results['lime'] = {
            'n_instances': len(explanations),
            'num_features': num_features
        }
        
        self.logger.info("LIME explanations computed successfully")
        
        return self.lime_explanations
    
    def compute_attention_weights(self, X, attention_layer_name=None, **kwargs):
        """
        Extract attention weights from transformer or LSTM models.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        attention_layer_name : str, optional
            Name of the attention layer. If None, will attempt to find it automatically.
        **kwargs :
            Additional parameters
            
        Returns:
        --------
        dict
            Dictionary with attention weights
        """
        # Check if model is compatible
        if self.model_type not in ['transformer', 'lstm', 'neural_network']:
            raise ValueError(f"Attention weights extraction is only for transformer/LSTM models, not {self.model_type}")
            
        # Check if TensorFlow is available
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not installed. Install with 'pip install tensorflow'")
            
        # Process input data
        X_processed, feature_names = self._preprocess_features(X)
        self.feature_names = feature_names
        
        # Extract parameters
        n_samples = kwargs.get('n_samples', min(100, X_processed.shape[0]))
        
        # Randomly select samples if needed
        if n_samples < X_processed.shape[0]:
            indices = np.random.choice(X_processed.shape[0], size=n_samples, replace=False)
            X_samples = X_processed[indices]
        else:
            indices = np.arange(X_processed.shape[0])
            X_samples = X_processed
            
        self.logger.info(f"Extracting attention weights for {len(X_samples)} samples")
        
        # Find attention layer if not specified
        if attention_layer_name is None:
            attention_layer_name = self._find_attention_layer()
            
        if attention_layer_name is None:
            raise ValueError("Could not find attention layer. Please specify attention_layer_name.")
            
        # Create a model that outputs attention weights
        attention_model = self._create_attention_model(attention_layer_name)
        
        # Get attention weights
        attention_weights = attention_model.predict(X_samples)
        
        # Store attention weights
        self.attention_weights = {
            'weights': attention_weights,
            'layer_name': attention_layer_name,
            'indices': indices,
            'feature_names': feature_names
        }
        
        self.explanation_results['attention'] = {
            'layer_name': attention_layer_name,
            'n_samples': len(X_samples)
        }
        
        self.logger.info(f"Attention weights extracted from layer {attention_layer_name}")
        
        return self.attention_weights
    
    def _find_attention_layer(self):
        """
        Find the attention layer in the model.
        
        Returns:
        --------
        str or None
            Name of the attention layer, or None if not found
        """
        # Check if it's a TensorFlow model
        if not hasattr(self.model, 'layers'):
            return None
            
        # Look for layers with 'attention' in the name or class
        for layer in self.model.layers:
            layer_name = layer.name.lower()
            layer_class = layer.__class__.__name__.lower()
            
            if 'attention' in layer_name or 'attention' in layer_class:
                return layer.name
                
        # Look for multi-head attention
        for layer in self.model.layers:
            layer_class = layer.__class__.__name__.lower()
            if 'multihead' in layer_class or 'multi_head' in layer_class:
                return layer.name
                
        # Look for layers with attention as output
        for layer in self.model.layers:
            if hasattr(layer, 'output_shape') and layer.output_shape and len(layer.output_shape) > 2:
                # Attention layers often have output shape (batch_size, sequence_length, sequence_length)
                return layer.name
                
        return None
    
    def _create_attention_model(self, attention_layer_name):
        """
        Create a model that outputs attention weights.
        
        Parameters:
        -----------
        attention_layer_name : str
            Name of the attention layer
            
        Returns:
        --------
        tf.keras.Model
            Model that outputs attention weights
        """
        # Get the attention layer
        attention_layer = None
        for layer in self.model.layers:
            if layer.name == attention_layer_name:
                attention_layer = layer
                break
                
        if attention_layer is None:
            raise ValueError(f"Could not find layer with name {attention_layer_name}")
            
        # Create a model that outputs the attention weights
        attention_model = Model(
            inputs=self.model.inputs,
            outputs=attention_layer.output
        )
        
        return attention_model
    
    def plot_feature_importance(self, n_features=20, **kwargs):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        n_features : int
            Number of top features to display
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size
            - title: Plot title
            - color: Bar color
            - show_values: Whether to display importance values
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.feature_importances is None:
            raise ValueError("Feature importances not computed. Call compute_feature_importance() first.")
            
        # Extract parameters
        figsize = kwargs.get('figsize', (10, max(5, n_features / 2)))
        title = kwargs.get('title', 'Feature Importance')
        color = kwargs.get('color', 'skyblue')
        show_values = kwargs.get('show_values', True)
        
        # Get top features
        top_features = self.feature_importances.head(n_features).copy()
        
        # Sort features for better visualization
        top_features = top_features.sort_values('importance')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        bars = ax.barh(top_features['feature'], top_features['importance'], color=color)
        
        # Add importance values if requested
        if show_values:
            for bar in bars:
                width = bar.get_width()
                ax.text(width * 1.01, bar.get_y() + bar.get_height() / 2,
                       f'{width:.3f}', va='center')
                
        # Customize plot
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_shap_summary(self, plot_type='bar', **kwargs):
        """
        Create SHAP summary plot.
        
        Parameters:
        -----------
        plot_type : str
            Type of SHAP summary plot:
            - 'bar': Bar chart of feature importance
            - 'dot': Dot plot showing feature effects
            - 'violin': Violin plot showing feature effects
            - 'layered_violin': Layered violin plot (only for classification)
        **kwargs :
            Additional parameters:
            - max_display: Maximum number of features to show
            - class_index: For multi-class, which class to show
            - plot_size: Tuple of (width, height) for the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with 'pip install shap'")
            
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
            
        # Extract SHAP data
        shap_values = self.shap_values['values']
        X = self.shap_values['X_to_explain']
        feature_names = self.shap_values['feature_names']
        
        # Extract parameters
        max_display = kwargs.get('max_display', 20)
        plot_size = kwargs.get('plot_size', (12, 12))
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # This is for multi-class classification
            class_index = kwargs.get('class_index', 0)
            
            if class_index >= len(shap_values):
                raise ValueError(f"Class index {class_index} out of range. Only {len(shap_values)} classes available.")
                
            # Use the specified class
            values_to_plot = shap_values[class_index]
            class_label = f"Class {class_index}"
        else:
            # This is for binary classification or regression
            values_to_plot = shap_values
            class_label = None
            
        # Create figure
        plt.figure(figsize=plot_size)
        
        # Create appropriate plot based on type
        if plot_type == 'bar':
            # Bar summary plot (just importance, no directionality)
            shap.summary_plot(
                values_to_plot, X, 
                feature_names=feature_names,
                max_display=max_display,
                plot_type='bar',
                show=False
            )
            title = f"SHAP Feature Importance"
            if class_label:
                title += f" - {class_label}"
                
            plt.title(title)
            
        elif plot_type == 'dot':
            # Dot summary plot (shows importance and directionality)
            shap.summary_plot(
                values_to_plot, X, 
                feature_names=feature_names,
                max_display=max_display,
                plot_type='dot',
                show=False
            )
            title = f"SHAP Feature Effects"
            if class_label:
                title += f" - {class_label}"
                
            plt.title(title)
            
        elif plot_type == 'violin':
            # Violin summary plot (shows distribution of feature effects)
            shap.summary_plot(
                values_to_plot, X, 
                feature_names=feature_names,
                max_display=max_display,
                plot_type='violin',
                show=False
            )
            title = f"SHAP Feature Effect Distribution"
            if class_label:
                title += f" - {class_label}"
                
            plt.title(title)
            
        elif plot_type == 'layered_violin':
            # Layered violin plot (for multi-class)
            if not isinstance(shap_values, list):
                raise ValueError("Layered violin plot is only for multi-class classification")
                
            # Get class names or indices
            if 'class_names' in kwargs:
                class_names = kwargs['class_names']
            else:
                class_names = [f"Class {i}" for i in range(len(shap_values))]
                
            # Call layered violin plot
            shap.summary_plot(
                shap_values, X, 
                feature_names=feature_names,
                max_display=max_display,
                plot_type='violin',
                class_names=class_names,
                show=False
            )
            plt.title("SHAP Feature Effects by Class")
            
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
        # Get the current figure
        fig = plt.gcf()
        
        return fig
    
    def plot_shap_dependence(self, feature, interaction_feature='auto', **kwargs):
        """
        Create SHAP dependence plot to show how a feature's impact changes with its value.
        
        Parameters:
        -----------
        feature : str
            Feature to analyze
        interaction_feature : str
            Feature to use for coloring points (or 'auto' to select automatically)
        **kwargs :
            Additional parameters:
            - class_index: For multi-class, which class to show
            - plot_size: Tuple of (width, height) for the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with 'pip install shap'")
            
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
            
        # Extract SHAP data
        shap_values = self.shap_values['values']
        X = self.shap_values['X_to_explain']
        feature_names = self.shap_values['feature_names']
        
        # Extract parameters
        plot_size = kwargs.get('plot_size', (10, 7))
        
        # Check if the feature exists
        if feature not in feature_names:
            raise ValueError(f"Feature '{feature}' not found. Available features: {feature_names}")
            
        # Get feature index
        feature_idx = list(feature_names).index(feature)
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # This is for multi-class classification
            class_index = kwargs.get('class_index', 0)
            
            if class_index >= len(shap_values):
                raise ValueError(f"Class index {class_index} out of range. Only {len(shap_values)} classes available.")
                
            # Use the specified class
            values_to_plot = shap_values[class_index]
            class_label = f"Class {class_index}"
        else:
            # This is for binary classification or regression
            values_to_plot = shap_values
            class_label = None
            
        # Create figure
        plt.figure(figsize=plot_size)
        
        # If interaction feature is 'auto', let SHAP decide
        if interaction_feature != 'auto':
            # Check if the interaction feature exists
            if interaction_feature not in feature_names:
                raise ValueError(f"Interaction feature '{interaction_feature}' not found. Available features: {feature_names}")
                
            # Get interaction feature index
            interaction_idx = list(feature_names).index(interaction_feature)
        else:
            interaction_idx = 'auto'
            
        # Create dependence plot
        shap.dependence_plot(
            feature_idx, values_to_plot, X,
            interaction_index=interaction_idx,
            feature_names=feature_names,
            show=False
        )
        
        # Add title
        title = f"SHAP Dependence: {feature}"
        if interaction_feature != 'auto':
            title += f" (interaction: {interaction_feature})"
        if class_label:
            title += f" - {class_label}"
            
        plt.title(title)
        
        # Get the current figure
        fig = plt.gcf()
        
        return fig
    
    def plot_shap_waterfall(self, instance_index=0, **kwargs):
        """
        Create SHAP waterfall plot to explain a single prediction.
        
        Parameters:
        -----------
        instance_index : int
            Index of the instance to explain
        **kwargs :
            Additional parameters:
            - class_index: For multi-class, which class to show
            - plot_size: Tuple of (width, height) for the plot
            - max_display: Maximum number of features to display
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with 'pip install shap'")
            
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
            
        # Extract SHAP data
        shap_values = self.shap_values['values']
        X = self.shap_values['X_to_explain']
        feature_names = self.shap_values['feature_names']
        
        # Extract parameters
        plot_size = kwargs.get('plot_size', (10, 12))
        max_display = kwargs.get('max_display', 20)
        
        # Check if the instance exists
        if instance_index >= X.shape[0]:
            raise ValueError(f"Instance index {instance_index} out of range. Only {X.shape[0]} instances available.")
            
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # This is for multi-class classification
            class_index = kwargs.get('class_index', 0)
            
            if class_index >= len(shap_values):
                raise ValueError(f"Class index {class_index} out of range. Only {len(shap_values)} classes available.")
                
            # Use the specified class
            values_to_plot = shap_values[class_index][instance_index]
            class_label = f"Class {class_index}"
            
            # Get expected value
            if hasattr(self.shap_values['explainer'], 'expected_value') and self.shap_values['explainer'].expected_value is not None:
                expected_value = self.shap_values['explainer'].expected_value[class_index]
            else:
                expected_value = 0
        else:
            # This is for binary classification or regression
            values_to_plot = shap_values[instance_index]
            class_label = None
            
            # Get expected value
            if hasattr(self.shap_values['explainer'], 'expected_value') and self.shap_values['explainer'].expected_value is not None:
                expected_value = self.shap_values['explainer'].expected_value
            else:
                expected_value = 0
                
        # Create figure
        plt.figure(figsize=plot_size)
        
        # Create waterfall plot
        shap.plots._waterfall.waterfall_legacy(
            expected_value,
            values_to_plot,
            features=X[instance_index],
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        
        # Add title
        title = f"SHAP Waterfall Plot - Instance {instance_index}"
        if class_label:
            title += f" - {class_label}"
            
        plt.title(title)
        
        # Get the current figure
        fig = plt.gcf()
        
        return fig
    
    def plot_lime_explanation(self, instance_index=0, **kwargs):
        """
        Plot LIME explanation for a specific instance.
        
        Parameters:
        -----------
        instance_index : int
            Index of the instance to explain
        **kwargs :
            Additional parameters:
            - class_index: For multi-class, which class to show
            - plot_size: Tuple of (width, height) for the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME not installed. Install with 'pip install lime'")
            
        if not self.lime_explanations or 'explanations' not in self.lime_explanations:
            raise ValueError("LIME explanations not computed. Call compute_lime_explanation() first.")
            
        # Check if the instance exists
        if instance_index not in self.lime_explanations['explanations']:
            raise ValueError(f"Instance index {instance_index} not found in LIME explanations.")
            
        # Get the instance explanation
        exp_data = self.lime_explanations['explanations'][instance_index]
        
        # If there was an error with this instance
        if 'error' in exp_data:
            plt.figure()
            plt.text(0.5, 0.5, f"Error in LIME explanation: {exp_data['error']}", 
                    ha='center', va='center')
            plt.title(f"LIME Explanation Error - Instance {instance_index}")
            return plt.gcf()
            
        # Get the explainer object
        exp = exp_data['explainer']
        
        # Extract parameters
        plot_size = kwargs.get('plot_size', (10, 7))
        
        # For classification, handle multiple classes
        if self.task_type == 'classification':
            # Get predicted class
            predicted_class = exp_data['predicted_class']
            
            # Check if user specified a different class
            class_index = kwargs.get('class_index', None)
            if class_index is not None:
                # Convert to string for consistency
                class_index = str(class_index)
                
                # Check if this class exists in explanations
                if class_index not in exp_data['class_explanations']:
                    # Try to get explanation for this class
                    try:
                        exp.as_list(label=int(class_index))
                        # If successful, it means the class exists
                        predicted_class = class_index
                    except:
                        self.logger.warning(f"Class {class_index} not found. Using predicted class {predicted_class} instead.")
                else:
                    predicted_class = class_index
                    
            # Create figure
            plt.figure(figsize=plot_size)
            
            # Plot the explanation
            exp.as_pyplot_figure(label=int(predicted_class))
            
            # Update the title
            plt.title(f"LIME Explanation - Instance {instance_index} - Class {predicted_class}")
        else:
            # Regression is simpler
            plt.figure(figsize=plot_size)
            
            # Plot the explanation
            exp.as_pyplot_figure()
            
            # Update the title
            plt.title(f"LIME Explanation - Instance {instance_index}")
            
        # Get the current figure
        fig = plt.gcf()
        
        return fig
    
    def plot_attention_heatmap(self, instance_index=0, **kwargs):
        """
        Plot attention weights heatmap for a specific instance.
        
        Parameters:
        -----------
        instance_index : int
            Index of the instance to visualize
        **kwargs :
            Additional parameters:
            - figsize: Figure size
            - cmap: Colormap to use
            - title: Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.attention_weights is None:
            raise ValueError("Attention weights not computed. Call compute_attention_weights() first.")
            
        # Get attention weights
        weights = self.attention_weights['weights']
        indices = self.attention_weights['indices']
        feature_names = self.attention_weights['feature_names']
        
        # Find the index in the original array
        if instance_index >= len(indices):
            raise ValueError(f"Instance index {instance_index} out of range. Only {len(indices)} instances available.")
            
        data_index = instance_index
        
        # Extract parameters
        figsize = kwargs.get('figsize', (12, 10))
        cmap = kwargs.get('cmap', 'viridis')
        title = kwargs.get('title', f"Attention Weights - Instance {data_index}")
        
        # Get the attention weights for this instance
        instance_weights = weights[instance_index]
        
        # Handle different attention weight shapes
        if len(instance_weights.shape) == 1:
            # 1D attention weights (e.g., feature attention)
            # Create a bar chart
            plt.figure(figsize=figsize)
            
            # Plot feature attention
            attention_df = pd.DataFrame({
                'feature': feature_names,
                'attention': instance_weights
            }).sort_values('attention', ascending=False)
            
            plt.bar(range(len(attention_df)), attention_df['attention'], color='skyblue')
            plt.xticks(range(len(attention_df)), attention_df['feature'], rotation=90)
            plt.xlabel('Feature')
            plt.ylabel('Attention Weight')
            plt.title(title)
            plt.tight_layout()
            
        elif len(instance_weights.shape) == 2:
            # 2D attention weights (e.g., self-attention in transformers)
            # Create a heatmap
            plt.figure(figsize=figsize)
            
            # Ensure we have feature names
            if len(feature_names) != instance_weights.shape[0]:
                # Use generic names based on size
                feature_names = [f"F{i}" for i in range(instance_weights.shape[0])]
                
            sns.heatmap(
                instance_weights, 
                cmap=cmap, 
                xticklabels=feature_names,
                yticklabels=feature_names,
                annot=kwargs.get('annot', instance_weights.shape[0] < 20),
                fmt='.2f'
            )
            plt.title(title)
            plt.tight_layout()
            
        elif len(instance_weights.shape) == 3:
            # 3D attention weights (e.g., multi-head attention)
            # Create a set of heatmaps for each attention head
            n_heads = instance_weights.shape[0]
            
            # Calculate grid dimensions
            grid_size = int(np.ceil(np.sqrt(n_heads)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
            
            # Ensure axes is a 2D array
            if n_heads == 1:
                axes = np.array([[axes]])
            elif grid_size == 1:
                axes = axes.reshape(1, -1)
                
            # Flatten for easy indexing
            axes_flat = axes.flatten()
            
            # Ensure we have feature names
            if len(feature_names) != instance_weights.shape[1]:
                # Use generic names based on size
                feature_names = [f"F{i}" for i in range(instance_weights.shape[1])]
                
            # Plot each attention head
            for i in range(n_heads):
                if i < len(axes_flat):
                    ax = axes_flat[i]
                    sns.heatmap(
                        instance_weights[i], 
                        cmap=cmap, 
                        xticklabels=feature_names if i == n_heads - 1 else [],
                        yticklabels=feature_names if i % grid_size == 0 else [],
                        annot=kwargs.get('annot', instance_weights.shape[1] < 10),
                        fmt='.2f',
                        ax=ax
                    )
                    ax.set_title(f"Head {i+1}")
                    
            # Remove empty subplots
            for i in range(n_heads, len(axes_flat)):
                fig.delaxes(axes_flat[i])
                
            # Add overall title
            fig.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
        else:
            raise ValueError(f"Unsupported attention weights shape: {instance_weights.shape}")
            
        # Return the current figure
        return plt.gcf()
    
    def plot_attention_aggregated(self, top_features=10, **kwargs):
        """
        Plot aggregated attention weights across all instances.
        
        Parameters:
        -----------
        top_features : int
            Number of top features to display
        **kwargs :
            Additional parameters:
            - figsize: Figure size
            - title: Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.attention_weights is None:
            raise ValueError("Attention weights not computed. Call compute_attention_weights() first.")
            
        # Get attention weights
        weights = self.attention_weights['weights']
        feature_names = self.attention_weights['feature_names']
        
        # Extract parameters
        figsize = kwargs.get('figsize', (10, 8))
        title = kwargs.get('title', "Aggregated Attention Weights")
        
        # Handle different attention weight shapes
        if len(weights.shape) == 2:
            # 2D: (instances, features)
            # Calculate mean attention for each feature
            mean_attention = np.mean(weights, axis=0)
            
            # Create DataFrame with feature names
            attention_df = pd.DataFrame({
                'feature': feature_names,
                'attention': mean_attention
            }).sort_values('attention', ascending=False)
            
            # Keep only top features
            attention_df = attention_df.head(top_features)
            
            # Create a bar chart
            plt.figure(figsize=figsize)
            plt.bar(attention_df['feature'], attention_df['attention'], color='skyblue')
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Feature')
            plt.ylabel('Mean Attention Weight')
            plt.title(title)
            plt.tight_layout()
            
        elif len(weights.shape) == 3:
            # 3D: (instances, sequence_length, sequence_length)
            # Calculate mean attention matrix
            mean_attention = np.mean(weights, axis=0)
            
            # Create a heatmap
            plt.figure(figsize=figsize)
            
            # Ensure we have feature names
            if len(feature_names) != mean_attention.shape[0]:
                # Use generic names based on size
                feature_names = [f"F{i}" for i in range(mean_attention.shape[0])]
                
            # Create the heatmap
            sns.heatmap(
                mean_attention, 
                cmap='viridis', 
                xticklabels=feature_names,
                yticklabels=feature_names,
                annot=kwargs.get('annot', mean_attention.shape[0] < 20),
                fmt='.2f'
            )
            plt.title(title)
            plt.tight_layout()
            
        elif len(weights.shape) == 4:
            # 4D: (instances, heads, sequence_length, sequence_length)
            # Calculate mean attention for each head
            mean_attention = np.mean(weights, axis=0)  # (heads, seq_len, seq_len)
            n_heads = mean_attention.shape[0]
            
            # Calculate grid dimensions
            grid_size = int(np.ceil(np.sqrt(n_heads)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
            
            # Ensure axes is a 2D array
            if n_heads == 1:
                axes = np.array([[axes]])
            elif grid_size == 1:
                axes = axes.reshape(1, -1)
                
            # Flatten for easy indexing
            axes_flat = axes.flatten()
            
            # Ensure we have feature names
            if len(feature_names) != mean_attention.shape[1]:
                # Use generic names based on size
                feature_names = [f"F{i}" for i in range(mean_attention.shape[1])]
                
            # Plot each attention head
            for i in range(n_heads):
                if i < len(axes_flat):
                    ax = axes_flat[i]
                    sns.heatmap(
                        mean_attention[i], 
                        cmap='viridis', 
                        xticklabels=feature_names if i == n_heads - 1 else [],
                        yticklabels=feature_names if i % grid_size == 0 else [],
                        annot=kwargs.get('annot', mean_attention.shape[1] < 10),
                        fmt='.2f',
                        ax=ax
                    )
                    ax.set_title(f"Head {i+1}")
                    
            # Remove empty subplots
            for i in range(n_heads, len(axes_flat)):
                fig.delaxes(axes_flat[i])
                
            # Add overall title
            fig.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
        else:
            raise ValueError(f"Unsupported attention weights shape: {weights.shape}")
            
        # Return the current figure
        return plt.gcf()
    
    def _preprocess_features(self, X):
        """
        Preprocess feature matrix and extract feature names.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
            
        Returns:
        --------
        tuple
            (X_processed, feature_names)
        """
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_processed = X.values
        else:
            X_processed = X
            if self.feature_names is not None:
                feature_names = self.feature_names
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
        return X_processed, feature_names
    
    def generate_explanation_report(self, X, y=None, output_format='html', **kwargs):
        """
        Generate a comprehensive explanation report combining multiple techniques.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray, optional
            The target variable
        output_format : str
            Format of the report ('html', 'json', or 'dict')
        **kwargs :
            Additional parameters for the explanations
            
        Returns:
        --------
        str or dict
            Report in the specified format
        """
        # Process input data
        X_processed, feature_names = self._preprocess_features(X)
        self.feature_names = feature_names
        
        # Extract parameters
        n_samples = kwargs.get('n_samples', min(5, X_processed.shape[0]))
        compute_importance = kwargs.get('compute_importance', True)
        compute_shap = kwargs.get('compute_shap', True)
        compute_lime = kwargs.get('compute_lime', True)
        compute_attention = kwargs.get('compute_attention', 
                                      self.model_type in ['transformer', 'lstm'])
                                      
        self.logger.info(f"Generating explanation report")
        
        # Create the report structure
        report = {
            'model_type': self.model_type,
            'task_type': self.task_type,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'n_samples': n_samples,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'explanations': {}
        }
        
        # 1. Feature Importance
        if compute_importance:
            self.logger.info("Computing feature importance...")
            importance_method = kwargs.get('importance_method', 'permutation')
            
            try:
                importances = self.compute_feature_importance(
                    X, y, method=importance_method, **kwargs
                )
                
                # Add to report
                report['explanations']['feature_importance'] = {
                    'method': importance_method,
                    'top_features': importances.head(20).to_dict('records')
                }
            except Exception as e:
                self.logger.error(f"Error computing feature importance: {str(e)}")
                report['explanations']['feature_importance'] = {
                    'error': str(e)
                }
                
        # 2. SHAP Analysis
        if compute_shap and SHAP_AVAILABLE:
            self.logger.info("Computing SHAP values...")
            try:
                # Compute SHAP values
                shap_results = self.compute_shap_values(
                    X, y, n_samples=n_samples, **kwargs
                )
                
                # Compute SHAP-based importance
                shap_importance = self._compute_shap_importance()
                
                # Add to report
                report['explanations']['shap'] = {
                    'success': True,
                    'explainer_type': self.explanation_results['shap']['explainer_type'],
                    'top_features': shap_importance.head(20).to_dict('records')
                }
                
                # Add sample explanations
                sample_explanations = {}
                for i in range(min(n_samples, len(shap_results['X_to_explain']))):
                    sample_idx = shap_results['sample_indices'][i] if shap_results['sample_indices'] is not None else i
                    
                    # Get the SHAP values for this instance
                    if isinstance(shap_results['values'], list):
                        # Multi-class classification
                        sample_values = {
                            f"class_{j}": values[i].tolist() 
                            for j, values in enumerate(shap_results['values'])
                        }
                    else:
                        # Binary classification or regression
                        sample_values = shap_results['values'][i].tolist()
                        
                    sample_explanations[str(sample_idx)] = {
                        'shap_values': sample_values
                    }
                    
                report['explanations']['shap']['samples'] = sample_explanations
                
            except Exception as e:
                self.logger.error(f"Error computing SHAP values: {str(e)}")
                report['explanations']['shap'] = {
                    'success': False,
                    'error': str(e)
                }
                
        # 3. LIME Explanations
        if compute_lime and LIME_AVAILABLE:
            self.logger.info("Computing LIME explanations...")
            try:
                # Compute LIME explanations
                lime_results = self.compute_lime_explanation(
                    X, n_samples=n_samples, **kwargs
                )
                
                # Add to report
                report['explanations']['lime'] = {
                    'success': True,
                    'mode': lime_results['mode'],
                    'samples': {}
                }
                
                # Extract explanations for each sample
                for idx, explanation in lime_results['explanations'].items():
                    if 'error' in explanation:
                        report['explanations']['lime']['samples'][str(idx)] = {
                            'success': False,
                            'error': explanation['error']
                        }
                    else:
                        if self.task_type == 'classification':
                            # For classification, get the predicted class explanation
                            pred_class = explanation['predicted_class']
                            class_exp = explanation['class_explanations'][pred_class]
                            
                            report['explanations']['lime']['samples'][str(idx)] = {
                                'success': True,
                                'predicted_class': pred_class,
                                'probability': float(class_exp['probability']),
                                'feature_weights': class_exp['feature_weights']
                            }
                        else:
                            # For regression
                            report['explanations']['lime']['samples'][str(idx)] = {
                                'success': True,
                                'predicted_value': explanation['predicted_value'],
                                'feature_weights': explanation['feature_weights']
                            }
            except Exception as e:
                self.logger.error(f"Error computing LIME explanations: {str(e)}")
                report['explanations']['lime'] = {
                    'success': False,
                    'error': str(e)
                }
                
        # 4. Attention Weights (for transformer/LSTM models)
        if compute_attention and self.model_type in ['transformer', 'lstm', 'neural_network']:
            self.logger.info("Computing attention weights...")
            try:
                # Compute attention weights
                attention_results = self.compute_attention_weights(
                    X, n_samples=n_samples, **kwargs
                )
                
                # Add to report
                report['explanations']['attention'] = {
                    'success': True,
                    'layer_name': attention_results['layer_name'],
                    'samples': {}
                }
                
                # Extract attention for each sample
                for i in range(min(n_samples, len(attention_results['weights']))):
                    idx = attention_results['indices'][i]
                    
                    # Get attention weights for this instance
                    instance_weights = attention_results['weights'][i]
                    
                    # Handle different shapes
                    if len(instance_weights.shape) == 1:
                        # 1D: feature attention
                        attention_dict = {
                            feature: float(weight) 
                            for feature, weight in zip(feature_names, instance_weights)
                        }
                        
                        report['explanations']['attention']['samples'][str(idx)] = {
                            'shape': '1D',
                            'feature_attention': attention_dict
                        }
                    elif len(instance_weights.shape) == 2:
                        # 2D: attention matrix (sequence x sequence)
                        # This can be large, so we'll summarize by feature importance
                        feature_importance = np.mean(instance_weights, axis=0)
                        
                        attention_dict = {
                            feature: float(weight) 
                            for feature, weight in zip(feature_names, feature_importance)
                        }
                        
                        report['explanations']['attention']['samples'][str(idx)] = {
                            'shape': '2D',
                            'feature_attention': attention_dict,
                            'matrix_shape': instance_weights.shape
                        }
                    elif len(instance_weights.shape) == 3:
                        # 3D: multi-head attention
                        # Summarize by averaging across heads
                        avg_attention = np.mean(instance_weights, axis=0)
                        feature_importance = np.mean(avg_attention, axis=0)
                        
                        attention_dict = {
                            feature: float(weight) 
                            for feature, weight in zip(feature_names, feature_importance)
                        }
                        
                        report['explanations']['attention']['samples'][str(idx)] = {
                            'shape': '3D',
                            'feature_attention': attention_dict,
                            'n_heads': instance_weights.shape[0],
                            'matrix_shape': instance_weights.shape[1:]
                        }
            except Exception as e:
                self.logger.error(f"Error computing attention weights: {str(e)}")
                report['explanations']['attention'] = {
                    'success': False,
                    'error': str(e)
                }
                
        # Generate the requested format
        if output_format.lower() == 'html':
            return self._generate_html_report(report)
        elif output_format.lower() == 'json':
            import json
            return json.dumps(report, indent=2)
        else:
            return report
    
    def _generate_html_report(self, report_data):
        """
        Generate an HTML report from the explanation data.
        
        Parameters:
        -----------
        report_data : dict
            Dictionary with explanation data
            
        Returns:
        --------
        str
            HTML report
        """
        # Import required modules
        from base64 import b64encode
        from io import BytesIO
        
        # Start building the HTML
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>Model Explanation Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        .section { margin-bottom: 30px; }",
            "        .subsection { margin-bottom: 20px; }",
            "        table { border-collapse: collapse; width: 100%; }",
            "        th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }",
            "        th { background-color: #f2f2f2; }",
            "        tr:nth-child(even) { background-color: #f9f9f9; }",
            "        .chart { margin: 20px 0; }",
            "        .positive { color: #00a000; }",
            "        .negative { color: #a00000; }",
            "        .plot-container { text-align: center; margin: 20px 0; }",
            "        h1 { color: #2c3e50; }",
            "        h2 { color: #34495e; border-bottom: 1px solid #eee; padding-bottom: 5px; }",
            "        h3 { color: #7f8c8d; }",
            "        .sample-card { border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 10px 0; }",
            "        .error { color: red; font-style: italic; }",
            "    </style>",
            "</head>",
            "<body>",
            f"    <h1>Model Explanation Report</h1>",
            f"    <p><strong>Model Type:</strong> {report_data['model_type']}</p>",
            f"    <p><strong>Task Type:</strong> {report_data['task_type']}</p>",
            f"    <p><strong>Number of Features:</strong> {report_data['n_features']}</p>",
            f"    <p><strong>Generated:</strong> {report_data['timestamp']}</p>"
        ]
        
        # 1. Feature Importance
        html.append("    <div class='section'>")
        html.append("        <h2>Feature Importance</h2>")
        
        if 'feature_importance' in report_data['explanations']:
            importance_data = report_data['explanations']['feature_importance']
            
            if 'error' in importance_data:
                html.append(f"        <p class='error'>Error: {importance_data['error']}</p>")
            else:
                html.append(f"        <p>Method: {importance_data['method']}</p>")
                
                # Create a bar chart
                try:
                    plt.figure(figsize=(10, 6))
                    features = [item['feature'] for item in importance_data['top_features']]
                    importances = [item['importance'] for item in importance_data['top_features']]
                    
                    # Sort for better visualization
                    sorted_idx = np.argsort(importances)
                    plt.barh([features[i] for i in sorted_idx], [importances[i] for i in sorted_idx], color='skyblue')
                    plt.xlabel('Importance')
                    plt.title('Feature Importance')
                    plt.grid(axis='x', linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    
                    # Convert plot to base64 for embedding in HTML
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    plot_data = b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close()
                    
                    # Add the plot to HTML
                    html.append("        <div class='plot-container'>")
                    html.append(f"            <img src='data:image/png;base64,{plot_data}' alt='Feature Importance Plot'>")
                    html.append("        </div>")
                except Exception as e:
                    html.append(f"        <p class='error'>Error generating plot: {str(e)}</p>")
                
                # Add importance table
                html.append("        <table>")
                html.append("            <tr><th>Feature</th><th>Importance</th></tr>")
                
                for item in importance_data['top_features']:
                    html.append(f"            <tr><td>{item['feature']}</td><td>{item['importance']:.4f}</td></tr>")
                    
                html.append("        </table>")
        else:
            html.append("        <p>Feature importance not computed.</p>")
            
        html.append("    </div>")
        
        # 2. SHAP Analysis
        html.append("    <div class='section'>")
        html.append("        <h2>SHAP Analysis</h2>")
        
        if 'shap' in report_data['explanations']:
            shap_data = report_data['explanations']['shap']
            
            if not shap_data.get('success', False):
                html.append(f"        <p class='error'>Error: {shap_data.get('error', 'Unknown error')}</p>")
            else:
                html.append(f"        <p>Explainer Type: {shap_data['explainer_type']}</p>")
                
                # Create a SHAP summary plot if possible
                if SHAP_AVAILABLE and hasattr(self, 'shap_values') and self.shap_values is not None:
                    try:
                        # Create SHAP summary plot
                        plt.figure(figsize=(10, 8))
                        shap_values = self.shap_values['values']
                        X_to_explain = self.shap_values['X_to_explain']
                        feature_names = self.shap_values['feature_names']
                        
                        # Handle multi-class
                        if isinstance(shap_values, list):
                            # Use the mean across classes
                            mean_shap = np.abs(np.array(shap_values)).mean(axis=0)
                            shap.summary_plot(mean_shap, X_to_explain, feature_names=feature_names,
                                           max_display=20, plot_type='bar', show=False)
                        else:
                            # For binary classification or regression
                            shap.summary_plot(shap_values, X_to_explain, feature_names=feature_names,
                                           max_display=20, plot_type='bar', show=False)
                            
                        # Convert plot to base64
                        buffer = BytesIO()
                        plt.savefig(buffer, format='png')
                        buffer.seek(0)
                        plot_data = b64encode(buffer.getvalue()).decode('utf-8')
                        plt.close()
                        
                        # Add the plot to HTML
                        html.append("        <div class='plot-container'>")
                        html.append(f"            <img src='data:image/png;base64,{plot_data}' alt='SHAP Summary Plot'>")
                        html.append("        </div>")
                        
                        # Also add a SHAP dot plot
                        plt.figure(figsize=(10, 10))
                        
                        if isinstance(shap_values, list):
                            # Use first class for multi-class
                            shap.summary_plot(shap_values[0], X_to_explain, feature_names=feature_names,
                                           max_display=20, show=False)
                        else:
                            shap.summary_plot(shap_values, X_to_explain, feature_names=feature_names,
                                           max_display=20, show=False)
                            
                        # Convert plot to base64
                        buffer = BytesIO()
                        plt.savefig(buffer, format='png')
                        buffer.seek(0)
                        plot_data = b64encode(buffer.getvalue()).decode('utf-8')
                        plt.close()
                        
                        # Add the plot to HTML
                        html.append("        <div class='plot-container'>")
                        html.append(f"            <img src='data:image/png;base64,{plot_data}' alt='SHAP Dot Plot'>")
                        html.append("        </div>")
                    except Exception as e:
                        html.append(f"        <p class='error'>Error generating SHAP plots: {str(e)}</p>")
                
                # Add SHAP-based importance table
                html.append("        <h3>SHAP Feature Importance</h3>")
                html.append("        <table>")
                html.append("            <tr><th>Feature</th><th>Importance</th></tr>")
                
                for item in shap_data['top_features']:
                    html.append(f"            <tr><td>{item['feature']}</td><td>{item['importance']:.4f}</td></tr>")
                    
                html.append("        </table>")
                
                # Add sample explanations
                if 'samples' in shap_data:
                    html.append("        <h3>Sample SHAP Explanations</h3>")
                    
                    # Create sample cards
                    for sample_idx, sample_data in shap_data['samples'].items():
                        html.append(f"        <div class='sample-card'>")
                        html.append(f"            <h4>Sample {sample_idx}</h4>")
                        
                        # Generate SHAP waterfall plot for this sample
                        try:
                            if hasattr(self, 'shap_values') and self.shap_values is not None:
                                sample_pos = list(self.shap_values['sample_indices']).index(int(sample_idx)) \
                                            if self.shap_values['sample_indices'] is not None \
                                            else int(sample_idx)
                                
                                # Create waterfall plot
                                plt.figure(figsize=(10, 8))
                                
                                # Get expected value
                                if hasattr(self.shap_values['explainer'], 'expected_value') and \
                                   self.shap_values['explainer'].expected_value is not None:
                                    expected_value = self.shap_values['explainer'].expected_value
                                else:
                                    expected_value = 0
                                    
                                # Handle multi-class
                                if isinstance(self.shap_values['values'], list):
                                    # Use the first class for now
                                    class_idx = 0
                                    
                                    if isinstance(expected_value, (list, np.ndarray)):
                                        class_expected = expected_value[class_idx]
                                    else:
                                        class_expected = expected_value
                                        
                                    shap.plots._waterfall.waterfall_legacy(
                                        class_expected, 
                                        self.shap_values['values'][class_idx][sample_pos],
                                        feature_names=self.feature_names,
                                        max_display=10,
                                        show=False
                                    )
                                else:
                                    shap.plots._waterfall.waterfall_legacy(
                                        expected_value, 
                                        self.shap_values['values'][sample_pos],
                                        feature_names=self.feature_names,
                                        max_display=10,
                                        show=False
                                    )
                                    
                                # Convert plot to base64
                                buffer = BytesIO()
                                plt.savefig(buffer, format='png')
                                buffer.seek(0)
                                plot_data = b64encode(buffer.getvalue()).decode('utf-8')
                                plt.close()
                                
                                # Add the plot to HTML
                                html.append("            <div class='plot-container'>")
                                html.append(f"                <img src='data:image/png;base64,{plot_data}' alt='SHAP Waterfall Plot'>")
                                html.append("            </div>")
                        except Exception as e:
                            html.append(f"            <p class='error'>Error generating waterfall plot: {str(e)}</p>")
                            
                        html.append("        </div>")
        else:
            html.append("        <p>SHAP analysis not computed.</p>")
            
        html.append("    </div>")
        
        # 3. LIME Explanations
        html.append("    <div class='section'>")
        html.append("        <h2>LIME Explanations</h2>")
        
        if 'lime' in report_data['explanations']:
            lime_data = report_data['explanations']['lime']
            
            if not lime_data.get('success', False):
                html.append(f"        <p class='error'>Error: {lime_data.get('error', 'Unknown error')}</p>")
            else:
                html.append(f"        <p>Mode: {lime_data['mode']}</p>")
                
                # Add sample explanations
                if 'samples' in lime_data:
                    html.append("        <h3>Sample LIME Explanations</h3>")
                    
                    # Create sample cards
                    for sample_idx, sample_data in lime_data['samples'].items():
                        html.append(f"        <div class='sample-card'>")
                        html.append(f"            <h4>Sample {sample_idx}</h4>")
                        
                        if not sample_data.get('success', True):
                            html.append(f"            <p class='error'>Error: {sample_data.get('error', 'Unknown error')}</p>")
                        else:
                            # Show prediction
                            if 'predicted_class' in sample_data:
                                html.append(f"            <p><strong>Predicted Class:</strong> {sample_data['predicted_class']}</p>")
                                html.append(f"            <p><strong>Probability:</strong> {sample_data['probability']:.4f}</p>")
                            elif 'predicted_value' in sample_data:
                                html.append(f"            <p><strong>Predicted Value:</strong> {sample_data['predicted_value']:.4f}</p>")
                                
                            # Show feature weights
                            html.append("            <table>")
                            html.append("                <tr><th>Feature</th><th>Weight</th></tr>")
                            
                            # Sort weights for better display
                            weights = [(feature, weight) for feature, weight in sample_data['feature_weights'].items()]
                            weights.sort(key=lambda x: abs(x[1]), reverse=True)
                            
                            for feature, weight in weights:
                                if weight > 0:
                                    html.append(f"                <tr><td>{feature}</td><td class='positive'>+{weight:.4f}</td></tr>")
                                else:
                                    html.append(f"                <tr><td>{feature}</td><td class='negative'>{weight:.4f}</td></tr>")
                                    
                            html.append("            </table>")
                            
                            # Generate LIME plot if possible
                            try:
                                if hasattr(self, 'lime_explanations') and 'explanations' in self.lime_explanations:
                                    if int(sample_idx) in self.lime_explanations['explanations']:
                                        exp = self.lime_explanations['explanations'][int(sample_idx)]['explainer']
                                        
                                        plt.figure(figsize=(10, 6))
                                        
                                        if self.task_type == 'classification':
                                            exp.as_pyplot_figure(label=int(sample_data['predicted_class']))
                                        else:
                                            exp.as_pyplot_figure()
                                            
                                        # Convert plot to base64
                                        buffer = BytesIO()
                                        plt.savefig(buffer, format='png')
                                        buffer.seek(0)
                                        plot_data = b64encode(buffer.getvalue()).decode('utf-8')
                                        plt.close()
                                        
                                        # Add the plot to HTML
                                        html.append("            <div class='plot-container'>")
                                        html.append(f"                <img src='data:image/png;base64,{plot_data}' alt='LIME Explanation Plot'>")
                                        html.append("            </div>")
                            except Exception as e:
                                html.append(f"            <p class='error'>Error generating LIME plot: {str(e)}</p>")
                                
                        html.append("        </div>")
        else:
            html.append("        <p>LIME explanations not computed.</p>")
            
        html.append("    </div>")
        
        # 4. Attention Weights
        html.append("    <div class='section'>")
        html.append("        <h2>Attention Weights</h2>")
        
        if 'attention' in report_data['explanations']:
            attention_data = report_data['explanations']['attention']
            
            if not attention_data.get('success', False):
                html.append(f"        <p class='error'>Error: {attention_data.get('error', 'Unknown error')}</p>")
            else:
                html.append(f"        <p>Layer: {attention_data['layer_name']}</p>")

                # Generate aggregated attention visualization
                try:
                    if hasattr(self, 'attention_weights') and self.attention_weights is not None:
                        weights = self.attention_weights['weights']

                        # Handle different shapes
                        if len(weights.shape) == 2:
                            # 1D feature attention (instance, features)
                            # Calculate average attention
                            mean_attention = np.mean(weights, axis=0)

                            # Create bar chart
                            plt.figure(figsize=(10, 6))

                            # Get top features
                            top_n = min(20, len(self.feature_names))
                            top_indices = np.argsort(mean_attention)[-top_n:]
                            top_features = [self.feature_names[i] for i in top_indices]
                            top_values = mean_attention[top_indices]

                            # Plot
                            plt.barh(range(len(top_features)), top_values, color='skyblue')
                            plt.yticks(range(len(top_features)), top_features)
                            plt.xlabel('Mean Attention')
                            plt.title('Feature Attention')
                            plt.grid(alpha=0.3)
                            plt.tight_layout()

                            # Convert plot to base64
                            buffer = BytesIO()
                            plt.savefig(buffer, format='png')
                            buffer.seek(0)
                            plot_data = b64encode(buffer.getvalue()).decode('utf-8')
                            plt.close()

                            # Add the plot to HTML
                            html.append("        <div class='plot-container'>")
                            html.append(f"            <img src='data:image/png;base64,{plot_data}' alt='Attention Weights'>")
                            html.append("        </div>")

                        elif len(weights.shape) >= 3:
                            # 2D+ attention
                            # We'll create a heatmap for the first sample
                            plt.figure(figsize=(10, 8))

                            # Get average attention across all samples
                            if len(weights.shape) == 3:
                                # (instance, seq, seq)
                                mean_attention = np.mean(weights, axis=0)
                            elif len(weights.shape) == 4:
                                # (instance, head, seq, seq)
                                # Average across instances and heads
                                mean_attention = np.mean(np.mean(weights, axis=0), axis=0)

                            # Create heatmap
                            sns.heatmap(mean_attention, cmap='viridis',
                                      xticklabels=self.feature_names if len(self.feature_names) < 20 else False,
                                      yticklabels=self.feature_names if len(self.feature_names) < 20 else False)
                            plt.title('Average Attention Weights')
                            plt.tight_layout()

                            # Convert plot to base64
                            buffer = BytesIO()
                            plt.savefig(buffer, format='png')
                            buffer.seek(0)
                            plot_data = b64encode(buffer.getvalue()).decode('utf-8')
                            plt.close()

                            # Add the plot to HTML
                            html.append("        <div class='plot-container'>")
                            html.append(f"            <img src='data:image/png;base64,{plot_data}' alt='Attention Heatmap'>")
                            html.append("        </div>")

                except Exception as e:
                    html.append(f"        <p class='error'>Error generating attention visualization: {str(e)}</p>")

                # Add sample explanations
                if 'samples' in attention_data:
                    html.append("        <h3>Sample Attention Weights</h3>")

                    # Create sample cards
                    for sample_idx, sample_data in attention_data['samples'].items():
                        html.append(f"        <div class='sample-card'>")
                        html.append(f"            <h4>Sample {sample_idx}</h4>")

                        # Show feature attention
                        html.append("            <table>")
                        html.append("                <tr><th>Feature</th><th>Attention Weight</th></tr>")

                        # Sort weights for better display
                        weights = [(feature, weight) for feature, weight in sample_data['feature_attention'].items()]
                        weights.sort(key=lambda x: x[1], reverse=True)

                        for feature, weight in weights[:10]:  # Top 10 features
                            html.append(f"                <tr><td>{feature}</td><td>{weight:.4f}</td></tr>")

                        html.append("            </table>")

                        # Add shape information
                        if 'shape' in sample_data:
                            html.append(f"            <p><strong>Attention Shape:</strong> {sample_data['shape']}</p>")

                        # Try to generate a visualization for this sample
                        try:
                            if hasattr(self, 'attention_weights') and self.attention_weights is not None:
                                sample_pos = list(self.attention_weights['indices']).index(int(sample_idx))
                                instance_weights = self.attention_weights['weights'][sample_pos]

                                # Handle different shapes
                                if len(instance_weights.shape) == 1:
                                    # 1D feature attention
                                    plt.figure(figsize=(10, 6))

                                    # Sort for visualization
                                    sorted_idx = np.argsort(instance_weights)[-10:]  # Top 10
                                    plt.barh([self.feature_names[i] for i in sorted_idx],
                                           [instance_weights[i] for i in sorted_idx], color='skyblue')
                                    plt.xlabel('Attention Weight')
                                    plt.title(f'Feature Attention - Sample {sample_idx}')
                                    plt.grid(alpha=0.3)
                                    plt.tight_layout()

                                    # Convert plot to base64
                                    buffer = BytesIO()
                                    plt.savefig(buffer, format='png')
                                    buffer.seek(0)
                                    plot_data = b64encode(buffer.getvalue()).decode('utf-8')
                                    plt.close()

                                    # Add the plot to HTML
                                    html.append("            <div class='plot-container'>")
                                    html.append(f"                <img src='data:image/png;base64,{plot_data}' alt='Sample Attention'>")
                                    html.append("            </div>")

                                elif len(instance_weights.shape) == 2:
                                    # 2D attention
                                    plt.figure(figsize=(8, 8))
                                    sns.heatmap(instance_weights, cmap='viridis',
                                              xticklabels=self.feature_names if len(self.feature_names) < 20 else False,
                                              yticklabels=self.feature_names if len(self.feature_names) < 20 else False)
                                    plt.title(f'Attention Heatmap - Sample {sample_idx}')
                                    plt.tight_layout()

                                    # Convert plot to base64
                                    buffer = BytesIO()
                                    plt.savefig(buffer, format='png')
                                    buffer.seek(0)
                                    plot_data = b64encode(buffer.getvalue()).decode('utf-8')
                                    plt.close()

                                    # Add the plot to HTML
                                    html.append("            <div class='plot-container'>")
                                    html.append(f"                <img src='data:image/png;base64,{plot_data}' alt='Sample Attention Heatmap'>")
                                    html.append("            </div>")
                        except Exception as e:
                            html.append(f"            <p class='error'>Error generating attention visualization: {str(e)}</p>")

                        html.append("        </div>")
        else:
            html.append("        <p>Attention weights not computed or not applicable for this model type.</p>")

        html.append("    </div>")

        # Close HTML
        html.append("</body>")
        html.append("</html>")

        return "\n".join(html)

    def save_explanations(self, filepath, format='pickle'):
        """
        Save explanation results to a file.

        Parameters:
        -----------
        filepath : str
            Path to save the results
        format : str
            Format to save the results ('pickle' or 'json')
        """
        if format.lower() == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self.explanation_results, f)
        elif format.lower() == 'json':
            import json

            # We need to convert numpy arrays to lists for JSON serialization
            def convert_to_json_serializable(obj):
                if isinstance(obj, (np.ndarray, pd.Series)):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                else:
                    return obj

            # Convert results to JSON-serializable format
            json_results = convert_to_json_serializable(self.explanation_results)

            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'.")

        self.logger.info(f"Explanations saved to {filepath} in {format} format")

    @classmethod
    def load_explanations(cls, filepath, format='pickle'):
        """
        Load explanation results from a file.

        Parameters:
        -----------
        filepath : str
            Path to the saved results
        format : str
            Format of the saved results ('pickle' or 'json')

        Returns:
        --------
        dict
            Dictionary of explanation results
        """
        if format.lower() == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format.lower() == 'json':
            import json
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'.")


