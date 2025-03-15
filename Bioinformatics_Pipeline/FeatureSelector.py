import numpy as np
import pandas as pd
import logging
from scipy import stats
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    VarianceThreshold, SelectFromModel, RFE
)
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K


class BioinformaticsFeatureSelector:
    """
    Step 2: Feature Selection
    
    A class for selecting the most informative features in bioinformatics data
    to reduce noise and computational complexity.
    
    This class implements various feature selection approaches:
    - Statistical tests (univariate methods)
    - Variance-based filtering
    - Regularization methods (L1/L2)
    - Tree-based feature importance
    - Wrapper methods (RFE)
    - Deep learning methods (Autoencoders)
    """
    
    def __init__(self, method='variance', task_type='classification', logger=None):
        """
        Initialize the feature selector.
        
        Parameters:
        -----------
        method : str
            Feature selection method to use:
            - 'statistical': univariate statistical tests
            - 'variance': variance threshold
            - 'regularization': LASSO/ElasticNet
            - 'tree': Random Forest or XGBoost importance
            - 'rfe': Recursive Feature Elimination
            - 'autoencoder': Autoencoder-based selection
            - 'vae': Variational Autoencoder-based selection
        task_type : str
            Type of machine learning task ('classification' or 'regression')
        logger : logging.Logger
            Logger for tracking the feature selection process
        """
        self.method = method
        self.task_type = task_type
        self.selected_features_ = None
        self.feature_importances_ = None
        self.selector_ = None
        self.model_ = None
        
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
    
    def _setup_logger(self):
        """Setup a basic logger if none is provided."""
        logger = logging.getLogger("BioinformaticsFeatureSelector")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def fit(self, X, y=None, **kwargs):
        """
        Fit the feature selector to the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray, optional
            The target variable (required for supervised methods)
        **kwargs : 
            Additional parameters specific to the feature selection method
            
        Returns:
        --------
        self : BioinformaticsFeatureSelector
            The fitted feature selector
        """
        if isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self.feature_names_ = X.columns if hasattr(X, 'columns') else np.arange(X.shape[1])
        
        self.logger.info(f"Starting feature selection with method: {self.method}")
        
        if self.method == 'statistical':
            self._fit_statistical(X, y, **kwargs)
        elif self.method == 'variance':
            self._fit_variance(X, **kwargs)
        elif self.method == 'regularization':
            self._fit_regularization(X, y, **kwargs)
        elif self.method == 'tree':
            self._fit_tree(X, y, **kwargs)
        elif self.method == 'rfe':
            self._fit_rfe(X, y, **kwargs)
        elif self.method == 'autoencoder':
            self._fit_autoencoder(X, **kwargs)
        elif self.method == 'vae':
            self._fit_vae(X, **kwargs)
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")
            
        self.logger.info(f"Feature selection completed. Selected {len(self.selected_features_)} features.")
        return self
    
    def transform(self, X):
        """
        Transform the data to use only the selected features.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
            
        Returns:
        --------
        pandas.DataFrame or numpy.ndarray
            The transformed data with only selected features
        """
        if isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
            
        if self.method in ['statistical', 'variance', 'regularization', 'tree', 'rfe']:
            # Methods using scikit-learn selectors
            X_transformed = self.selector_.transform(X)
            if isinstance(X, pd.DataFrame):
                return pd.DataFrame(X_transformed, index=X.index, columns=self.selected_features_)
            return X_transformed
        
        elif self.method in ['autoencoder', 'vae']:
            # For autoencoder/VAE, we just return the selected features
            return X[self.selected_features_]
        
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")
    
    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit the feature selector to the data and transform the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray, optional
            The target variable (required for supervised methods)
        **kwargs : 
            Additional parameters specific to the feature selection method
            
        Returns:
        --------
        pandas.DataFrame or numpy.ndarray
            The transformed data with only selected features
        """
        return self.fit(X, y, **kwargs).transform(X)
    
    def _fit_statistical(self, X, y, **kwargs):
        """
        Fit a statistical test-based feature selector.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        **kwargs :
            k : int
                Number of top features to select
            score_func : callable
                Function for scoring features (default depends on task_type)
            alpha : float
                p-value threshold for feature selection (used if k is None)
        """
        if y is None:
            raise ValueError("Target variable (y) must be provided for statistical feature selection")
            
        k = kwargs.get('k', min(X.shape[1] // 10, 100))  # Default: select top 10% features or top 100
        alpha = kwargs.get('alpha', 0.05)
        
        # Select scoring function based on task type
        if 'score_func' in kwargs:
            score_func = kwargs['score_func']
        else:
            if self.task_type == 'classification':
                test_type = kwargs.get('test_type', 'f_test')
                if test_type == 'f_test':
                    score_func = f_classif
                elif test_type == 'mutual_info':
                    score_func = mutual_info_classif
                elif test_type == 'chi2':
                    from sklearn.feature_selection import chi2
                    score_func = chi2
                elif test_type == 'kruskal':
                    # Custom Kruskal-Wallis scoring function
                    def kruskal_score(X, y):
                        scores = []
                        pvals = []
                        for i in range(X.shape[1]):
                            feature = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
                            unique_classes = np.unique(y)
                            groups = [feature[y == c] for c in unique_classes]
                            h, p = stats.kruskal(*groups)
                            scores.append(h)
                            pvals.append(p)
                        return np.array(scores), np.array(pvals)
                    score_func = kruskal_score
            else:  # regression
                test_type = kwargs.get('test_type', 'f_test')
                if test_type == 'f_test':
                    score_func = f_regression
                elif test_type == 'mutual_info':
                    score_func = mutual_info_regression
                
        # Use SelectKBest for feature selection
        self.selector_ = SelectKBest(score_func=score_func, k=k)
        self.selector_.fit(X, y)
        
        # Get feature importances and selected features
        self.feature_importances_ = self.selector_.scores_
        
        if hasattr(self.selector_, 'get_support'):
            selected_indices = self.selector_.get_support(indices=True)
            self.selected_features_ = [self.feature_names_[i] for i in selected_indices]
        else:
            # For custom statistical tests, use p-values
            p_values = self.selector_.pvalues_ if hasattr(self.selector_, 'pvalues_') else None
            if p_values is not None:
                selected_indices = np.where(p_values < alpha)[0]
                self.selected_features_ = [self.feature_names_[i] for i in selected_indices]
            else:
                # Fallback to top k features by score
                top_indices = np.argsort(self.feature_importances_)[::-1][:k]
                self.selected_features_ = [self.feature_names_[i] for i in top_indices]
    
    def _fit_variance(self, X, **kwargs):
        """
        Fit a variance threshold-based feature selector.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        **kwargs :
            threshold : float
                Variance threshold for feature selection
        """
        threshold = kwargs.get('threshold', 0.05)
        
        # Use VarianceThreshold for feature selection
        self.selector_ = VarianceThreshold(threshold=threshold)
        self.selector_.fit(X)
        
        # Get selected features
        selected_indices = self.selector_.get_support(indices=True)
        self.selected_features_ = [self.feature_names_[i] for i in selected_indices]
        
        # Calculate feature importances as variances
        self.feature_importances_ = np.var(X.values, axis=0)
    
    def _fit_regularization(self, X, y, **kwargs):
        """
        Fit a regularization-based feature selector.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        **kwargs :
            model_type : str
                Type of regularization model ('lasso' or 'elasticnet')
            alpha : float
                Regularization strength
            l1_ratio : float
                Ratio of L1 penalty for ElasticNet (0 <= l1_ratio <= 1)
            max_features : int
                Maximum number of features to select
        """
        if y is None:
            raise ValueError("Target variable (y) must be provided for regularization-based feature selection")
            
        model_type = kwargs.get('model_type', 'lasso')
        alpha = kwargs.get('alpha', 0.01)
        max_features = kwargs.get('max_features', None)
        
        # Create regularization model
        if model_type == 'lasso':
            model = Lasso(alpha=alpha)
        elif model_type == 'elasticnet':
            l1_ratio = kwargs.get('l1_ratio', 0.5)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        else:
            raise ValueError(f"Unsupported regularization model: {model_type}")
            
        # Use SelectFromModel for feature selection
        self.selector_ = SelectFromModel(model, max_features=max_features)
        self.selector_.fit(X, y)
        
        # Get feature importances and selected features
        self.model_ = self.selector_.estimator_
        self.feature_importances_ = np.abs(self.model_.coef_)
        
        selected_indices = self.selector_.get_support(indices=True)
        self.selected_features_ = [self.feature_names_[i] for i in selected_indices]
    
    def _fit_tree(self, X, y, **kwargs):
        """
        Fit a tree-based feature selector.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        **kwargs :
            model_type : str
                Type of tree model ('rf' for Random Forest or 'xgb' for XGBoost)
            n_estimators : int
                Number of trees in the forest/ensemble
            max_features : int
                Maximum number of features to select
            threshold : str or float
                Threshold for feature selection ('mean', 'median' or float)
        """
        if y is None:
            raise ValueError("Target variable (y) must be provided for tree-based feature selection")
            
        model_type = kwargs.get('model_type', 'rf')
        n_estimators = kwargs.get('n_estimators', 100)
        max_features = kwargs.get('max_features', None)
        threshold = kwargs.get('threshold', 'mean')
        
        # Create tree-based model
        if model_type == 'rf':
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        elif model_type == 'xgb':
            if self.task_type == 'classification':
                model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=42)
            else:
                model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42)
        else:
            raise ValueError(f"Unsupported tree model: {model_type}")
            
        # Use SelectFromModel for feature selection
        self.selector_ = SelectFromModel(model, max_features=max_features, threshold=threshold)
        self.selector_.fit(X, y)
        
        # Get feature importances and selected features
        self.model_ = self.selector_.estimator_
        self.feature_importances_ = self.model_.feature_importances_
        
        selected_indices = self.selector_.get_support(indices=True)
        self.selected_features_ = [self.feature_names_[i] for i in selected_indices]
    
    def _fit_rfe(self, X, y, **kwargs):
        """
        Fit a Recursive Feature Elimination feature selector.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        **kwargs :
            estimator : estimator object
                The base estimator for RFE
            n_features_to_select : int
                Number of features to select
            step : int or float
                Number or percentage of features to remove at each iteration
        """
        if y is None:
            raise ValueError("Target variable (y) must be provided for RFE feature selection")
            
        n_features_to_select = kwargs.get('n_features_to_select', min(X.shape[1] // 10, 100))
        step = kwargs.get('step', 1)
        
        # Create base estimator if not provided
        if 'estimator' in kwargs:
            estimator = kwargs['estimator']
        else:
            if self.task_type == 'classification':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                
        # Use RFE for feature selection
        self.selector_ = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
        self.selector_.fit(X, y)
        
        # Get feature importances and selected features
        self.feature_importances_ = self.selector_.ranking_
        
        selected_indices = self.selector_.get_support(indices=True)
        self.selected_features_ = [self.feature_names_[i] for i in selected_indices]
    
    def _fit_autoencoder(self, X, **kwargs):
        """
        Fit an autoencoder-based feature selector.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        **kwargs :
            encoding_dim : int
                Dimensionality of the encoded space
            epochs : int
                Number of training epochs
            batch_size : int
                Batch size for training
            learning_rate : float
                Learning rate for optimizer
            activation : str
                Activation function for hidden layers
            n_features_to_select : int
                Number of features to select based on reconstruction weights
        """
        # Convert to numpy array for TensorFlow
        X_values = X.values if hasattr(X, 'values') else X
        
        # Autoencoder parameters
        encoding_dim = kwargs.get('encoding_dim', min(X.shape[1] // 2, 50))
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 32)
        learning_rate = kwargs.get('learning_rate', 0.001)
        activation = kwargs.get('activation', 'relu')
        n_features_to_select = kwargs.get('n_features_to_select', min(X.shape[1] // 10, 100))
        
        # Build the autoencoder model
        input_dim = X_values.shape[1]
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(encoding_dim, activation=activation)(input_layer)
        
        # Decoder
        decoded = Dense(input_dim, activation='linear')(encoded)
        
        # Autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        encoder = Model(inputs=input_layer, outputs=encoded)
        
        # Compile the model
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='mean_squared_error')
        
        # Train the autoencoder
        autoencoder.fit(X_values, X_values, 
                     epochs=epochs, 
                     batch_size=batch_size, 
                     shuffle=True,
                     verbose=0)
        
        # Get the weights from the decoder layer
        decoder_weights = autoencoder.layers[-1].get_weights()[0]  # shape: (encoding_dim, input_dim)
        
        # Calculate feature importance as the sum of absolute weights
        feature_importance = np.sum(np.abs(decoder_weights), axis=0)
        
        # Select top features
        top_indices = np.argsort(feature_importance)[::-1][:n_features_to_select]
        
        # Save results
        self.model_ = autoencoder
        self.feature_importances_ = feature_importance
        self.selected_features_ = [self.feature_names_[i] for i in top_indices]
        
        # Create a simple selector function for transform method
        class AutoencoderSelector:
            def __init__(self, selected_indices):
                self.selected_indices = selected_indices
                
            def transform(self, X):
                if hasattr(X, 'iloc'):
                    return X.iloc[:, self.selected_indices]
                return X[:, self.selected_indices]
                
            def get_support(self, indices=False):
                mask = np.zeros(len(feature_importance), dtype=bool)
                mask[top_indices] = True
                return top_indices if indices else mask
                
        self.selector_ = AutoencoderSelector(top_indices)
    
    def _fit_vae(self, X, **kwargs):
        """
        Fit a Variational Autoencoder (VAE) based feature selector.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        **kwargs :
            encoding_dim : int
                Dimensionality of the encoded space
            epochs : int
                Number of training epochs
            batch_size : int
                Batch size for training
            learning_rate : float
                Learning rate for optimizer
            activation : str
                Activation function for hidden layers
            n_features_to_select : int
                Number of features to select based on reconstruction weights
        """
        # Convert to numpy array for TensorFlow
        X_values = X.values if hasattr(X, 'values') else X
        
        # VAE parameters
        encoding_dim = kwargs.get('encoding_dim', min(X.shape[1] // 2, 50))
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 32)
        learning_rate = kwargs.get('learning_rate', 0.001)
        activation = kwargs.get('activation', 'relu')
        n_features_to_select = kwargs.get('n_features_to_select', min(X.shape[1] // 10, 100))
        
        # Build the VAE model
        input_dim = X_values.shape[1]
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        h = Dense(encoding_dim * 2, activation=activation)(input_layer)
        z_mean = Dense(encoding_dim)(h)
        z_log_var = Dense(encoding_dim)(h)
        
        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
        # Sample from the latent space
        z = Lambda(sampling)([z_mean, z_log_var])
        
        # Decoder
        decoder_h = Dense(encoding_dim * 2, activation=activation)
        decoder_mean = Dense(input_dim, activation='linear')
        
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
        
        # VAE model
        vae = Model(inputs=input_layer, outputs=x_decoded_mean)
        
        # VAE loss
        def vae_loss(x, x_decoded_mean):
            reconstruction_loss = input_dim * tf.keras.losses.mean_squared_error(x, x_decoded_mean)
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(reconstruction_loss + kl_loss)
        
        # Compile the model
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=vae_loss)
        
        # Train the VAE
        vae.fit(X_values, X_values, 
             epochs=epochs, 
             batch_size=batch_size, 
             shuffle=True,
             verbose=0)
        
        # Get the weights from the decoder layer
        decoder_weights = vae.layers[-1].get_weights()[0]  # shape: (encoding_dim*2, input_dim)
        
        # Calculate feature importance as the sum of absolute weights
        feature_importance = np.sum(np.abs(decoder_weights), axis=0)
        
        # Select top features
        top_indices = np.argsort(feature_importance)[::-1][:n_features_to_select]
        
        # Save results
        self.model_ = vae
        self.feature_importances_ = feature_importance
        self.selected_features_ = [self.feature_names_[i] for i in top_indices]
        
        # Create a simple selector function for transform method
        class VAESelector:
            def __init__(self, selected_indices):
                self.selected_indices = selected_indices
                
            def transform(self, X):
                if hasattr(X, 'iloc'):
                    return X.iloc[:, self.selected_indices]
                return X[:, self.selected_indices]
                
            def get_support(self, indices=False):
                mask = np.zeros(len(feature_importance), dtype=bool)
                mask[top_indices] = True
                return top_indices if indices else mask
                
        self.selector_ = VAESelector(top_indices)
    
    def get_feature_importances(self):
        """
        Get the importance scores for all features.
        
        Returns:
        --------
        pandas.Series
            Feature importance scores indexed by feature names
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available. Run fit() first.")
            
        return pd.Series(self.feature_importances_, index=self.feature_names_)
    
    def get_selected_features(self):
        """
        Get the names of the selected features.
        
        Returns:
        --------
        list
            Names of the selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Selected features not available. Run fit() first.")
            
        return self.selected_features_


