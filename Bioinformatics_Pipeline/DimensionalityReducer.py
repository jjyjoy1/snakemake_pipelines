import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import logging
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings

# Optional imports - will be handled gracefully if not available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not installed. Install with 'pip install umap-learn' to use UMAP method.")


class BioinformaticsDimensionalityReducer:
    """
    Step 3: Dimensionality Reduction
    
    A class for reducing the dimensionality of bioinformatics data for visualization
    and interpretation while preserving key biological signals.
    
    This class implements various dimensionality reduction techniques:
    - Linear methods: PCA, LDA
    - Nonlinear methods: t-SNE, UMAP, Autoencoders
    
    Each method has different strengths for revealing patterns in biological data.
    """
    
    def __init__(self, method='pca', n_components=2, random_state=42, logger=None):
        """
        Initialize the dimensionality reducer.
        
        Parameters:
        -----------
        method : str
            Dimensionality reduction method to use:
            - 'pca': Principal Component Analysis (linear, preserves global structure)
            - 'lda': Linear Discriminant Analysis (supervised, maximizes class separation)
            - 'tsne': t-SNE (nonlinear, preserves local structure)
            - 'umap': UMAP (nonlinear, balances local and global structure)
            - 'autoencoder': Autoencoder (nonlinear, deep learning based)
        n_components : int
            Number of dimensions to reduce to (default: 2 for visualization)
        random_state : int
            Random seed for reproducibility
        logger : logging.Logger
            Logger for tracking the dimensionality reduction process
        """
        self.method = method.lower()
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = None
        self.model = None  # For autoencoder
        self.encoder = None  # For autoencoder
        self.is_fitted = False
        self.feature_names_ = None
        self.components_ = None
        
        # Set up logger
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
            
        # Validate method
        valid_methods = ['pca', 'lda', 'tsne', 'umap', 'autoencoder']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Choose from: {', '.join(valid_methods)}")
        
        # Check if UMAP is available
        if self.method == 'umap' and not UMAP_AVAILABLE:
            raise ImportError("UMAP method requires the 'umap-learn' package. Install with 'pip install umap-learn'")
    
    def _setup_logger(self):
        """Setup a basic logger if none is provided."""
        logger = logging.getLogger("BioinformaticsDimensionalityReducer")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def fit(self, X, y=None, **kwargs):
        """
        Fit the dimensionality reduction model to the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray, optional
            The target variable (required for LDA)
        **kwargs : 
            Additional parameters specific to the dimensionality reduction method
            
        Returns:
        --------
        self : BioinformaticsDimensionalityReducer
            The fitted dimensionality reducer
        """
        if isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self.feature_names_ = X.columns if hasattr(X, 'columns') else None
        
        self.logger.info(f"Starting dimensionality reduction with method: {self.method}")
        
        if self.method == 'pca':
            self._fit_pca(X, **kwargs)
        elif self.method == 'lda':
            if y is None:
                raise ValueError("Target variable (y) must be provided for LDA")
            self._fit_lda(X, y, **kwargs)
        elif self.method == 'tsne':
            self._fit_tsne(X, **kwargs)
        elif self.method == 'umap':
            self._fit_umap(X, y, **kwargs)
        elif self.method == 'autoencoder':
            self._fit_autoencoder(X, **kwargs)
        
        self.is_fitted = True
        self.logger.info(f"Dimensionality reduction completed. Reduced to {self.n_components} dimensions.")
        return self
    
    def transform(self, X):
        """
        Transform the data to reduced dimensions.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
            
        Returns:
        --------
        numpy.ndarray or pandas.DataFrame
            The reduced data
        """
        if not self.is_fitted:
            raise ValueError("Dimensionality reducer not fitted. Call fit() first.")
            
        self.logger.info(f"Transforming data using fitted {self.method} model")
        
        if isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if self.method in ['pca', 'lda']:
            # For sklearn methods
            X_reduced = self.reducer.transform(X)
        elif self.method == 'tsne':
            # t-SNE doesn't support transform on new data, so we rerun fit_transform
            # This is a limitation of t-SNE
            self.logger.warning("t-SNE doesn't support transform on new data. Running new fit_transform.")
            X_reduced = self._fit_transform_tsne(X)
        elif self.method == 'umap':
            # UMAP supports transform
            X_reduced = self.reducer.transform(X)
        elif self.method == 'autoencoder':
            # For autoencoder, use the encoder part
            X_reduced = self.encoder.predict(X)
        
        # Convert to DataFrame for better interpretability
        if isinstance(X, pd.DataFrame):
            component_names = [f"{self.method.upper()}_{i+1}" for i in range(self.n_components)]
            X_reduced = pd.DataFrame(X_reduced, index=X.index, columns=component_names)
            
        return X_reduced
    
    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit the dimensionality reducer to the data and transform the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray, optional
            The target variable (required for LDA)
        **kwargs : 
            Additional parameters specific to the dimensionality reduction method
            
        Returns:
        --------
        numpy.ndarray or pandas.DataFrame
            The reduced data
        """
        return self.fit(X, y, **kwargs).transform(X)
    
    def _fit_pca(self, X, **kwargs):
        """
        Fit PCA model.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        **kwargs :
            Additional parameters for PCA
        """
        # Extract PCA specific parameters
        svd_solver = kwargs.get('svd_solver', 'auto')
        whiten = kwargs.get('whiten', False)
        
        # Initialize and fit PCA
        self.reducer = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
            svd_solver=svd_solver,
            whiten=whiten
        )
        self.reducer.fit(X)
        
        # Store components for later analysis
        self.components_ = self.reducer.components_
        
        # Log explained variance if applicable
        if hasattr(self.reducer, 'explained_variance_ratio_'):
            total_var = sum(self.reducer.explained_variance_ratio_)
            self.logger.info(f"PCA total explained variance: {total_var:.2%}")
    
    def _fit_lda(self, X, y, **kwargs):
        """
        Fit LDA model.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target variable
        **kwargs :
            Additional parameters for LDA
        """
        # Extract LDA specific parameters
        solver = kwargs.get('solver', 'svd')
        shrinkage = kwargs.get('shrinkage', None)
        
        # Determine max possible components for LDA (min of n_features and n_classes - 1)
        n_classes = len(np.unique(y))
        max_components = min(X.shape[1], n_classes - 1)
        
        if self.n_components > max_components:
            self.logger.warning(
                f"Requested {self.n_components} components, but LDA can only compute "
                f"{max_components} components. Using {max_components} components instead."
            )
            n_components = max_components
        else:
            n_components = self.n_components
        
        # Initialize and fit LDA
        self.reducer = LinearDiscriminantAnalysis(
            n_components=n_components,
            solver=solver,
            shrinkage=shrinkage
        )
        self.reducer.fit(X, y)
        
        # Store components
        if hasattr(self.reducer, 'coef_'):
            self.components_ = self.reducer.coef_
        elif hasattr(self.reducer, 'scalings_'):
            self.components_ = self.reducer.scalings_
    
    def _fit_tsne(self, X, **kwargs):
        """
        Fit t-SNE model.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        **kwargs :
            Additional parameters for t-SNE
        """
        # Extract t-SNE specific parameters
        perplexity = kwargs.get('perplexity', min(30, max(X.shape[0] // 5, 5)))
        learning_rate = kwargs.get('learning_rate', 200.0)
        n_iter = kwargs.get('n_iter', 1000)
        metric = kwargs.get('metric', 'euclidean')
        
        # Initialize and fit t-SNE
        self.reducer = TSNE(
            n_components=self.n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            metric=metric,
            random_state=self.random_state
        )
        
        # t-SNE doesn't have a separate fit method, so we fit and store the result
        self.embedding_ = self.reducer.fit_transform(X)
    
    def _fit_transform_tsne(self, X, **kwargs):
        """
        Fit and transform using t-SNE.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        **kwargs :
            Additional parameters for t-SNE
            
        Returns:
        --------
        numpy.ndarray
            The reduced data
        """
        # Create a new t-SNE instance with same parameters
        perplexity = kwargs.get('perplexity', min(30, max(X.shape[0] // 5, 5)))
        learning_rate = kwargs.get('learning_rate', 200.0)
        n_iter = kwargs.get('n_iter', 1000)
        metric = kwargs.get('metric', 'euclidean')
        
        tsne = TSNE(
            n_components=self.n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            metric=metric,
            random_state=self.random_state
        )
        
        return tsne.fit_transform(X)
    
    def _fit_umap(self, X, y=None, **kwargs):
        """
        Fit UMAP model.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        y : pandas.Series or numpy.ndarray, optional
            The target variable for supervised UMAP
        **kwargs :
            Additional parameters for UMAP
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP method requires the 'umap-learn' package")
            
        # Extract UMAP specific parameters
        n_neighbors = kwargs.get('n_neighbors', 15)
        min_dist = kwargs.get('min_dist', 0.1)
        metric = kwargs.get('metric', 'euclidean')
        
        # Initialize and fit UMAP
        self.reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=self.random_state
        )
        
        if y is not None:
            self.reducer.fit(X, y)  # Supervised UMAP
        else:
            self.reducer.fit(X)  # Unsupervised UMAP
    
    def _fit_autoencoder(self, X, **kwargs):
        """
        Build and train an autoencoder for dimensionality reduction.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix
        **kwargs :
            Additional parameters for the autoencoder
            - encoding_dim: Dimensionality of the encoding space (default: n_components)
            - hidden_layers: List of units in each hidden layer (default: [])
            - activation: Activation function (default: 'relu')
            - epochs: Number of training epochs (default: 100)
            - batch_size: Batch size for training (default: 32)
            - learning_rate: Learning rate for optimizer (default: 0.001)
            - dropout_rate: Dropout rate for regularization (default: 0.2)
            - validation_split: Fraction of data for validation (default: 0.2)
        """
        # Extract autoencoder specific parameters
        encoding_dim = kwargs.get('encoding_dim', self.n_components)
        hidden_layers = kwargs.get('hidden_layers', [])
        activation = kwargs.get('activation', 'relu')
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 32)
        learning_rate = kwargs.get('learning_rate', 0.001)
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        validation_split = kwargs.get('validation_split', 0.2)
        use_batch_norm = kwargs.get('use_batch_norm', True)
        
        # Convert to numpy array for TensorFlow
        X_values = X.values if hasattr(X, 'values') else X
        
        # Define the input layer
        input_dim = X_values.shape[1]
        inputs = Input(shape=(input_dim,))
        x = inputs
        
        # Add encoder layers
        if len(hidden_layers) > 0:
            # Add hidden layers with decreasing number of units
            for units in hidden_layers:
                x = Dense(units, activation=activation)(x)
                
                if use_batch_norm:
                    x = BatchNormalization()(x)
                    
                if dropout_rate > 0:
                    x = Dropout(dropout_rate)(x)
        
        # Add bottleneck layer (encoding)
        encoded = Dense(encoding_dim, activation=activation, name='bottleneck')(x)
        
        # Add decoder layers (symmetric to encoder)
        x = encoded
        
        if len(hidden_layers) > 0:
            # Add hidden layers with increasing number of units
            for units in reversed(hidden_layers):
                x = Dense(units, activation=activation)(x)
                
                if use_batch_norm:
                    x = BatchNormalization()(x)
                    
                if dropout_rate > 0:
                    x = Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = Dense(input_dim, activation='linear')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Create encoder model
        self.encoder = Model(inputs=inputs, outputs=encoded)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        # Train the model
        self.logger.info("Training autoencoder...")
        self.model.fit(
            X_values, X_values,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0  # Set to 1 or 2 for more verbose output
        )
        
        self.logger.info("Autoencoder training complete")
        
        # Store the bottleneck layer weights as components
        # This could be used similar to PCA loadings for interpretation
        bottleneck_layer = self.model.get_layer('bottleneck')
        self.components_ = bottleneck_layer.get_weights()[0].T
    
    def plot_reduced_data(self, X=None, y=None, X_reduced=None, **kwargs):
        """
        Plot the reduced dimensional data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray, optional
            Original data to reduce (not needed if X_reduced is provided)
        y : pandas.Series or numpy.ndarray, optional
            Target variable for coloring the points
        X_reduced : pandas.DataFrame or numpy.ndarray, optional
            Pre-reduced data (if already transformed)
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size (default: (10, 8))
            - alpha: Transparency of points (default: 0.7)
            - s: Point size (default: 50)
            - title: Plot title (default: method name)
            - cmap: Colormap for categorical data (default: 'viridis')
            - palette: Color palette for categorical data (default: None)
            - add_legend: Whether to add a legend (default: True if y is provided)
            - legend_title: Title for the legend (default: 'Class')
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if X_reduced is None:
            if X is None:
                raise ValueError("Either X or X_reduced must be provided")
            if not self.is_fitted:
                raise ValueError("Model not fitted. Call fit() first.")
            X_reduced = self.transform(X)
            
        # Extract plotting parameters
        figsize = kwargs.get('figsize', (10, 8))
        alpha = kwargs.get('alpha', 0.7)
        s = kwargs.get('s', 50)
        title = kwargs.get('title', f"{self.method.upper()} Visualization")
        cmap = kwargs.get('cmap', 'viridis')
        palette = kwargs.get('palette', None)
        add_legend = kwargs.get('add_legend', y is not None)
        legend_title = kwargs.get('legend_title', 'Class')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract first two components for plotting
        if X_reduced.shape[1] < 2:
            raise ValueError("Need at least 2 components for visualization")
            
        x1 = X_reduced[:, 0] if isinstance(X_reduced, np.ndarray) else X_reduced.iloc[:, 0]
        x2 = X_reduced[:, 1] if isinstance(X_reduced, np.ndarray) else X_reduced.iloc[:, 1]
        
        # Generate component names
        if isinstance(X_reduced, pd.DataFrame):
            xlabel = X_reduced.columns[0]
            ylabel = X_reduced.columns[1]
        else:
            xlabel = f"{self.method.upper()} 1"
            ylabel = f"{self.method.upper()} 2"
        
        # Plot with different styles based on presence of labels
        if y is not None:
            # Convert to categorical if numeric
            if pd.api.types.is_numeric_dtype(y):
                y_cat = pd.Categorical(y)
                color_values = y_cat.codes
                legend_labels = y_cat.categories
            else:
                y_unique = pd.unique(y)
                color_dict = {val: i for i, val in enumerate(y_unique)}
                color_values = np.array([color_dict[val] for val in y])
                legend_labels = y_unique
            
            # Create scatter plot with categorical colors
            scatter = ax.scatter(
                x1, x2, 
                c=color_values, 
                alpha=alpha, 
                s=s, 
                cmap=cmap
            )
            
            # Add legend if requested
            if add_legend:
                if palette is not None:
                    # If palette is provided, create custom legend
                    colors = plt.cm.get_cmap(cmap, len(legend_labels))
                    for i, label in enumerate(legend_labels):
                        ax.scatter([], [], c=[colors(i)], label=label, s=s)
                    ax.legend(title=legend_title)
                else:
                    # Create legend based on scatter plot
                    legend = ax.legend(
                        *scatter.legend_elements(),
                        title=legend_title,
                        loc="best"
                    )
                    ax.add_artist(legend)
        else:
            # Simple scatter plot without categories
            ax.scatter(x1, x2, alpha=alpha, s=s)
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Add grid
        ax.grid(alpha=0.3)
        
        # Add colorbar if continuous color scale
        if y is not None and pd.api.types.is_numeric_dtype(y) and not isinstance(y, pd.Categorical):
            plt.colorbar(scatter, ax=ax, label=legend_title)
        
        plt.tight_layout()
        return fig
    
    def plot_explained_variance(self, X=None, **kwargs):
        """
        Plot the explained variance ratio for PCA.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray, optional
            Data to fit PCA on (if not already done)
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size (default: (10, 6))
            - cumulative: Whether to plot cumulative variance (default: True)
            - n_components: Number of components to plot (default: 10)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.method != 'pca':
            raise ValueError("Explained variance is only available for PCA")
            
        if not self.is_fitted and X is None:
            raise ValueError("Either the model must be fitted or X must be provided")
            
        # Fit PCA if not already done
        if not self.is_fitted and X is not None:
            self.fit(X)
            
        # Extract plotting parameters
        figsize = kwargs.get('figsize', (10, 6))
        cumulative = kwargs.get('cumulative', True)
        n_components = min(kwargs.get('n_components', 10), len(self.reducer.explained_variance_ratio_))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get variance ratios
        var_ratio = self.reducer.explained_variance_ratio_[:n_components]
        components = range(1, n_components + 1)
        
        # Plot individual variance
        ax.bar(components, var_ratio, alpha=0.5, label='Individual')
        ax.set_xlabel('Principal Components')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_xticks(components)
        
        # Add cumulative variance if requested
        if cumulative:
            cum_var_ratio = np.cumsum(var_ratio)
            ax2 = ax.twinx()
            ax2.plot(components, cum_var_ratio, 'r-', marker='o', label='Cumulative')
            ax2.set_ylabel('Cumulative Explained Variance Ratio')
            ax2.set_ylim([0, 1.05])
            
            # Add threshold lines
            for threshold in [0.5, 0.8, 0.9, 0.95]:
                if cum_var_ratio[-1] >= threshold:
                    # Find the component that exceeds the threshold
                    comp_idx = np.where(cum_var_ratio >= threshold)[0][0]
                    comp_num = components[comp_idx]
                    ax2.axhline(y=threshold, linestyle='--', alpha=0.3, color='gray')
                    ax2.text(
                        n_components * 0.7, 
                        threshold, 
                        f'{threshold:.0%} at PC{comp_num}', 
                        verticalalignment='bottom'
                    )
        
        # Add title
        plt.title('Explained Variance by Principal Components')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        if cumulative:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax.legend(loc='upper right')
            
        plt.tight_layout()
        return fig
    
    def plot_feature_loadings(self, n_features=20, component_idx=0, **kwargs):
        """
        Plot the feature loadings/coefficients for the selected component.
        
        Parameters:
        -----------
        n_features : int
            Number of top features to display
        component_idx : int
            Index of the component to analyze (0 for first component)
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size (default: (12, n_features/2))
            - palette: Color palette (default: 'viridis')
            - absolute: Whether to sort by absolute values (default: True)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.components_ is None:
            raise ValueError("Feature loadings not available. Method may not support it.")
            
        if component_idx >= self.components_.shape[0]:
            raise ValueError(f"Component index {component_idx} out of range (max: {self.components_.shape[0]-1})")
            
        # Extract loadings for the selected component
        loadings = self.components_[component_idx]
        
        # Create feature names if not available
        feature_names = (
            self.feature_names_ 
            if self.feature_names_ is not None 
            else [f"Feature_{i}" for i in range(len(loadings))]
        )
        
        # Create DataFrame with loadings
        loadings_df = pd.DataFrame({
            'Feature': feature_names,
            'Loading': loadings
        })
        
        # Sort by absolute value or raw value
        absolute = kwargs.get('absolute', True)
        if absolute:
            loadings_df['Abs_Loading'] = np.abs(loadings_df['Loading'])
            loadings_df = loadings_df.sort_values('Abs_Loading', ascending=False)
        else:
            loadings_df = loadings_df.sort_values('Loading', ascending=False)
            
        # Take top N features
        loadings_df = loadings_df.head(n_features)
        
        # Extract plotting parameters
        figsize = kwargs.get('figsize', (12, max(6, n_features/2)))
        palette = kwargs.get('palette', 'viridis')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        bars = ax.barh(
            loadings_df['Feature'], 
            loadings_df['Loading'],
            color=plt.cm.get_cmap(palette)(np.abs(loadings_df['Loading']) / np.max(np.abs(loadings_df['Loading'])))
        )
        
        # Add labels and title
        component_name = (
            f"{self.method.upper()}_{component_idx+1}"
            if self.method != 'lda'
            else f"LD{component_idx+1}"
        )
        ax.set_title(f"Feature Loadings for {component_name}")
        ax.set_xlabel('Loading Coefficient')
        
        # Add zero line
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add grid
        ax.grid(alpha=0.3)
        
        # Invert y-axis to show highest loadings at the top
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def get_explained_variance_ratio(self):
        """
        Get the explained variance ratio for PCA.
        
        Returns:
        --------
        numpy.ndarray
            The explained variance ratio for each component
        """
        if self.method != 'pca':
            raise ValueError("Explained variance ratio is only available for PCA")
            
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return self.reducer.explained_variance_ratio_
    
    def get_feature_loadings(self):
        """
        Get the feature loadings/components.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature loadings for each component
        """
        if self.components_ is None:
            raise ValueError("Feature loadings not available. Method may not support it.")
            
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Create feature names if not available
        feature_names = (
            self.feature_names_ 
            if self.feature_names_ is not None 
            else [f"Feature_{i}" for i in range(self.components_.shape[1])]
        )
        
        # Create component names
        if self.method == 'lda':
            component_names = [f"LD{i+1}" for i in range(self.components_.shape[0])]
        else:
            component_names = [f"{self.method.upper()}_{i+1}" for i in range(self.components_.shape[0])]
            
        # Create DataFrame with loadings
        loadings_df = pd.DataFrame(
            self.components_,
            index=component_names,
            columns=feature_names
        )
            
        return loadings_df
        
    def save_model(self, filepath):
        """
        Save the dimensionality reduction model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        import pickle
        
        if self.method == 'autoencoder':
            # For autoencoder, save the keras model separately
            model_path = f"{filepath}_autoencoder.h5"
            self.model.save(model_path)
            self.logger.info(f"Autoencoder model saved to {model_path}")
            
            # Create a copy without the model for pickling
            import copy
            reducer_copy = copy.copy(self)
            reducer_copy.model = None
            reducer_copy.encoder = None
            
            # Save the reducer object
            with open(filepath, 'wb') as f:
                pickle.dump(reducer_copy, f)
        else:
            # For other methods, just pickle the object
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
                
        self.logger.info(f"Model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath):
        """
        Load a dimensionality reduction model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        BioinformaticsDimensionalityReducer
            The loaded model
        """
        import pickle
        
        # Load the reducer object
        with open(filepath, 'rb') as f:
            reducer = pickle.load(f)
            
        # If it's an autoencoder, load the keras model
        if reducer.method == 'autoencoder':
            model_path = f"{filepath}_autoencoder.h5"
            reducer.model = load_model(model_path)
            
            # Recreate the encoder model
            inputs = reducer.model.input
            bottleneck_layer = reducer.model.get_layer('bottleneck')
            reducer.encoder = Model(inputs=inputs, outputs=bottleneck_layer.output)
            
        return reducer


