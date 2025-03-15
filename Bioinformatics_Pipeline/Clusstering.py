import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial import distance
import logging
import warnings
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

# Try importing optional dependencies
try:
    import sklearnex
    INTEL_EXTENSION = True
    # Patch the estimators for better performance if available
    sklearnex.patch_sklearn()
except ImportError:
    INTEL_EXTENSION = False

try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except ImportError:
    KMEDOIDS_AVAILABLE = False
    warnings.warn("scikit-learn-extra not installed. KMedoids clustering unavailable. "
                  "Install with 'pip install scikit-learn-extra'.")

try:
    import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    warnings.warn("rpy2 not installed. WGCNA clustering unavailable. "
                  "Install with 'pip install rpy2'.")


class BioinformaticsClustering:
    """
    Step 4: Clustering
    
    A class for clustering bioinformatics data to discover biological groups.
    
    This class implements various clustering algorithms:
    - Partition-based: K-means, K-medoids (PAM)
    - Hierarchical: Agglomerative clustering (Ward's, complete, average, single linkage)
    - Density-based: DBSCAN
    - Model-based: Gaussian Mixture Models (GMM)
    - Network-based: WGCNA (weighted gene co-expression network analysis)
    - Spectral: Graph-based spectral clustering
    
    Each method has different strengths for identifying patterns in biological data.
    """
    
    def __init__(self, method='kmeans', random_state=42, logger=None):
        """
        Initialize the clustering object.
        
        Parameters:
        -----------
        method : str
            Clustering method to use:
            - 'kmeans': K-means clustering
            - 'kmedoids': K-medoids (PAM) clustering
            - 'hierarchical': Hierarchical clustering (agglomerative)
            - 'dbscan': DBSCAN (Density-Based Spatial Clustering)
            - 'gmm': Gaussian Mixture Models
            - 'wgcna': Weighted Gene Co-expression Network Analysis
            - 'spectral': Spectral clustering
        random_state : int
            Random seed for reproducibility
        logger : logging.Logger
            Logger for tracking the clustering process
        """
        self.method = method.lower()
        self.random_state = random_state
        self.cluster_model = None
        self.labels_ = None
        self.n_clusters_ = None
        self.feature_names_ = None
        self.sample_names_ = None
        self.is_fitted = False
        self.modularity_score_ = None
        self.silhouette_score_ = None
        
        # Set up logger
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
            
        # Validate method
        valid_methods = ['kmeans', 'kmedoids', 'hierarchical', 'dbscan', 
                         'gmm', 'wgcna', 'spectral']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Choose from: {', '.join(valid_methods)}")
        
        # Check requirements for specific methods
        if self.method == 'kmedoids' and not KMEDOIDS_AVAILABLE:
            raise ImportError("K-medoids requires scikit-learn-extra. "
                             "Install with 'pip install scikit-learn-extra'")
        
        if self.method == 'wgcna' and not R_AVAILABLE:
            raise ImportError("WGCNA requires rpy2 and R with WGCNA package. "
                             "Install rpy2 with 'pip install rpy2'")
    
    def _setup_logger(self):
        """Setup a basic logger if none is provided."""
        logger = logging.getLogger("BioinformaticsClustering")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def fit(self, X, y=None, **kwargs):
        """
        Fit the clustering model to the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix (samples x features)
        y : pandas.Series or numpy.ndarray, optional
            Ignored. This parameter exists for compatibility with sklearn API.
        **kwargs : 
            Additional parameters specific to the clustering method
            
        Returns:
        --------
        self : BioinformaticsClustering
            The fitted clustering model
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
            self.sample_names_ = X.index
            X_values = X.values
        else:
            X_values = X
            self.feature_names_ = [f"Feature_{i}" for i in range(X.shape[1])]
            self.sample_names_ = [f"Sample_{i}" for i in range(X.shape[0])]
            
        self.logger.info(f"Starting clustering with method: {self.method}")
        
        if self.method == 'kmeans':
            self._fit_kmeans(X_values, **kwargs)
        elif self.method == 'kmedoids':
            self._fit_kmedoids(X_values, **kwargs)
        elif self.method == 'hierarchical':
            self._fit_hierarchical(X_values, **kwargs)
        elif self.method == 'dbscan':
            self._fit_dbscan(X_values, **kwargs)
        elif self.method == 'gmm':
            self._fit_gmm(X_values, **kwargs)
        elif self.method == 'wgcna':
            if isinstance(X, pd.DataFrame):
                self._fit_wgcna(X, **kwargs)
            else:
                X_df = pd.DataFrame(X, columns=self.feature_names_, index=self.sample_names_)
                self._fit_wgcna(X_df, **kwargs)
        elif self.method == 'spectral':
            self._fit_spectral(X_values, **kwargs)
            
        self.is_fitted = True
        
        # Calculate silhouette score if possible and not already calculated
        if self.labels_ is not None and self.silhouette_score_ is None and len(np.unique(self.labels_)) > 1:
            try:
                self.silhouette_score_ = silhouette_score(X_values, self.labels_)
                self.logger.info(f"Silhouette score: {self.silhouette_score_:.4f}")
            except:
                self.logger.warning("Could not calculate silhouette score")
        
        return self
    
    def fit_predict(self, X, **kwargs):
        """
        Fit the clustering model and predict cluster labels.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        **kwargs : 
            Additional parameters specific to the clustering method
            
        Returns:
        --------
        numpy.ndarray
            Cluster labels
        """
        self.fit(X, **kwargs)
        return self.labels_
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Note: Some clustering methods like DBSCAN and hierarchical clustering
        don't naturally support prediction on new data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
            
        Returns:
        --------
        numpy.ndarray
            Predicted cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Cluster model not fitted. Call fit() first.")
            
        self.logger.info(f"Predicting clusters using fitted {self.method} model")
        
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if self.method in ['kmeans', 'gmm']:
            # These models have a predict method
            return self.cluster_model.predict(X_values)
        elif self.method == 'kmedoids' and KMEDOIDS_AVAILABLE:
            return self.cluster_model.predict(X_values)
        elif self.method == 'spectral':
            # For spectral, we can use the trained model's predict
            return self.cluster_model.predict(X_values)
        elif self.method == 'hierarchical':
            # For hierarchical, compute distances to cluster centers
            if not hasattr(self, 'cluster_centers_'):
                self.logger.warning("Hierarchical model doesn't have cluster centers. Computing them now.")
                # Compute cluster centers from original data
                self._compute_cluster_centers()
                
            # Assign each point to the nearest cluster center
            distances = distance.cdist(X_values, self.cluster_centers_)
            return np.argmin(distances, axis=1)
        elif self.method == 'dbscan':
            # For DBSCAN, we need to implement a custom predict function
            # This is done by assigning points to the nearest core point's cluster
            # or as noise if too far from any core point
            self.logger.warning("DBSCAN prediction on new data uses nearest core points heuristic")
            
            if not hasattr(self, 'components_'):
                self.logger.error("DBSCAN model doesn't have core samples. Cannot predict.")
                return np.full(X_values.shape[0], -1)
                
            # Get core samples and their labels
            core_mask = self.cluster_model.core_sample_indices_
            core_samples = self.X_fit_[core_mask]
            core_labels = self.labels_[core_mask]
            
            # Calculate distances to core samples
            dist = distance.cdist(X_values, core_samples)
            
            # Find nearest core sample for each point
            min_dist_idx = np.argmin(dist, axis=1)
            min_dist = np.min(dist, axis=1)
            
            # Get labels from nearest core points
            labels = core_labels[min_dist_idx]
            
            # Mark points beyond eps as noise
            labels[min_dist > self.cluster_model.eps] = -1
            
            return labels
        elif self.method == 'wgcna':
            self.logger.error("WGCNA doesn't support prediction on new data")
            return np.full(X_values.shape[0], -1)
        else:
            self.logger.error(f"Prediction not implemented for {self.method}")
            return np.full(X_values.shape[0], -1)
    
    def _fit_kmeans(self, X, **kwargs):
        """
        Fit K-means clustering model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The feature matrix
        **kwargs :
            n_clusters : int
                Number of clusters
            init : str
                Initialization method ('k-means++', 'random')
            max_iter : int
                Maximum number of iterations
            n_init : int
                Number of initializations
        """
        # Extract parameters
        n_clusters = kwargs.get('n_clusters', 3)
        init = kwargs.get('init', 'k-means++')
        max_iter = kwargs.get('max_iter', 300)
        n_init = kwargs.get('n_init', 10)
        
        # Initialize K-means
        self.cluster_model = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            n_init=n_init,
            random_state=self.random_state
        )
        
        # Fit the model
        self.cluster_model.fit(X)
        
        # Store cluster labels and number of clusters
        self.labels_ = self.cluster_model.labels_
        self.n_clusters_ = n_clusters
        
        # Log results
        self.logger.info(f"K-means clustering with {n_clusters} clusters completed")
        self.logger.info(f"Inertia: {self.cluster_model.inertia_:.4f}")
    
    def _fit_kmedoids(self, X, **kwargs):
        """
        Fit K-medoids (PAM) clustering model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The feature matrix
        **kwargs :
            n_clusters : int
                Number of clusters
            metric : str
                Distance metric
            method : str
                PAM algorithm variant
            init : str
                Initialization method
            max_iter : int
                Maximum number of iterations
        """
        if not KMEDOIDS_AVAILABLE:
            raise ImportError("K-medoids requires scikit-learn-extra")
            
        # Extract parameters
        n_clusters = kwargs.get('n_clusters', 3)
        metric = kwargs.get('metric', 'euclidean')
        method = kwargs.get('method', 'alternate')
        init = kwargs.get('init', 'k-medoids++')
        max_iter = kwargs.get('max_iter', 300)
        
        # Initialize K-medoids
        self.cluster_model = KMedoids(
            n_clusters=n_clusters,
            metric=metric,
            method=method,
            init=init,
            max_iter=max_iter,
            random_state=self.random_state
        )
        
        # Fit the model
        self.cluster_model.fit(X)
        
        # Store cluster labels and number of clusters
        self.labels_ = self.cluster_model.labels_
        self.n_clusters_ = n_clusters
        
        # Log results
        self.logger.info(f"K-medoids clustering with {n_clusters} clusters completed")
        self.logger.info(f"Inertia: {self.cluster_model.inertia_:.4f}")
    
    def _fit_hierarchical(self, X, **kwargs):
        """
        Fit hierarchical (agglomerative) clustering model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The feature matrix
        **kwargs :
            n_clusters : int
                Number of clusters
            linkage : str
                Linkage criterion ('ward', 'complete', 'average', 'single')
            affinity : str
                Distance metric
            compute_distances : bool
                Whether to compute and store the distance matrix
            compute_full_tree : bool or str
                Whether to compute the full tree or stop at n_clusters
        """
        # Extract parameters
        n_clusters = kwargs.get('n_clusters', 3)
        linkage = kwargs.get('linkage', 'ward')
        affinity = kwargs.get('affinity', 'euclidean')
        compute_distances = kwargs.get('compute_distances', False)
        compute_full_tree = kwargs.get('compute_full_tree', 'auto')
        
        # Initialize hierarchical clustering
        self.cluster_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            affinity=affinity,
            compute_distances=compute_distances,
            compute_full_tree=compute_full_tree
        )
        
        # Store original data for later use in prediction
        self.X_fit_ = X.copy()
        
        # Fit the model
        self.cluster_model.fit(X)
        
        # Store cluster labels and number of clusters
        self.labels_ = self.cluster_model.labels_
        self.n_clusters_ = n_clusters
        
        # Compute cluster centers for prediction
        self._compute_cluster_centers()
        
        # Log results
        self.logger.info(f"Hierarchical clustering with {n_clusters} clusters and {linkage} linkage completed")
    
    def _compute_cluster_centers(self):
        """Compute cluster centers for hierarchical clustering."""
        if not hasattr(self, 'X_fit_') or not hasattr(self, 'labels_'):
            raise ValueError("Model must be fitted first")
            
        # Compute cluster centers (centroids)
        self.cluster_centers_ = np.array([
            self.X_fit_[self.labels_ == i].mean(axis=0)
            for i in range(self.n_clusters_)
        ])
    
    def _fit_dbscan(self, X, **kwargs):
        """
        Fit DBSCAN clustering model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The feature matrix
        **kwargs :
            eps : float
                Maximum distance between samples for neighborhood
            min_samples : int
                Minimum number of samples in a neighborhood for a core point
            metric : str
                Distance metric
            algorithm : str
                Algorithm for nearest neighbors search
            leaf_size : int
                Leaf size for BallTree or KDTree
        """
        # Extract parameters
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        metric = kwargs.get('metric', 'euclidean')
        algorithm = kwargs.get('algorithm', 'auto')
        leaf_size = kwargs.get('leaf_size', 30)
        
        # Initialize DBSCAN
        self.cluster_model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size
        )
        
        # Store original data for later use in prediction
        self.X_fit_ = X.copy()
        
        # Fit the model
        self.cluster_model.fit(X)
        
        # Store cluster labels and number of clusters
        self.labels_ = self.cluster_model.labels_
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        
        # Log results
        n_noise = list(self.labels_).count(-1)
        self.logger.info(f"DBSCAN clustering completed. Found {self.n_clusters_} clusters and {n_noise} noise points")
    
    def _fit_gmm(self, X, **kwargs):
        """
        Fit Gaussian Mixture Model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The feature matrix
        **kwargs :
            n_components : int
                Number of mixture components (clusters)
            covariance_type : str
                Type of covariance parameters ('full', 'tied', 'diag', 'spherical')
            reg_covar : float
                Regularization for covariance matrices
            max_iter : int
                Maximum number of iterations
            init_params : str
                Initialization method
        """
        # Extract parameters
        n_components = kwargs.get('n_components', 3)
        covariance_type = kwargs.get('covariance_type', 'full')
        reg_covar = kwargs.get('reg_covar', 1e-6)
        max_iter = kwargs.get('max_iter', 100)
        init_params = kwargs.get('init_params', 'kmeans')
        
        # Initialize GMM
        self.cluster_model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            max_iter=max_iter,
            init_params=init_params,
            random_state=self.random_state
        )
        
        # Fit the model
        self.cluster_model.fit(X)
        
        # Store cluster labels and number of clusters
        self.labels_ = self.cluster_model.predict(X)
        self.n_clusters_ = n_components
        
        # Log results
        self.logger.info(f"GMM clustering with {n_components} components completed")
        self.logger.info(f"Converged: {self.cluster_model.converged_}")
        if hasattr(self.cluster_model, 'aic'):
            self.logger.info(f"AIC: {self.cluster_model.aic(X):.4f}")
            self.logger.info(f"BIC: {self.cluster_model.bic(X):.4f}")
    
    def _fit_wgcna(self, X, **kwargs):
        """
        Fit Weighted Gene Co-expression Network Analysis (WGCNA).
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix (samples x genes)
        **kwargs :
            power : int or float
                Soft threshold power for adjacency
            minModuleSize : int
                Minimum module size
            deepSplit : int
                Module detection sensitivity (0-4)
            mergeCutHeight : float
                Dendrogram cut height for module merging
            networkType : str
                Network type ('signed', 'unsigned', 'signed hybrid')
            corType : str
                Correlation type ('pearson', 'bicor')
            maxBlockSize : int
                Maximum block size for module detection
            reassignThreshold : float
                Threshold for module reassignment
            pamRespectsDendro : bool
                Whether PAM stage respects dendrogram
            verbose : int
                Verbosity level
        """
        if not R_AVAILABLE:
            raise ImportError("WGCNA requires rpy2 and R with WGCNA package")
            
        # Prepare parameters
        power = kwargs.get('power', 6)
        min_module_size = kwargs.get('minModuleSize', 30)
        deep_split = kwargs.get('deepSplit', 2)
        merge_cut_height = kwargs.get('mergeCutHeight', 0.25)
        network_type = kwargs.get('networkType', 'signed')
        cor_type = kwargs.get('corType', 'pearson')
        max_block_size = kwargs.get('maxBlockSize', 5000)
        reassign_threshold = kwargs.get('reassignThreshold', 0.2)
        pam_respects_dendro = kwargs.get('pamRespectsDendro', True)
        verbose = kwargs.get('verbose', 3)
        
        # Transpose if needed (WGCNA expects genes as rows)
        if X.shape[0] > X.shape[1]:  # If more samples than features
            self.logger.warning("Transposing data matrix for WGCNA (expects genes as rows)")
            X = X.T
        
        # Activate pandas conversion
        pandas2ri.activate()
        
        # Import WGCNA package
        wgcna = importr("WGCNA")
        
        # Set up R environment
        robjects.r("options(stringsAsFactors = FALSE)")
        
        # Convert DataFrame to R format
        r_data = pandas2ri.py2rpy(X)
        
        # Construct network
        self.logger.info("Constructing WGCNA network...")
        robjects.globalenv['data'] = r_data
        robjects.globalenv['power'] = power
        robjects.globalenv['networkType'] = network_type
        robjects.globalenv['corType'] = cor_type
        
        # Compute adjacency matrix
        robjects.r("""
        adjacency = adjacency(data, power=power, type=networkType, corFnc=corType)
        TOM = TOMsimilarity(adjacency, TOMType="unsigned")
        dissTOM = 1 - TOM
        """)
        
        # Hierarchical clustering
        robjects.r("""
        geneTree = hclust(as.dist(dissTOM), method="average")
        """)
        
        # Module identification
        robjects.r(f"""
        dynamicMods = cutreeDynamic(
            dendro=geneTree, 
            distM=dissTOM, 
            deepSplit={deep_split}, 
            pamRespectsDendro={str(pam_respects_dendro).upper()}, 
            minClusterSize={min_module_size}
        )
        """)
        
        # Module merging
        robjects.r(f"""
        # Convert numeric labels to colors
        dynamicColors = labels2colors(dynamicMods)
        
        # Calculate module eigengenes
        MEList = moduleEigengenes(data, colors=dynamicColors)
        MEs = MEList$eigengenes
        
        # Calculate dissimilarity of module eigengenes
        MEDiss = 1 - cor(MEs)
        
        # Cluster module eigengenes
        METree = hclust(as.dist(MEDiss), method="average")
        
        # Merge close modules
        merge = mergeCloseModules(data, dynamicColors, cutHeight={merge_cut_height}, verbose={verbose})
        mergedColors = merge$colors
        mergedMEs = merge$newMEs
        """)
        
        # Get module assignments
        module_colors = robjects.r("mergedColors")
        
        # Convert module colors to numeric labels
        unique_colors = np.unique(module_colors)
        color_to_label = {color: i for i, color in enumerate(unique_colors)}
        labels = np.array([color_to_label[color] for color in module_colors])
        
        # Store results
        self.labels_ = labels
        self.n_clusters_ = len(unique_colors)
        self.module_colors_ = np.array(module_colors)
        self.module_eigengenes_ = pandas2ri.rpy2py(robjects.r("mergedMEs"))
        
        # Store module membership and significance
        robjects.r("""
        # Calculate module membership and significance
        datKME = signedKME(data, mergedMEs)
        """)
        self.module_membership_ = pandas2ri.rpy2py(robjects.r("datKME"))
        
        # Compute modularity score (quality of modules)
        robjects.r("""
        # Compute modularity
        ADJ = adjacency(data, power=power, type=networkType)
        modularity = modularityStatistics(ADJ, mergedColors)$modularity
        """)
        self.modularity_score_ = robjects.r("modularity")[0]
        
        # Clean up
        pandas2ri.deactivate()
        
        # Log results
        self.logger.info(f"WGCNA analysis completed. Found {self.n_clusters_} modules")
        self.logger.info(f"Modularity score: {self.modularity_score_:.4f}")
    
    def _fit_spectral(self, X, **kwargs):
        """
        Fit Spectral clustering model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The feature matrix
        **kwargs :
            n_clusters : int
                Number of clusters
            affinity : str
                Affinity type ('nearest_neighbors', 'rbf', 'precomputed')
            n_neighbors : int
                Number of neighbors for affinity matrix
            gamma : float
                Kernel coefficient for rbf kernel
            assign_labels : str
                Strategy for assigning labels ('kmeans', 'discretize')
            n_init : int
                Number of initializations for k-means
            eigen_solver : str
                Eigenvalue decomposition strategy
        """
        # Extract parameters
        n_clusters = kwargs.get('n_clusters', 3)
        affinity = kwargs.get('affinity', 'rbf')
        n_neighbors = kwargs.get('n_neighbors', 10)
        gamma = kwargs.get('gamma', 1.0)
        assign_labels = kwargs.get('assign_labels', 'kmeans')
        n_init = kwargs.get('n_init', 10)
        eigen_solver = kwargs.get('eigen_solver', None)
        
        # Initialize spectral clustering
        self.cluster_model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            n_neighbors=n_neighbors,
            gamma=gamma,
            assign_labels=assign_labels,
            n_init=n_init,
            eigen_solver=eigen_solver,
            random_state=self.random_state
        )
        
        # Fit the model
        self.cluster_model.fit(X)
        
        # Store cluster labels and number of clusters
        self.labels_ = self.cluster_model.labels_
        self.n_clusters_ = n_clusters
        
        # Log results
        self.logger.info(f"Spectral clustering with {n_clusters} clusters completed")
    
    def evaluate_clusters(self, X, y_true=None, **kwargs):
        """
        Evaluate clustering performance using internal and external metrics.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y_true : pandas.Series or numpy.ndarray, optional
            True labels for external validation
        **kwargs :
            Additional parameters
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Cluster model not fitted. Call fit() first.")
            
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        self.logger.info("Evaluating clustering performance")
        
        # Initialize metrics dictionary
        metrics_dict = {}
        
        # Check if we have cluster labels
        if self.labels_ is None:
            self.logger.error("No cluster labels available for evaluation")
            return metrics_dict
            
        # Get number of unique labels (excluding noise points)
        unique_labels = np.unique(self.labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Skip evaluation if only one cluster or all noise
        if n_clusters <= 1:
            self.logger.warning("Cannot evaluate clustering with 0 or 1 clusters")
            metrics_dict['n_clusters'] = n_clusters
            return metrics_dict
            
        # Internal validation metrics (don't require true labels)
        try:
            # Silhouette score (higher is better)
            silhouette = silhouette_score(X_values, self.labels_)
            metrics_dict['silhouette_score'] = silhouette
            
            # Davies-Bouldin index (lower is better)
            davies_bouldin = davies_bouldin_score(X_values, self.labels_)
            metrics_dict['davies_bouldin_score'] = davies_bouldin
            
            # Calinski-Harabasz index (higher is better)
            calinski_harabasz = calinski_harabasz_score(X_values, self.labels_)
            metrics_dict['calinski_harabasz_score'] = calinski_harabasz
            
            # Store silhouette score in object
            self.silhouette_score_ = silhouette
            
        except Exception as e:
            self.logger.warning(f"Error computing internal validation metrics: {str(e)}")
            
        # External validation metrics (require true labels)
        if y_true is not None:
            try:
                # Adjusted Rand index (higher is better)
                ari = adjusted_rand_score(y_true, self.labels_)
                metrics_dict['adjusted_rand_score'] = ari
                
                # Adjusted mutual information (higher is better)
                ami = adjusted_mutual_info_score(y_true, self.labels_)
                metrics_dict['adjusted_mutual_info_score'] = ami
                
            except Exception as e:
                self.logger.warning(f"Error computing external validation metrics: {str(e)}")
        
        # Add WGCNA-specific metrics
        if self.method == 'wgcna' and hasattr(self, 'modularity_score_'):
            metrics_dict['modularity_score'] = self.modularity_score_
            
        # Log results
        self.logger.info(f"Evaluation metrics: {metrics_dict}")
        
        return metrics_dict
    
    def determine_optimal_clusters(self, X, max_clusters=10, methods=None, **kwargs):
        """
        Determine the optimal number of clusters using multiple methods.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        max_clusters : int
            Maximum number of clusters to consider
        methods : list
            List of methods to use for determining optimal clusters:
            - 'elbow': Elbow method (for KMeans)
            - 'silhouette': Silhouette score
            - 'gap': Gap statistic
            - 'bic': Bayesian Information Criterion (for GMM)
            - 'calinski_harabasz': Calinski-Harabasz index
            - 'davies_bouldin': Davies-Bouldin index
        **kwargs :
            Additional parameters specific to clustering methods
            
        Returns:
        --------
        dict
            Dictionary with results from each method and recommended number of clusters
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        # Set default methods based on clustering algorithm
        if methods is None:
            if self.method == 'kmeans' or self.method == 'kmedoids':
                methods = ['elbow', 'silhouette', 'calinski_harabasz']
            elif self.method == 'hierarchical':
                methods = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
            elif self.method == 'gmm':
                methods = ['bic', 'silhouette']
            elif self.method == 'spectral':
                methods = ['silhouette', 'calinski_harabasz']
            else:
                methods = ['silhouette']
                
        self.logger.info(f"Determining optimal number of clusters using methods: {methods}")
        
        # Initialize results dictionary
        results = {method: [] for method in methods}
        results['n_clusters'] = list(range(2, max_clusters + 1))
        
        # Store original method and parameters
        original_method = self.method
        original_params = kwargs.copy()
        
        # Evaluate each number of clusters
        for n_clusters in range(2, max_clusters + 1):
            self.logger.info(f"Evaluating {n_clusters} clusters")
            
            # Update parameters for this run
            run_params = original_params.copy()
            
            if 'elbow' in methods and self.method in ['kmeans', 'kmedoids']:
                # Fit model with current number of clusters
                run_params['n_clusters'] = n_clusters
                self.fit(X_values, **run_params)
                
                # Store inertia for elbow method
                if hasattr(self.cluster_model, 'inertia_'):
                    results['elbow'].append(self.cluster_model.inertia_)
                else:
                    results['elbow'].append(None)
                    
            # Metrics that apply to most clustering methods
            if any(m in methods for m in ['silhouette', 'calinski_harabasz', 'davies_bouldin']):
                # Skip if already fitted with correct parameters
                if not (hasattr(self, 'n_clusters_') and self.n_clusters_ == n_clusters):
                    run_params['n_clusters'] = n_clusters
                    self.fit(X_values, **run_params)
                
                # Calculate metrics
                labels = self.labels_
                
                if 'silhouette' in methods:
                    try:
                        score = silhouette_score(X_values, labels)
                        results['silhouette'].append(score)
                    except:
                        results['silhouette'].append(None)
                        
                if 'calinski_harabasz' in methods:
                    try:
                        score = calinski_harabasz_score(X_values, labels)
                        results['calinski_harabasz'].append(score)
                    except:
                        results['calinski_harabasz'].append(None)
                        
                if 'davies_bouldin' in methods:
                    try:
                        score = davies_bouldin_score(X_values, labels)
                        results['davies_bouldin'].append(score)
                    except:
                        results['davies_bouldin'].append(None)
            
            # BIC for GMM
            if 'bic' in methods and self.method == 'gmm':
                # Skip if already fitted with correct parameters
                if not (hasattr(self, 'n_clusters_') and self.n_clusters_ == n_clusters):
                    run_params['n_components'] = n_clusters
                    self.fit(X_values, **run_params)
                    
                # Calculate BIC
                if hasattr(self.cluster_model, 'bic'):
                    bic = self.cluster_model.bic(X_values)
                    results['bic'].append(bic)
                else:
                    results['bic'].append(None)
                    
            # Gap statistic (computationally expensive)
            if 'gap' in methods:
                try:
                    from sklearn.cluster import KMeans
                    from sklearn_extra.cluster import KMedoids
                    
                    # Use selected clustering algorithm
                    if self.method == 'kmeans':
                        model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                    elif self.method == 'kmedoids':
                        model = KMedoids(n_clusters=n_clusters, random_state=self.random_state)
                    else:
                        # Default to KMeans for other methods
                        model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                        
                    # Calculate gap statistic using bootstrapping
                    gap = self._gap_statistic(X_values, model, n_clusters, n_refs=5)
                    results['gap'].append(gap)
                except Exception as e:
                    self.logger.warning(f"Error computing gap statistic: {str(e)}")
                    results['gap'].append(None)
        
        # Determine optimal number of clusters for each method
        recommendations = {}
        
        for method in methods:
            if method not in results or not results[method]:
                continue
                
            values = np.array(results[method])
            
            if method in ['elbow', 'davies_bouldin', 'bic']:
                # Lower is better, look for the elbow point
                optimal = self._find_elbow_point(
                    results['n_clusters'], values)
            elif method in ['silhouette', 'calinski_harabasz', 'gap']:
                # Higher is better, take the max
                if any(values is not None for values in results[method]):
                    optimal = results['n_clusters'][np.nanargmax(values)]
                else:
                    optimal = None
            else:
                optimal = None
                
            recommendations[method] = optimal
            
        # Overall recommendation (majority vote)
        valid_recommendations = [r for r in recommendations.values() if r is not None]
        if valid_recommendations:
            recommendations['overall'] = int(np.median(valid_recommendations))
        else:
            recommendations['overall'] = None
            
        # Log results
        self.logger.info(f"Optimal cluster recommendations: {recommendations}")
        
        return {
            'metrics': results,
            'recommendations': recommendations
        }
        
    def _find_elbow_point(self, x, y):
        """
        Find the elbow point in a curve using the point of maximum curvature.
        
        Parameters:
        -----------
        x : array-like
            X-values (number of clusters)
        y : array-like
            Y-values (inertia, BIC, etc.)
            
        Returns:
        --------
        int
            The x-value at the elbow point
        """
        # Filter out None values
        valid_indices = [i for i, val in enumerate(y) if val is not None]
        if not valid_indices:
            return None
            
        x_valid = [x[i] for i in valid_indices]
        y_valid = [y[i] for i in valid_indices]
        
        if len(x_valid) < 3:
            return x_valid[0]
            
        # Normalize data
        x_norm = np.array(x_valid)
        y_norm = np.array(y_valid)
        
        x_range = max(x_norm) - min(x_norm)
        y_range = max(y_norm) - min(y_norm)
        
        if x_range == 0 or y_range == 0:
            return x_valid[0]
            
        x_norm = (x_norm - min(x_norm)) / x_range
        y_norm = (y_norm - min(y_norm)) / y_range
        
        # Calculate curvature
        dx = np.gradient(x_norm)
        dy = np.gradient(y_norm)
        d2y = np.gradient(dy)
        
        curvature = np.abs(d2y) / (1 + dy**2)**1.5
        
        # Find point of maximum curvature
        elbow_idx = np.argmax(curvature)
        
        return x_valid[elbow_idx]
    
    def _gap_statistic(self, X, model, n_clusters, n_refs=5, random_state=None):
        """
        Calculate the gap statistic for evaluating the optimal number of clusters.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data matrix
        model : object
            Clustering model with fit and predict methods
        n_clusters : int
            Number of clusters
        n_refs : int
            Number of reference datasets
        random_state : int
            Random seed
            
        Returns:
        --------
        float
            Gap statistic value
        """
        # Generate reference datasets from random uniform distribution
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        
        # Dispersion of the original data
        model.fit(X)
        labels = model.predict(X)
        dispersion_orig = self._calculate_dispersion(X, labels)
        
        # Calculate dispersion of reference datasets
        dispersion_refs = []
        
        for i in range(n_refs):
            # Generate random data with same range as original
            X_ref = np.random.uniform(X_min, X_max, size=X.shape)
            
            # Cluster the reference data
            model.fit(X_ref)
            ref_labels = model.predict(X_ref)
            
            # Calculate dispersion
            dispersion_ref = self._calculate_dispersion(X_ref, ref_labels)
            dispersion_refs.append(dispersion_ref)
            
        # Calculate gap statistic
        gap = np.log(np.mean(dispersion_refs)) - np.log(dispersion_orig)
        
        return gap
    
    def _calculate_dispersion(self, X, labels):
        """
        Calculate the dispersion (within-cluster sum of squares) for gap statistic.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data matrix
        labels : numpy.ndarray
            Cluster labels
            
        Returns:
        --------
        float
            Dispersion value
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        total_dispersion = 0
        
        for k in unique_labels:
            if k == -1:  # Skip noise points
                continue
                
            # Get points in this cluster
            cluster_points = X[labels == k]
            
            if len(cluster_points) <= 1:
                continue
                
            # Calculate cluster centroid
            centroid = cluster_points.mean(axis=0)
            
            # Calculate within-cluster sum of squares
            cluster_dispersion = np.sum(np.sum((cluster_points - centroid)**2, axis=1))
            total_dispersion += cluster_dispersion
            
        return total_dispersion
    
    def plot_clusters(self, X=None, y=None, method='pca', **kwargs):
        """
        Plot clusters in 2D using dimensionality reduction.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray, optional
            Data matrix (if not using precomputed coordinates)
        y : pandas.Series or numpy.ndarray, optional
            True labels for comparison
        method : str
            Dimensionality reduction method ('pca', 'tsne', 'umap')
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size
            - alpha: Point transparency
            - s: Point size
            - title: Plot title
            - show_legend: Whether to show the legend
            - n_components: Number of components for dimensionality reduction
            - random_state: Random seed for dimensionality reduction
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if not self.is_fitted:
            raise ValueError("Cluster model not fitted. Call fit() first.")
            
        if self.labels_ is None:
            raise ValueError("No cluster labels available for plotting")
            
        # Extract plotting parameters
        figsize = kwargs.get('figsize', (12, 10))
        alpha = kwargs.get('alpha', 0.7)
        s = kwargs.get('s', 50)
        title = kwargs.get('title', f"Cluster Visualization ({self.method.upper()})")
        show_legend = kwargs.get('show_legend', True)
        
        # If X is provided, perform dimensionality reduction
        if X is not None:
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
                
            # Dimensionality reduction parameters
            n_components = kwargs.get('n_components', 2)
            random_state = kwargs.get('random_state', self.random_state)
            
            # Perform dimensionality reduction
            if method == 'pca':
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components, random_state=random_state)
                X_reduced = reducer.fit_transform(X_values)
                
                # Get explained variance for axis labels
                explained_var = reducer.explained_variance_ratio_
                x_label = f"PC1 ({explained_var[0]:.1%})"
                y_label = f"PC2 ({explained_var[1]:.1%})"
                
            elif method == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=n_components, random_state=random_state)
                X_reduced = reducer.fit_transform(X_values)
                
                x_label = "t-SNE 1"
                y_label = "t-SNE 2"
                
            elif method == 'umap':
                try:
                    import umap
                    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
                    X_reduced = reducer.fit_transform(X_values)
                    
                    x_label = "UMAP 1"
                    y_label = "UMAP 2"
                except ImportError:
                    self.logger.warning("UMAP not installed. Falling back to PCA.")
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=n_components, random_state=random_state)
                    X_reduced = reducer.fit_transform(X_values)
                    
                    explained_var = reducer.explained_variance_ratio_
                    x_label = f"PC1 ({explained_var[0]:.1%})"
                    y_label = f"PC2 ({explained_var[1]:.1%})"
            else:
                raise ValueError(f"Unknown dimensionality reduction method: {method}")
        else:
            # Use precomputed coordinates (assuming 2D)
            X_reduced = kwargs.get('coordinates')
            if X_reduced is None:
                raise ValueError("Either X or coordinates must be provided")
                
            x_label = kwargs.get('x_label', 'Component 1')
            y_label = kwargs.get('y_label', 'Component 2')
        
        # Extract first two components for plotting
        if X_reduced.shape[1] < 2:
            raise ValueError("Need at least 2 components for visualization")
            
        x_coords = X_reduced[:, 0]
        y_coords = X_reduced[:, 1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot clusters
        unique_labels = np.unique(self.labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Get colormap
        colormap = kwargs.get('colormap', 'tab10')
        if n_clusters > 10 and colormap == 'tab10':
            colormap = 'tab20'
            
        colors = plt.cm.get_cmap(colormap, max(10, n_clusters))
        
        # Plot each cluster
        for i, cluster in enumerate(unique_labels):
            if cluster == -1:
                # Plot noise points in black
                cluster_points = (self.labels_ == cluster)
                ax.scatter(
                    x_coords[cluster_points], 
                    y_coords[cluster_points],
                    alpha=alpha,
                    s=s,
                    c='black',
                    marker='x',
                    label='Noise'
                )
            else:
                # Plot regular clusters with colors
                cluster_points = (self.labels_ == cluster)
                ax.scatter(
                    x_coords[cluster_points], 
                    y_coords[cluster_points],
                    alpha=alpha,
                    s=s,
                    c=[colors(i % colors.N)],
                    label=f'Cluster {cluster}'
                )
        
        # Add true labels as markers if provided
        if y is not None:
            # Plot a second scatter with hollow markers for true labels
            unique_y = np.unique(y)
            markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
            
            for i, label in enumerate(unique_y):
                mask = (y == label)
                ax.scatter(
                    x_coords[mask], 
                    y_coords[mask],
                    s=s*1.5,
                    facecolors='none',
                    edgecolors='black',
                    linewidths=1.5,
                    marker=markers[i % len(markers)],
                    label=f'True: {label}'
                )
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Add grid
        ax.grid(alpha=0.3)
        
        # Add legend if requested
        if show_legend:
            if n_clusters > 10:
                # Put legend outside for many clusters
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_dendrogram(self, X, **kwargs):
        """
        Plot dendrogram for hierarchical clustering.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Data matrix
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size
            - title: Plot title
            - orientation: Dendrogram orientation ('top', 'bottom', 'left', 'right')
            - labels: Labels for leaves
            - color_threshold: Color threshold for dendrogram
            - linkage_method: Linkage method ('ward', 'complete', 'average', 'single')
            - metric: Distance metric
            - truncate_mode: Dendrogram truncation mode
            - p: Truncation parameter
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            sample_labels = X.index
        else:
            X_values = X
            sample_labels = [f"Sample_{i}" for i in range(X_values.shape[0])]
            
        # Extract parameters
        figsize = kwargs.get('figsize', (12, 8))
        title = kwargs.get('title', 'Hierarchical Clustering Dendrogram')
        orientation = kwargs.get('orientation', 'top')
        color_threshold = kwargs.get('color_threshold', 0.7 * np.max(self._linkage_matrix))
        truncate_mode = kwargs.get('truncate_mode', None)
        p = kwargs.get('p', 30)
        
        # Use provided labels or sample names
        labels = kwargs.get('labels', sample_labels)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute linkage matrix if needed
        if not hasattr(self, '_linkage_matrix'):
            linkage_method = kwargs.get('linkage_method', 'ward')
            metric = kwargs.get('metric', 'euclidean')
            
            # Compute distance matrix
            if metric == 'euclidean':
                dist_matrix = distance.pdist(X_values)
            else:
                dist_matrix = distance.pdist(X_values, metric=metric)
                
            # Compute linkage
            self._linkage_matrix = hierarchy.linkage(dist_matrix, method=linkage_method)
        
        # Plot dendrogram
        hierarchy.dendrogram(
            self._linkage_matrix,
            ax=ax,
            orientation=orientation,
            labels=labels,
            color_threshold=color_threshold,
            truncate_mode=truncate_mode,
            p=p
        )
        
        # Add title
        ax.set_title(title)
        
        # Adjust layout based on orientation
        if orientation in ['top', 'bottom']:
            ax.set_xlabel('Samples')
            ax.set_ylabel('Distance')
            # Rotate labels if there are many
            if len(labels) > 10:
                plt.xticks(rotation=90)
        else:
            ax.set_xlabel('Distance')
            ax.set_ylabel('Samples')
        
        plt.tight_layout()
        return fig
    
    def plot_silhouette(self, X, **kwargs):
        """
        Plot silhouette analysis for clustering.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Data matrix
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size
            - title: Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if not self.is_fitted:
            raise ValueError("Cluster model not fitted. Call fit() first.")
            
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        # Check that we have cluster labels
        if self.labels_ is None:
            raise ValueError("No cluster labels available for silhouette analysis")
            
        # Skip silhouette analysis for improper clustering
        unique_labels = np.unique(self.labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if n_clusters <= 1:
            self.logger.warning("Cannot perform silhouette analysis with 0 or 1 clusters")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Silhouette analysis requires at least 2 clusters", 
                   ha='center', va='center')
            return fig
            
        # Extract parameters
        figsize = kwargs.get('figsize', (12, 8))
        title = kwargs.get('title', 'Silhouette Analysis')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Calculate silhouette scores for each sample
        try:
            silhouette_vals = metrics.silhouette_samples(X_values, self.labels_)
            silhouette_avg = np.mean(silhouette_vals)
        except:
            self.logger.warning("Error calculating silhouette scores")
            fig.text(0.5, 0.5, "Error calculating silhouette scores", 
                    ha='center', va='center')
            return fig
        
        # The 1st subplot shows the silhouette plot
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, X_values.shape[0] + (n_clusters + 1) * 10])
        
        y_lower = 10
        
        # Get colormap
        colormap = kwargs.get('colormap', 'tab10')
        if n_clusters > 10 and colormap == 'tab10':
            colormap = 'tab20'
            
        colors = plt.cm.get_cmap(colormap, max(10, n_clusters))
        
        # Plot silhouette scores for each cluster
        for i, cluster in enumerate(unique_labels):
            if cluster == -1:
                continue  # Skip noise points
                
            # Get silhouette values for this cluster
            cluster_silhouette_vals = silhouette_vals[self.labels_ == cluster]
            cluster_silhouette_vals.sort()
            
            size_cluster = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster
            
            # Fill the silhouette
            color = colors(i % colors.N)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_vals,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            
            # Label the silhouette plots with cluster numbers
            ax1.text(-0.05, y_lower + 0.5 * size_cluster, f'Cluster {cluster}')
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10
            
        # Add average silhouette score line
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_title("Silhouette plot for the clusters")
        ax1.set_xlabel("Silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        
        # 2nd subplot shows the actual clusters
        try:
            # Use PCA to reduce to 2D for visualization
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=self.random_state)
            X_pca = pca.fit_transform(X_values)
            
            # Plot each cluster
            for i, cluster in enumerate(unique_labels):
                if cluster == -1:
                    # Plot noise points in black
                    cluster_points = (self.labels_ == cluster)
                    ax2.scatter(
                        X_pca[cluster_points, 0], 
                        X_pca[cluster_points, 1],
                        alpha=0.7,
                        s=50,
                        c='black',
                        marker='x',
                        label='Noise'
                    )
                else:
                    # Plot regular clusters with colors
                    cluster_points = (self.labels_ == cluster)
                    ax2.scatter(
                        X_pca[cluster_points, 0], 
                        X_pca[cluster_points, 1],
                        alpha=0.7,
                        s=50,
                        c=[colors(i % colors.N)],
                        label=f'Cluster {cluster}'
                    )
                    
            # Add cluster centers if available
            if hasattr(self, 'cluster_centers_') and self.method in ['kmeans', 'kmedoids']:
                # Transform centers to PCA space
                centers_pca = pca.transform(self.cluster_model.cluster_centers_)
                ax2.scatter(
                    centers_pca[:, 0],
                    centers_pca[:, 1],
                    s=200,
                    marker='*',
                    c='red',
                    alpha=0.8,
                    label='Centroids'
                )
                
            ax2.set_title("Cluster visualization (PCA)")
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            ax2.legend()
            ax2.grid(alpha=0.3)
            
        except Exception as e:
            self.logger.warning(f"Error creating cluster visualization: {str(e)}")
            ax2.text(0.5, 0.5, "Error creating visualization", 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Add overall title
        fig.suptitle(
            f"{title}\nAverage silhouette score: {silhouette_avg:.3f}",
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def plot_wgcna_network(self, min_module_size=30, **kwargs):
        """
        Plot WGCNA network visualization.
        
        Parameters:
        -----------
        min_module_size : int
            Minimum size of modules to include in visualization
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size
            - title: Plot title
            - node_size: Size of nodes
            - edge_threshold: Threshold for edge weights
            
        Returns:
        --------
        matplotlib.figure.Figure or None
            The plot figure or None if WGCNA not fitted
        """
        if self.method != 'wgcna' or not hasattr(self, 'module_colors_'):
            self.logger.error("WGCNA network visualization only available for WGCNA clustering")
            return None
            
        try:
            import networkx as nx
        except ImportError:
            self.logger.error("networkx not installed. Cannot create network visualization.")
            return None
            
        # Extract parameters
        figsize = kwargs.get('figsize', (12, 12))
        title = kwargs.get('title', 'WGCNA Network Visualization')
        node_size = kwargs.get('node_size', 100)
        edge_threshold = kwargs.get('edge_threshold', 0.5)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get module information
        module_colors = np.unique(self.module_colors_)
        
        # Create network
        G = nx.Graph()
        
        # Add nodes and edges from module membership
        for i, color in enumerate(module_colors):
            if color == 'grey':  # Skip unassigned genes
                continue
                
            # Get genes in this module
            module_genes = np.where(self.module_colors_ == color)[0]
            
            # Skip small modules
            if len(module_genes) < min_module_size:
                continue
                
            # Get module eigengene name
            me_name = f"ME{color}"
            
            # Add nodes for each gene in the module
            for gene_idx in module_genes:
                gene_name = self.feature_names_[gene_idx]
                
                # Get gene membership strength to this module
                if hasattr(self, 'module_membership_'):
                    try:
                        membership = abs(self.module_membership_.iloc[gene_idx][me_name])
                    except:
                        membership = 0.5  # Default if membership info not available
                else:
                    membership = 0.5
                    
                # Add node with attributes
                G.add_node(
                    gene_name,
                    module=color,
                    membership=membership,
                    node_type='gene'
                )
            
            # Add edges between genes with high membership
            for idx1, gene1_idx in enumerate(module_genes):
                gene1 = self.feature_names_[gene1_idx]
                member1 = G.nodes[gene1]['membership']
                
                # Only connect genes with membership above threshold
                if member1 < edge_threshold:
                    continue
                    
                # Add edges to other genes in the same module
                for gene2_idx in module_genes[idx1+1:]:
                    gene2 = self.feature_names_[gene2_idx]
                    member2 = G.nodes[gene2]['membership']
                    
                    # Only connect to genes with membership above threshold
                    if member2 < edge_threshold:
                        continue
                        
                    # Edge weight based on product of memberships
                    weight = min(member1, member2)
                    
                    # Add edge if weight above threshold
                    if weight >= edge_threshold:
                        G.add_edge(gene1, gene2, weight=weight)
        
        # Check if network is empty
        if len(G.nodes) == 0:
            self.logger.warning("No modules found above the minimum size threshold")
            ax.text(0.5, 0.5, "No modules found above the minimum size threshold", 
                   ha='center', va='center')
            return fig
            
        # Create layout
        layout = nx.spring_layout(G, k=0.2, iterations=100)
        
        # Draw the network
        # Node colors based on module
        node_colors = [G.nodes[n]['module'] for n in G.nodes]
        
        # Node sizes based on membership
        node_sizes = [G.nodes[n]['membership'] * node_size for n in G.nodes]
        
        # Edge widths based on weight
        edge_weights = [G.edges[e]['weight'] * 2 for e in G.edges]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, layout,
            ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, layout,
            ax=ax,
            width=edge_weights,
            alpha=0.3,
            edge_color='gray'
        )
        
        # Add labels for larger nodes
        large_nodes = [n for n in G.nodes if G.nodes[n]['membership'] > 0.7]
        nx.draw_networkx_labels(
            G, layout,
            ax=ax,
            font_size=8,
            font_weight='bold',
            labels={n: n for n in large_nodes}
        )
        
        # Add title
        plt.title(title)
        
        # Turn off axis
        plt.axis('off')
        
        # Add module color legend
        unique_colors = list(set(node_colors))
        if len(unique_colors) <= 10:  # Only show legend if not too many colors
            for i, color in enumerate(unique_colors):
                plt.plot([0], [0], 'o', color=color, label=f'Module {color}')
            plt.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_validation(self, X, max_clusters=10, methods=None, **kwargs):
        """
        Plot validation metrics for different numbers of clusters.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Data matrix
        max_clusters : int
            Maximum number of clusters to consider
        methods : list
            List of methods to plot
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size
            - title: Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        # Get optimal clusters data
        results = self.determine_optimal_clusters(X, max_clusters, methods, **kwargs)
        
        metrics_data = results['metrics']
        recommendations = results['recommendations']
        
        # Extract parameters
        figsize = kwargs.get('figsize', (14, 10))
        title = kwargs.get('title', 'Cluster Validation Metrics')
        
        # Determine number of subplots
        active_methods = [m for m in metrics_data.keys() if m != 'n_clusters' and metrics_data[m]]
        n_plots = len(active_methods)
        
        if n_plots == 0:
            self.logger.warning("No validation metrics available for plotting")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No validation metrics available", 
                   ha='center', va='center')
            return fig
            
        # Calculate grid size
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Ensure axes is a 2D array for consistent indexing
        if n_plots == 1:
            axes = np.array([[axes]])
        elif n_plots <= 2:
            axes = axes.reshape(1, -1)
        
        # Plot each metric
        for i, method in enumerate(active_methods):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Get data
            x = metrics_data['n_clusters']
            y = metrics_data[method]
            
            # Replace None values with NaN for plotting
            y = [float('nan') if v is None else v for v in y]
            
            # Plot the metric
            line = ax.plot(x, y, 'o-', label=method)
            color = line[0].get_color()
            
            # Mark the recommended value
            if method in recommendations and recommendations[method] is not None:
                opt_k = recommendations[method]
                opt_idx = x.index(opt_k)
                opt_y = y[opt_idx]
                
                ax.scatter([opt_k], [opt_y], s=100, c=color, marker='*', edgecolor='black')
                ax.text(opt_k, opt_y, f'  k={opt_k}', verticalalignment='bottom')
            
            # Set labels
            ax.set_title(f"{method.replace('_', ' ').title()}")
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('Score')
            
            # Add grid
            ax.grid(alpha=0.3)
            
            # Set x-axis ticks
            ax.set_xticks(x)
            
            # For metrics where lower is better, add note
            if method in ['elbow', 'davies_bouldin', 'bic']:
                ax.text(0.05, 0.05, 'Lower is better', transform=ax.transAxes, 
                       bbox=dict(facecolor='white', alpha=0.7))
            else:
                ax.text(0.05, 0.05, 'Higher is better', transform=ax.transAxes,
                      bbox=dict(facecolor='white', alpha=0.7))
        
        # Remove unused subplots
        for i in range(n_plots, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        # Add overall title with recommendation
        if 'overall' in recommendations and recommendations['overall'] is not None:
            overall = recommendations['overall']
            fig.suptitle(
                f"{title}\nRecommended number of clusters: {overall}",
                fontsize=14, fontweight='bold'
            )
        else:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def plot_heatmap(self, X, **kwargs):
        """
        Plot heatmap of data with cluster annotations.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Data matrix
        **kwargs :
            Additional parameters for plotting:
            - figsize: Figure size
            - title: Plot title
            - cmap: Colormap
            - zscore: Whether to z-score normalize the data
            - cluster_genes: Whether to cluster genes/features
            - method: Linkage method for hierarchical clustering
            - metric: Distance metric for hierarchical clustering
            - n_genes: Maximum number of genes to show (for large datasets)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if not self.is_fitted:
            raise ValueError("Cluster model not fitted. Call fit() first.")
            
        # Process input data
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        else:
            # Convert to DataFrame with default names
            data = pd.DataFrame(
                X, 
                index=[f"Sample_{i}" for i in range(X.shape[0])],
                columns=[f"Feature_{i}" for i in range(X.shape[1])]
            )
            
        # Extract parameters
        figsize = kwargs.get('figsize', (12, 10))
        title = kwargs.get('title', 'Expression Heatmap with Clusters')
        cmap = kwargs.get('cmap', 'viridis')
        zscore = kwargs.get('zscore', True)
        cluster_genes = kwargs.get('cluster_genes', True)
        method = kwargs.get('method', 'average')
        metric = kwargs.get('metric', 'euclidean')
        
        # Limit number of genes for large datasets
        n_genes = kwargs.get('n_genes', 100)
        if data.shape[1] > n_genes:
            self.logger.warning(f"Data has {data.shape[1]} features. Showing only top {n_genes}.")
            
            # Calculate variance of each feature
            feature_vars = data.var(axis=0).sort_values(ascending=False)
            top_features = feature_vars.index[:n_genes]
            data = data[top_features]
        
        # Z-score normalize if requested
        if zscore:
            if data.shape[0] > 1:  # Need at least 2 samples for z-scoring
                data = (data - data.mean(axis=0)) / data.std(axis=0, ddof=0)
                # Replace NaN values (from features with std=0)
                data.fillna(0, inplace=True)
        
        # Sort by cluster
        ordered_index = data.index[np.argsort(self.labels_)]
        data = data.loc[ordered_index]
        
        # Cluster genes/features if requested
        if cluster_genes and data.shape[1] > 1:
            from scipy.cluster import hierarchy
            
            # Compute linkage for genes
            gene_linkage = hierarchy.linkage(
                data.T, 
                method=method,
                metric=metric
            )
            
            # Reorder columns based on linkage
            gene_order = hierarchy.leaves_list(gene_linkage)
            data = data.iloc[:, gene_order]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(data.values, aspect='auto', cmap=cmap)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        if zscore:
            cbar.set_label('Z-score')
        
        # Set labels
        ax.set_title(title)
        ax.set_xlabel('Features')
        ax.set_ylabel('Samples')
        
        # Set tick labels
        if data.shape[1] < 50:
            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_xticklabels(data.columns, rotation=90)
        else:
            # Too many features, show only a subset
            tick_step = max(1, data.shape[1] // 20)
            ax.set_xticks(np.arange(0, data.shape[1], tick_step))
            ax.set_xticklabels(data.columns[::tick_step], rotation=90)
            
        if data.shape[0] < 50:
            ax.set_yticks(np.arange(data.shape[0]))
            ax.set_yticklabels(data.index)
        else:
            # Too many samples, show only a subset
            tick_step = max(1, data.shape[0] // 20)
            ax.set_yticks(np.arange(0, data.shape[0], tick_step))
            ax.set_yticklabels(data.index[::tick_step])
        
        # Add cluster annotations as colored bars on y-axis
        ordered_labels = self.labels_[np.argsort(self.labels_)]
        unique_clusters = np.unique(ordered_labels)
        n_clusters = len(unique_clusters)
        
        # Create side axis for cluster annotation
        divider = make_axes_locatable(ax)
        ax_clusters = divider.append_axes("left", size="5%", pad=0.1)
        
        # Get colormap
        colormap = kwargs.get('cluster_cmap', 'tab10')
        if n_clusters > 10 and colormap == 'tab10':
            colormap = 'tab20'
            
        colors = plt.cm.get_cmap(colormap, max(10, n_clusters))
        
        # Create cluster annotation data
        cluster_data = np.zeros((data.shape[0], 1))
        for i, cluster in enumerate(unique_clusters):
            mask = (ordered_labels == cluster)
            cluster_data[mask, 0] = i
            
        # Plot cluster annotations
        ax_clusters.imshow(
            cluster_data, 
            aspect='auto', 
            cmap=colors,
            interpolation='nearest'
        )
        
        # Configure cluster annotation axis
        ax_clusters.set_title('Cluster')
        ax_clusters.set_xticks([])
        ax_clusters.set_yticks([])
        
        # Create custom legend for clusters
        cluster_handles = []
        for i, cluster in enumerate(unique_clusters):
            if cluster == -1:
                cluster_handles.append(plt.Line2D([0], [0], marker='s', color='white',
                                               markerfacecolor='black', markersize=10, label='Noise'))
            else:
                cluster_handles.append(plt.Line2D([0], [0], marker='s', color='white',
                                               markerfacecolor=colors(i % colors.N), markersize=10, 
                                               label=f'Cluster {cluster}'))
        
        # Add legend
        ax.legend(
            handles=cluster_handles, 
            loc='upper left', 
            bbox_to_anchor=(1.01, 1),
            title='Clusters'
        )
        
        plt.tight_layout()
        return fig


