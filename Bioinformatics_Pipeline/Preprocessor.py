import numpy as np
import pandas as pd
from scipy import stats
import logging

class BioinformaticsPreprocessor:
    """
    Step 1: Preprocessing & Normalization
    
    A class for preprocessing and normalizing various types of bioinformatics data matrices.
    Converts raw values (counts, peak intensities, abundances, quantitative trait measurements) 
    into normalized values suitable for ML/DL models.
    """
    
    def __init__(self, data_type=None, logger=None):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        data_type : str
            Type of bioinformatics data ('rna_seq', 'chip_seq', 'metagenomics', 'wgas_eqtl')
        logger : logging.Logger
            Logger for tracking the preprocessing steps
        """
        self.data_type = data_type
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
            
    def _setup_logger(self):
        """Setup a basic logger if none is provided."""
        logger = logging.getLogger("BioinformaticsPreprocessor")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
            
    def preprocess(self, data, method=None, **kwargs):
        """
        Preprocess and normalize the input data based on the data type.
        
        Parameters:
        -----------
        data : pandas.DataFrame or numpy.ndarray
            The raw data to be preprocessed
        method : str
            Specific normalization method to use. If None, a default method 
            will be selected based on the data type.
        **kwargs : 
            Additional parameters specific to the normalization method
            
        Returns:
        --------
        pandas.DataFrame or numpy.ndarray
            The normalized data
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
            
        self.logger.info(f"Starting preprocessing with data type: {self.data_type}, method: {method}")
        
        if self.data_type == 'rna_seq':
            normalized_data = self._normalize_rna_seq(data, method, **kwargs)
        elif self.data_type == 'chip_seq':
            normalized_data = self._normalize_chip_seq(data, method, **kwargs)
        elif self.data_type == 'metagenomics':
            normalized_data = self._normalize_metagenomics(data, method, **kwargs)
        elif self.data_type == 'wgas_eqtl':
            normalized_data = self._normalize_wgas_eqtl(data, method, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}. Please specify a valid data type.")
            
        self.logger.info("Preprocessing completed successfully")
        return normalized_data
    
    def _normalize_rna_seq(self, data, method=None, **kwargs):
        """
        Normalize RNA-seq data.
        
        Methods:
        - 'log': Simple log2 transformation with pseudocount
        - 'vst': Variance stabilizing transformation (approximation)
        - 'tpm': Transcripts Per Million normalization
        - 'deseq': DESeq2-like size factor normalization (simplified)
        - 'tmm': Trimmed Mean of M-values normalization (simplified)
        """
        if method is None:
            method = 'log'  # Default method
            
        self.logger.info(f"Normalizing RNA-seq data using {method} method")
            
        if method == 'log':
            # Log2 transformation with pseudocount
            pseudocount = kwargs.get('pseudocount', 1)
            return pd.DataFrame(np.log2(data + pseudocount), index=data.index, columns=data.columns)
            
        elif method == 'vst':
            # Simplified variance stabilizing transformation
            return pd.DataFrame(np.sqrt(data), index=data.index, columns=data.columns)
            
        elif method == 'tpm':
            # Simple TPM normalization
            gene_lengths = kwargs.get('gene_lengths')
            if gene_lengths is None:
                raise ValueError("Gene lengths must be provided for TPM normalization")
                
            # TPM calculation
            rpk = data.div(gene_lengths, axis=0) * 1000
            scaling_factors = rpk.sum(axis=0) / 1e6
            tpm = rpk.div(scaling_factors, axis=1)
            return tpm
            
        elif method == 'deseq':
            # Simplified DESeq2 normalization
            # Calculate size factors using median ratio method
            geometric_means = stats.gmean(data.values, axis=1)
            size_factors = data.div(geometric_means, axis=0).median(axis=0)
            normalized = data.div(size_factors, axis=1)
            return normalized
            
        elif method == 'tmm':
            # Simplified TMM normalization
            lib_sizes = data.sum(axis=0)
            normalized = data.div(lib_sizes, axis=1) * 1e6  # Simplification to CPM
            return normalized
            
        else:
            raise ValueError(f"Unsupported RNA-seq normalization method: {method}")
    
    def _normalize_chip_seq(self, data, method=None, **kwargs):
        """
        Normalize ChIP-seq data.
        
        Methods:
        - 'rpm': Reads Per Million normalization
        - 'snr': Signal-to-noise ratio normalization
        """
        if method is None:
            method = 'rpm'  # Default method
            
        self.logger.info(f"Normalizing ChIP-seq data using {method} method")
            
        if method == 'rpm':
            # Reads per million normalization
            lib_sizes = data.sum(axis=0)
            rpm = data.div(lib_sizes, axis=1) * 1e6
            return rpm
            
        elif method == 'snr':
            # Signal-to-noise ratio normalization
            control_samples = kwargs.get('control_samples')
            if control_samples is None:
                raise ValueError("Control samples must be provided for SNR normalization")
                
            # Extract control data
            control_data = data[control_samples]
            treatment_data = data.drop(columns=control_samples)
            
            # Calculate mean of control for each feature
            control_means = control_data.mean(axis=1)
            
            # Calculate signal-to-noise ratio
            snr = treatment_data.div(control_means, axis=0)
            return snr
            
        else:
            raise ValueError(f"Unsupported ChIP-seq normalization method: {method}")
    
    def _normalize_metagenomics(self, data, method=None, **kwargs):
        """
        Normalize metagenomic data.
        
        Methods:
        - 'relative': Relative abundance normalization
        - 'clr': Centered log-ratio normalization
        """
        if method is None:
            method = 'relative'  # Default method
            
        self.logger.info(f"Normalizing metagenomic data using {method} method")
            
        if method == 'relative':
            # Relative abundance normalization
            sample_totals = data.sum(axis=0)
            relative_abundance = data.div(sample_totals, axis=1)
            return relative_abundance
            
        elif method == 'clr':
            # Centered log-ratio normalization
            pseudocount = kwargs.get('pseudocount', 1e-6)
            
            # Add pseudocount and calculate geometric mean for each sample
            data_with_pseudocount = data + pseudocount
            
            # Calculate CLR
            clr = data_with_pseudocount.apply(lambda x: np.log(x / stats.gmean(x)), axis=0)
            return clr
            
        else:
            raise ValueError(f"Unsupported metagenomic normalization method: {method}")
    
    def _normalize_wgas_eqtl(self, data, method=None, **kwargs):
        """
        Normalize WGAS or eQTL data.
        
        Methods:
        - 'standardize': Z-score standardization
        - 'rank': Rank-based normalization
        """
        if method is None:
            method = 'standardize'  # Default method
            
        self.logger.info(f"Normalizing WGAS/eQTL data using {method} method")
            
        if method == 'standardize':
            # Z-score standardization
            centered = data.sub(data.mean(axis=0), axis=1)
            standardized = centered.div(data.std(axis=0), axis=1)
            return standardized
            
        elif method == 'rank':
            # Rank-based normalization
            ranked = data.rank(axis=0)
            total = data.shape[0]
            
            # Scale to [0,1]
            normalized = (ranked - 0.5) / total
            
            # Optional inverse normal transformation
            if kwargs.get('inverse_normal', False):
                from scipy.stats import norm
                normalized = normalized.apply(lambda x: norm.ppf(x))
                
            return normalized
            
        else:
            raise ValueError(f"Unsupported WGAS/eQTL normalization method: {method}")
            
    def fit_transform(self, data, method=None, **kwargs):
        """
        Alias for preprocess() to maintain API compatibility with scikit-learn.
        """
        return self.preprocess(data, method, **kwargs)
        
    def transform(self, data, method=None, **kwargs):
        """
        Alias for preprocess() to maintain API compatibility with scikit-learn.
        """
        return self.preprocess(data, method, **kwargs)


