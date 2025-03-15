import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Union, Optional, Tuple
import logging
import os

class BiologicalInterpreter:
    """
    Class for biological interpretation of selected features including pathway analysis
    and functional enrichment. Integrates with common bioinformatics resources.
    
    This class provides methods to:
    1. Perform Over-Representation Analysis (ORA)
    2. Run Gene Set Enrichment Analysis (GSEA)
    3. Map genes to pathways (KEGG, Reactome)
    4. Perform GO term enrichment
    5. Visualize enrichment results
    6. Generate pathway reports
    7. Create interactive network visualizations
    8. Export results for downstream analysis
    """
    
    def __init__(self, organism: str = "human", data_dir: str = "./data"):
        """
        Initialize the BiologicalInterpreter class
        
        Parameters:
        -----------
        organism: str
            Organism for pathway analysis ('human', 'mouse', etc.)
        data_dir: str
            Directory to store downloaded annotation files
        """
        self.organism = organism
        self.organism_id = self._get_organism_id(organism)
        self.data_dir = data_dir
        self.pathway_data = {}
        self.go_data = {}
        self.gene_mappings = {}
        self.logger = self._setup_logger()
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the class"""
        logger = logging.getLogger("BiologicalInterpreter")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _get_organism_id(self, organism: str) -> str:
        """
        Convert organism name to taxonomy ID
        
        Parameters:
        -----------
        organism: str
            Common name of organism
            
        Returns:
        --------
        str: NCBI taxonomy ID
        """
        organism_map = {
            "human": "9606",
            "mouse": "10090",
            "rat": "10116",
            "zebrafish": "7955",
            "fruitfly": "7227",
            "yeast": "4932",
            "c_elegans": "6239",
            "arabidopsis": "3702"
        }
        
        if organism.lower() in organism_map:
            return organism_map[organism.lower()]
        else:
            self.logger.warning(f"Organism {organism} not found in mapping. Using human as default.")
            return organism_map["human"]
    
    def load_gene_set(self, source: str, category: Optional[str] = None) -> Dict:
        """
        Load gene set data from various sources
        
        Parameters:
        -----------
        source: str
            Source of gene sets ('KEGG', 'GO', 'Reactome', 'DO')
        category: str, optional
            For GO, specify category ('BP', 'MF', 'CC')
            
        Returns:
        --------
        Dict: Dictionary of gene sets
        """
        source = source.upper()
        
        if source == "KEGG":
            return self._load_kegg_pathways()
        elif source == "GO":
            return self._load_go_terms(category)
        elif source == "REACTOME":
            return self._load_reactome_pathways()
        elif source == "DO":
            return self._load_disease_ontology()
        else:
            self.logger.error(f"Unknown gene set source: {source}")
            return {}
    
    def _load_kegg_pathways(self) -> Dict:
        """Load KEGG pathway data"""
        self.logger.info("Loading KEGG pathway data...")
        
        # Check if we already have the data cached
        kegg_file = os.path.join(self.data_dir, f"kegg_pathways_{self.organism}.csv")
        
        if os.path.exists(kegg_file):
            self.logger.info("Loading KEGG pathways from cache")
            pathways_df = pd.read_csv(kegg_file)
            pathways = pathways_df.groupby('pathway_id')['gene_id'].apply(list).to_dict()
        else:
            self.logger.info("Downloading KEGG pathways")
            # In a real implementation, we would use KEGG API or a package like bioservices
            # This is a simplified implementation
            
            # Mock download and processing:
            # 1. Get list of pathways for organism
            # 2. For each pathway, get genes
            
            # Storing example data for demonstration purposes
            pathways = {
                'path:hsa00010': ['3101', '3098', '3099', '3101', '2821'],  # Glycolysis
                'path:hsa00020': ['4967', '4968', '1737', '1738'],        # TCA cycle
                'path:hsa04010': ['1432', '5594', '5595', '6416']         # MAPK signaling
            }
            
            # Save to cache
            pathway_rows = []
            for pathway_id, genes in pathways.items():
                for gene in genes:
                    pathway_rows.append({'pathway_id': pathway_id, 'gene_id': gene})
            
            pathways_df = pd.DataFrame(pathway_rows)
            pathways_df.to_csv(kegg_file, index=False)
        
        self.pathway_data['KEGG'] = pathways
        return pathways
    
    def _load_go_terms(self, category: Optional[str] = None) -> Dict:
        """
        Load Gene Ontology terms
        
        Parameters:
        -----------
        category: str, optional
            GO category ('BP', 'MF', 'CC')
            
        Returns:
        --------
        Dict: Dictionary of GO terms and associated genes
        """
        self.logger.info(f"Loading GO terms for category: {category}")
        
        if category:
            category = category.upper()
            if category not in ['BP', 'MF', 'CC']:
                self.logger.warning(f"Invalid GO category: {category}. Using all categories.")
                category = None
        
        # Check if we already have the data cached
        go_file = os.path.join(self.data_dir, f"go_terms_{self.organism}_{category or 'all'}.csv")
        
        if os.path.exists(go_file):
            self.logger.info("Loading GO terms from cache")
            go_df = pd.read_csv(go_file)
            if category:
                go_df = go_df[go_df['category'] == category]
            go_terms = go_df.groupby('go_id')['gene_id'].apply(list).to_dict()
        else:
            self.logger.info("Downloading GO terms")
            # In a real implementation, we would use GO API or a package like goatools
            # This is a simplified implementation
            
            # Mock data for demonstration
            go_terms = {
                'GO:0006096': ['3101', '3098', '3099'],  # glycolytic process (BP)
                'GO:0016491': ['3098', '1738', '4967'],  # oxidoreductase activity (MF)
                'GO:0005739': ['4967', '4968', '1737']   # mitochondrion (CC)
            }
            
            # Save to cache
            go_rows = []
            for go_id, genes in go_terms.items():
                # In real implementation, we would also store the GO category
                cat = 'BP'  # Mock category assignment
                for gene in genes:
                    go_rows.append({'go_id': go_id, 'gene_id': gene, 'category': cat})
            
            go_df = pd.DataFrame(go_rows)
            go_df.to_csv(go_file, index=False)
        
        self.go_data[category or 'all'] = go_terms
        return go_terms
    
    def _load_reactome_pathways(self) -> Dict:
        """Load Reactome pathway data"""
        self.logger.info("Loading Reactome pathway data...")
        
        # Check if we already have the data cached
        reactome_file = os.path.join(self.data_dir, f"reactome_pathways_{self.organism}.csv")
        
        if os.path.exists(reactome_file):
            self.logger.info("Loading Reactome pathways from cache")
            pathways_df = pd.read_csv(reactome_file)
            pathways = pathways_df.groupby('pathway_id')['gene_id'].apply(list).to_dict()
        else:
            self.logger.info("Downloading Reactome pathways")
            # In a real implementation, we would use Reactome API
            # This is a simplified implementation
            
            # Mock data
            pathways = {
                'R-HSA-1640170': ['3101', '3098', '3099'],  # Cell Cycle
                'R-HSA-162582': ['4967', '4968', '1737'],   # Signal Transduction
                'R-HSA-73894': ['1432', '5594', '5595']     # DNA Repair
            }
            
            # Save to cache
            pathway_rows = []
            for pathway_id, genes in pathways.items():
                for gene in genes:
                    pathway_rows.append({'pathway_id': pathway_id, 'gene_id': gene})
            
            pathways_df = pd.DataFrame(pathway_rows)
            pathways_df.to_csv(reactome_file, index=False)
        
        self.pathway_data['REACTOME'] = pathways
        return pathways
    
    def _load_disease_ontology(self) -> Dict:
        """Load Disease Ontology data"""
        self.logger.info("Loading Disease Ontology data...")
        
        # Check if we already have the data cached
        do_file = os.path.join(self.data_dir, f"disease_ontology_{self.organism}.csv")
        
        if os.path.exists(do_file):
            self.logger.info("Loading Disease Ontology from cache")
            do_df = pd.read_csv(do_file)
            diseases = do_df.groupby('do_id')['gene_id'].apply(list).to_dict()
        else:
            self.logger.info("Downloading Disease Ontology")
            # In a real implementation, we would use DO API
            # This is a simplified implementation
            
            # Mock data
            diseases = {
                'DOID:9351': ['3101', '3098', '5594'],   # Diabetes
                'DOID:162': ['4967', '4968', '1737'],    # Cancer
                'DOID:934': ['1432', '5594', '5595']     # Viral infection
            }
            
            # Save to cache
            do_rows = []
            for do_id, genes in diseases.items():
                for gene in genes:
                    do_rows.append({'do_id': do_id, 'gene_id': gene})
            
            do_df = pd.DataFrame(do_rows)
            do_df.to_csv(do_file, index=False)
        
        self.pathway_data['DO'] = diseases
        return diseases
    
    def run_enrichment_analysis(self, 
                               gene_list: List[str], 
                               source: str, 
                               category: Optional[str] = None,
                               method: str = 'ora', 
                               background_size: Optional[int] = None,
                               pvalue_cutoff: float = 0.05) -> pd.DataFrame:
        """
        Run enrichment analysis on a list of genes
        
        Parameters:
        -----------
        gene_list: List[str]
            List of gene IDs to analyze
        source: str
            Source of gene sets ('KEGG', 'GO', 'REACTOME', 'DO')
        category: str, optional
            For GO, specify category ('BP', 'MF', 'CC')
        method: str
            Analysis method ('ora' for Over-Representation Analysis, 
                            'gsea' for Gene Set Enrichment Analysis)
        background_size: int, optional
            Number of genes in background (default is estimated)
        pvalue_cutoff: float
            P-value cutoff for significance
            
        Returns:
        --------
        pd.DataFrame: Enrichment results
        """
        source = source.upper()
        method = method.lower()
        
        # Load gene sets if not already loaded
        if source not in self.pathway_data and source != 'GO':
            self.load_gene_set(source)
        elif source == 'GO' and (not category or category not in self.go_data):
            self.load_gene_set(source, category)
        
        # Get appropriate gene set dictionary
        if source == 'GO':
            gene_sets = self.go_data.get(category or 'all', {})
        else:
            gene_sets = self.pathway_data.get(source, {})
        
        if not gene_sets:
            self.logger.error(f"No gene sets found for {source} {category or ''}")
            return pd.DataFrame()
        
        # Run appropriate enrichment method
        if method == 'ora':
            return self._run_ora(gene_list, gene_sets, background_size, pvalue_cutoff)
        elif method == 'gsea':
            self.logger.warning("GSEA requires gene scores. Please use run_gsea method instead.")
            return pd.DataFrame()
        else:
            self.logger.error(f"Unknown enrichment method: {method}")
            return pd.DataFrame()
    
    def _run_ora(self, 
                gene_list: List[str], 
                gene_sets: Dict[str, List[str]], 
                background_size: Optional[int] = None,
                pvalue_cutoff: float = 0.05) -> pd.DataFrame:
        """
        Run Over-Representation Analysis
        
        Parameters:
        -----------
        gene_list: List[str]
            List of gene IDs to analyze
        gene_sets: Dict[str, List[str]]
            Dictionary of gene sets
        background_size: int, optional
            Number of genes in background (default is estimated)
        pvalue_cutoff: float
            P-value cutoff for significance
            
        Returns:
        --------
        pd.DataFrame: ORA results
        """
        self.logger.info("Running Over-Representation Analysis...")
        
        # Convert gene list to set for faster lookups
        gene_set = set(gene_list)
        
        # Estimate background size if not provided
        if not background_size:
            # In a real implementation, this would depend on the platform/species
            background_size = 20000  # Approximate number of human protein-coding genes
        
        # Calculate enrichment for each gene set
        results = []
        
        for term_id, term_genes in gene_sets.items():
            # Number of genes in this term
            term_size = len(term_genes)
            
            # Number of input genes in this term
            overlap = len(gene_set.intersection(set(term_genes)))
            
            # Skip terms with no overlap
            if overlap == 0:
                continue
                
            # Calculate hypergeometric test
            # M = background_size (population size)
            # n = len(gene_list) (sample size)
            # N = term_size (successes in population)
            # k = overlap (successes in sample)
            
            # Calculate p-value using hypergeometric test
            # Subtract 1 because scipy.stats.hypergeom.sf is P(X > k)
            # and we want P(X >= k)
            pvalue = stats.hypergeom.sf(overlap-1, background_size, term_size, len(gene_list))
            
            # Calculate fold enrichment
            expected = (term_size / background_size) * len(gene_list)
            fold_enrichment = overlap / expected if expected > 0 else float('inf')
            
            results.append({
                'term_id': term_id,
                'p_value': pvalue,
                'fold_enrichment': fold_enrichment,
                'genes_in_term': term_size,
                'genes_overlap': overlap,
                'genes_in_list': len(gene_list),
                'background_size': background_size,
                'overlap_genes': ','.join(gene_set.intersection(set(term_genes)))
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            self.logger.warning("No enriched terms found.")
            return pd.DataFrame()
        
        # Calculate FDR (Benjamini-Hochberg)
        if not results_df.empty:
            results_df['fdr'] = self._calculate_fdr(results_df['p_value'])
            
            # Filter by FDR
            results_df = results_df[results_df['fdr'] <= pvalue_cutoff]
            
            # Sort by p-value
            results_df = results_df.sort_values('p_value')
        
        return results_df
    
    def _calculate_fdr(self, pvalues: pd.Series) -> pd.Series:
        """
        Calculate FDR using Benjamini-Hochberg procedure
        
        Parameters:
        -----------
        pvalues: pd.Series
            Series of p-values
            
        Returns:
        --------
        pd.Series: Adjusted p-values (FDR)
        """
        # Sort p-values
        sorted_pvalues = pvalues.sort_values()
        
        # Get ranks, adjusting for ties
        ranks = np.arange(1, len(sorted_pvalues) + 1)
        
        # Calculate FDR
        fdr = sorted_pvalues * len(sorted_pvalues) / ranks
        
        # Ensure FDR is monotonic
        for i in range(len(fdr) - 1, 0, -1):
            fdr.iloc[i-1] = min(fdr.iloc[i-1], fdr.iloc[i])
        
        # Cap at 1
        fdr = fdr.clip(upper=1)
        
        # Return to original order
        return fdr.reindex(pvalues.index)
    
    def run_gsea(self, 
                gene_scores: Dict[str, float], 
                source: str, 
                category: Optional[str] = None,
                permutations: int = 1000,
                min_set_size: int = 15,
                max_set_size: int = 500,
                pvalue_cutoff: float = 0.05) -> pd.DataFrame:
        """
        Run Gene Set Enrichment Analysis (GSEA)
        
        Parameters:
        -----------
        gene_scores: Dict[str, float]
            Dictionary mapping gene IDs to scores (e.g., log fold change)
        source: str
            Source of gene sets ('KEGG', 'GO', 'REACTOME', 'DO')
        category: str, optional
            For GO, specify category ('BP', 'MF', 'CC')
        permutations: int
            Number of permutations for statistical testing
        min_set_size: int
            Minimum size of gene sets to consider
        max_set_size: int
            Maximum size of gene sets to consider
        pvalue_cutoff: float
            P-value cutoff for significance
            
        Returns:
        --------
        pd.DataFrame: GSEA results
        """
        self.logger.info("Running Gene Set Enrichment Analysis...")
        
        source = source.upper()
        
        # Load gene sets if not already loaded
        if source not in self.pathway_data and source != 'GO':
            self.load_gene_set(source)
        elif source == 'GO' and (not category or category not in self.go_data):
            self.load_gene_set(source, category)
        
        # Get appropriate gene set dictionary
        if source == 'GO':
            gene_sets = self.go_data.get(category or 'all', {})
        else:
            gene_sets = self.pathway_data.get(source, {})
        
        if not gene_sets:
            self.logger.error(f"No gene sets found for {source} {category or ''}")
            return pd.DataFrame()
        
        # Sort gene scores from highest to lowest
        sorted_genes = pd.Series(gene_scores).sort_values(ascending=False)
        
        # Create rank metric
        rank_metric = sorted_genes.values
        gene_list = sorted_genes.index.tolist()
        
        # Run GSEA for each gene set
        results = []
        
        for term_id, term_genes in gene_sets.items():
            # Filter term genes that are in our gene list
            term_genes_in_list = [g for g in term_genes if g in gene_scores]
            
            # Skip small or large gene sets
            if len(term_genes_in_list) < min_set_size or len(term_genes_in_list) > max_set_size:
                continue
            
            # Calculate ES and NES
            es, nes, pvalue, leading_edge = self._calculate_gsea_stats(
                gene_list, term_genes_in_list, rank_metric, permutations)
            
            results.append({
                'term_id': term_id,
                'es': es,
                'nes': nes,
                'p_value': pvalue,
                'genes_in_term': len(term_genes),
                'genes_in_list': len(term_genes_in_list),
                'leading_edge': ','.join(leading_edge)
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            self.logger.warning("No enriched terms found.")
            return pd.DataFrame()
        
        # Calculate FDR
        if not results_df.empty:
            # Separate positive and negative ES
            pos_df = results_df[results_df['es'] > 0]
            neg_df = results_df[results_df['es'] < 0]
            
            # Calculate FDR separately
            if not pos_df.empty:
                pos_df['fdr'] = self._calculate_fdr(pos_df['p_value'])
            if not neg_df.empty:
                neg_df['fdr'] = self._calculate_fdr(neg_df['p_value'])
            
            # Combine and filter
            results_df = pd.concat([pos_df, neg_df])
            results_df = results_df[results_df['fdr'] <= pvalue_cutoff]
            
            # Sort by absolute NES
            results_df['abs_nes'] = results_df['nes'].abs()
            results_df = results_df.sort_values('abs_nes', ascending=False)
            results_df = results_df.drop('abs_nes', axis=1)
        
        return results_df
    
    def _calculate_gsea_stats(self, 
                            gene_list: List[str], 
                            term_genes: List[str], 
                            rank_metric: np.ndarray, 
                            permutations: int) -> Tuple[float, float, float, List[str]]:
        """
        Calculate GSEA statistics for a gene set
        
        Parameters:
        -----------
        gene_list: List[str]
            Ordered list of genes (from highest to lowest score)
        term_genes: List[str]
            Genes in the term
        rank_metric: np.ndarray
            Array of gene scores
        permutations: int
            Number of permutations
            
        Returns:
        --------
        Tuple[float, float, float, List[str]]: 
            ES, NES, p-value, and leading edge genes
        """
        # This is a simplified GSEA implementation
        # In a real application, you would use a more robust implementation
        # like gseapy or GSEApy
        
        # Calculate hit indices
        hit_indices = [i for i, g in enumerate(gene_list) if g in term_genes]
        
        # Calculate enrichment score
        N = len(gene_list)
        Nh = len(hit_indices)
        
        # Skip if no hits
        if Nh == 0:
            return 0, 0, 1.0, []
        
        # Calculate enrichment score
        running_sum = 0
        max_deviation = 0
        min_deviation = 0
        max_position = 0
        
        # Constants for weight calculation
        p = 1  # Weighted
        
        # Calculate running sum
        hit_indicator = np.zeros(N)
        hit_indicator[hit_indices] = 1
        
        # Compute weighted score
        weights = np.zeros(N)
        weights[hit_indices] = np.abs(rank_metric[hit_indices])**p
        weights[hit_indices] = weights[hit_indices] / np.sum(weights[hit_indices])
        
        # Compute penalty
        penalty = 1 / (N - Nh)
        
        # Calculate running sum
        running_sums = np.zeros(N)
        for i in range(N):
            if hit_indicator[i] == 1:
                running_sum += weights[i]
            else:
                running_sum -= penalty
            
            running_sums[i] = running_sum
            
            if running_sum > max_deviation:
                max_deviation = running_sum
                max_position = i
            if running_sum < min_deviation:
                min_deviation = running_sum
        
        # ES is the maximum deviation from zero
        es = max_deviation if abs(max_deviation) > abs(min_deviation) else min_deviation
        
        # Calculate leading edge
        if es > 0:
            leading_edge_idx = hit_indices[:hit_indices.index(max_position) + 1]
        else:
            leading_edge_idx = hit_indices[hit_indices.index(max_position):]
        
        leading_edge = [gene_list[i] for i in leading_edge_idx]
        
        # Calculate NES and p-value through permutation testing
        # This is a simplified approach
        perm_es = []
        for _ in range(permutations):
            # Permute gene ranks
            perm_ranks = np.random.permutation(rank_metric)
            
            # Calculate ES for permutation
            perm_running_sum = 0
            perm_max_dev = 0
            perm_min_dev = 0
            
            perm_weights = np.zeros(N)
            perm_weights[hit_indices] = np.abs(perm_ranks[hit_indices])**p
            perm_weights[hit_indices] = perm_weights[hit_indices] / np.sum(perm_weights[hit_indices])
            
            for i in range(N):
                if hit_indicator[i] == 1:
                    perm_running_sum += perm_weights[i]
                else:
                    perm_running_sum -= penalty
                
                if perm_running_sum > perm_max_dev:
                    perm_max_dev = perm_running_sum
                if perm_running_sum < perm_min_dev:
                    perm_min_dev = perm_running_sum
            
            perm_es.append(perm_max_dev if abs(perm_max_dev) > abs(perm_min_dev) else perm_min_dev)
        
        # Calculate NES
        mean_perm_es = np.mean(perm_es)
        std_perm_es = np.std(perm_es)
        
        nes = es / mean_perm_es if mean_perm_es != 0 else 0
        
        # Calculate p-value
        if es >= 0:
            pvalue = np.sum(np.array(perm_es) >= es) / permutations
        else:
            pvalue = np.sum(np.array(perm_es) <= es) / permutations
        
        # Ensure p-value is not zero (for FDR calculation)
        pvalue = max(pvalue, 1/permutations)
        
        return es, nes, pvalue, leading_edge
    
    def visualize_enrichment(self, 
                           results: pd.DataFrame, 
                           n_terms: int = 20,
                           save_path: Optional[str] = None,
                           plot_type: str = 'barplot') -> plt.Figure:
        """
        Visualize enrichment results
        
        Parameters:
        -----------
        results: pd.DataFrame
            Enrichment results from run_enrichment_analysis or run_gsea
        n_terms: int
            Number of top terms to include
        save_path: str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure: Matplotlib figure
        """
        if results.empty:
            self.logger.warning("No results to visualize.")
            return None
        
        # Check if it's ORA or GSEA results
        is_gsea = 'nes' in results.columns
        
        # Take top N terms
        results = results.head(n_terms)
        
        # Set up figure
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'barplot':
            if is_gsea:
                # For GSEA, plot NES
                ax = sns.barplot(x='nes', y='term_id', data=results,
                                palette=sns.diverging_palette(240, 10, s=80, l=50,
                                                            n=len(results),
                                                            center="dark"))
                plt.xlabel('Normalized Enrichment Score (NES)')
                plt.title('Gene Set Enrichment Analysis Results')
                
                # Add FDR as text
                for i, row in enumerate(results.itertuples()):
                    plt.text(row.nes + (0.1 if row.nes >= 0 else -0.1),
                            i, f'FDR={row.fdr:.2e}',
                            va='center', ha='left' if row.nes >= 0 else 'right')
            else:
                # For ORA, plot -log10(p-value)
                results = results.copy()
                results['-log10(p)'] = -np.log10(results['p_value'])
                
                ax = sns.barplot(x='-log10(p)', y='term_id', data=results, color='steelblue')
                plt.xlabel('-log10(p-value)')
                plt.title('Over-Representation Analysis Results')
                
                # Add fold enrichment as text
                for i, row in enumerate(results.itertuples()):
                    plt.text(row.__getattribute__('-log10(p)') + 0.1,
                            i, f'Fold={row.fold_enrichment:.1f}',
                            va='center')
        
        elif plot_type == 'bubble':
            # Create bubble plot for enrichment
            plt.figure(figsize=(12, 10))
            
            # Prepare data
            if is_gsea:
                x = results['nes']
                size_col = 'genes_in_list'
                color_col = 'p_value'
                title = 'Gene Set Enrichment Analysis Results'
                xlabel = 'Normalized Enrichment Score (NES)'
            else:
                results = results.copy()
                results['-log10(p)'] = -np.log10(results['p_value'])
                x = results['-log10(p)']
                size_col = 'fold_enrichment'
                color_col = 'p_value'
                title = 'Over-Representation Analysis Results'
                xlabel = '-log10(p-value)'
            
            # Scale sizes for better visualization
            sizes = results[size_col] * 10
            
            # Create scatter plot
            scatter = plt.scatter(
                x=x,
                y=results.index,
                s=sizes,
                c=-np.log10(results[color_col]),
                cmap='viridis',
                alpha=0.7
            )
            
            # Add labels
            plt.yticks(results.index, results['term_id'])
            plt.xlabel(xlabel)
            plt.title(title)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('-log10(p-value)')
            
        elif plot_type == 'heatmap':
            # Extract genes from overlaps
            if is_gsea:
                # For GSEA, use leading edge genes
                genes_by_term = results.apply(
                    lambda row: set(row['leading_edge'].split(',') if isinstance(row['leading_edge'], str) else []), 
                    axis=1
                )
            else:
                # For ORA, use overlap genes
                genes_by_term = results.apply(
                    lambda row: set(row['overlap_genes'].split(',') if isinstance(row['overlap_genes'], str) else []), 
                    axis=1
                )
            
            # Get unique genes across all terms
            all_genes = set()
            for genes in genes_by_term:
                all_genes.update(genes)
                
            # Create matrix for heatmap
            heatmap_data = np.zeros((len(results), len(all_genes)))
            all_genes_list = list(all_genes)
            
            # Fill matrix
            for i, genes in enumerate(genes_by_term):
                for j, gene in enumerate(all_genes_list):
                    heatmap_data[i, j] = 1 if gene in genes else 0
            
            # Create heatmap
            plt.figure(figsize=(max(12, len(all_genes) * 0.3), len(results) * 0.4))
            sns.heatmap(
                heatmap_data,
                cmap='Blues',
                yticklabels=results['term_id'],
                xticklabels=all_genes_list,
                cbar=False
            )
            
            plt.title('Gene-Term Associations')
            plt.xlabel('Genes')
            plt.xticks(rotation=90)
            plt.tight_layout()
        
        plt.ylabel('')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
        
    def generate_pathway_report(self, 
                              results: pd.DataFrame,
                              source: str,
                              output_dir: str = "./reports",
                              include_genes: bool = True,
                              include_plots: bool = True) -> str:
        """
        Generate an HTML report for pathway enrichment results
        
        Parameters:
        -----------
        results: pd.DataFrame
            Enrichment results from run_enrichment_analysis or run_gsea
        source: str
            Source of gene sets ('KEGG', 'GO', 'REACTOME', 'DO')
        output_dir: str
            Directory to save the report
        include_genes: bool
            Whether to include overlapping genes in the report
        include_plots: bool
            Whether to include visualizations in the report
            
        Returns:
        --------
        str: Path to the generated report
        """
        if results.empty:
            self.logger.warning("No results to generate report.")
            return ""
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Determine if GSEA or ORA results
        is_gsea = 'nes' in results.columns
        
        # Generate report filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        method = "gsea" if is_gsea else "ora"
        report_name = f"{source.lower()}_{method}_report_{timestamp}.html"
        report_path = os.path.join(output_dir, report_name)
        
        # Generate plots if requested
        plot_paths = []
        if include_plots:
            plots_dir = os.path.join(output_dir, "plots")
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            
            # Generate barplot
            barplot_path = os.path.join(plots_dir, f"{source.lower()}_{method}_barplot_{timestamp}.png")
            self.visualize_enrichment(results, plot_type='barplot', save_path=barplot_path)
            plot_paths.append(("Barplot", barplot_path))
            
            # Generate bubble plot
            bubble_path = os.path.join(plots_dir, f"{source.lower()}_{method}_bubble_{timestamp}.png")
            self.visualize_enrichment(results, plot_type='bubble', save_path=bubble_path)
            plot_paths.append(("Bubble Plot", bubble_path))
            
            # Generate heatmap
            heatmap_path = os.path.join(plots_dir, f"{source.lower()}_{method}_heatmap_{timestamp}.png")
            self.visualize_enrichment(results, plot_type='heatmap', save_path=heatmap_path)
            plot_paths.append(("Heatmap", heatmap_path))
        
        # Generate HTML report
        with open(report_path, 'w') as f:
            # Write header
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>{source} {method.upper()} Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .plot {{ margin: 20px 0; text-align: center; }}
        .plot img {{ max-width: 100%; }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>{source} {method.upper()} Enrichment Analysis Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>Analysis performed: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Method: {method.upper()}</p>
        <p>Database: {source}</p>
        <p>Number of enriched terms: {len(results)}</p>
    </div>
""")
            
            # Add plots if included
            if include_plots:
                f.write('<h2>Visualizations</h2>\n')
                for plot_title, plot_path in plot_paths:
                    rel_path = os.path.relpath(plot_path, output_dir)
                    f.write(f'<div class="plot">\n')
                    f.write(f'<h3>{plot_title}</h3>\n')
                    f.write(f'<img src="{rel_path}" alt="{plot_title}">\n')
                    f.write('</div>\n')
            
            # Add results table
            f.write('<h2>Enriched Terms</h2>\n')
            f.write('<table>\n')
            
            # Write table header
            if is_gsea:
                f.write('<tr><th>Term ID</th><th>ES</th><th>NES</th><th>P-value</th><th>FDR</th><th>Genes in Term</th>')
            else:
                f.write('<tr><th>Term ID</th><th>P-value</th><th>FDR</th><th>Fold Enrichment</th><th>Genes in Term</th><th>Genes Overlap</th>')
            
            if include_genes:
                f.write('<th>Genes</th>')
            
            f.write('</tr>\n')
            
            # Write table rows
            for _, row in results.iterrows():
                f.write('<tr>')
                
                if is_gsea:
                    f.write(f'<td>{row["term_id"]}</td>')
                    f.write(f'<td>{row["es"]:.4f}</td>')
                    f.write(f'<td>{row["nes"]:.4f}</td>')
                    f.write(f'<td>{row["p_value"]:.4e}</td>')
                    f.write(f'<td>{row["fdr"]:.4e}</td>')
                    f.write(f'<td>{row["genes_in_term"]}</td>')
                    
                    if include_genes:
                        leading_edge = row.get("leading_edge", "")
                        f.write(f'<td>{leading_edge}</td>')
                else:
                    f.write(f'<td>{row["term_id"]}</td>')
                    f.write(f'<td>{row["p_value"]:.4e}</td>')
                    f.write(f'<td>{row["fdr"]:.4e}</td>')
                    f.write(f'<td>{row["fold_enrichment"]:.2f}</td>')
                    f.write(f'<td>{row["genes_in_term"]}</td>')
                    f.write(f'<td>{row["genes_overlap"]}</td>')
                    
                    if include_genes:
                        overlap_genes = row.get("overlap_genes", "")
                        f.write(f'<td>{overlap_genes}</td>')
                
                f.write('</tr>\n')
            
            f.write('</table>\n')
            
            # Write footer
            f.write("""
    <div style="margin-top: 30px; font-size: 0.8em; color: #666; text-align: center;">
        <p>Generated using BiologicalInterpreter</p>
    </div>
</body>
</html>
""")
        
        self.logger.info(f"Report generated: {report_path}")
        return report_path
    
    def create_network_visualization(self,
                                   results: pd.DataFrame,
                                   n_terms: int = 10,
                                   include_genes: bool = True,
                                   min_edge_weight: float = 0.3,
                                   output_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Create a network visualization of enrichment results
        
        Parameters:
        -----------
        results: pd.DataFrame
            Enrichment results from run_enrichment_analysis or run_gsea
        n_terms: int
            Number of top terms to include
        include_genes: bool
            Whether to include genes in the network
        min_edge_weight: float
            Minimum edge weight to include in network
        output_path: str, optional
            Path to save the visualization
            
        Returns:
        --------
        Optional[plt.Figure]: Matplotlib figure if available
        """
        if results.empty:
            self.logger.warning("No results to visualize.")
            return None
        
        # Check if networkx is available
        try:
            import networkx as nx
        except ImportError:
            self.logger.warning("NetworkX is not available. Please install it: pip install networkx")
            return None
        
        # Determine if GSEA or ORA results
        is_gsea = 'nes' in results.columns
        
        # Take top N terms
        results = results.head(n_terms)
        
        # Create network graph
        G = nx.Graph()
        
        # Add term nodes
        for i, row in results.iterrows():
            term_id = row['term_id']
            
            # Node size based on significance
            if is_gsea:
                size = abs(row['nes']) * 100
                color = 'red' if row['nes'] > 0 else 'blue'
            else:
                size = -np.log10(row['p_value']) * 20
                color = 'green'
            
            G.add_node(term_id, type='term', size=size, color=color)
            
            # Add gene nodes and edges if requested
            if include_genes:
                if is_gsea and 'leading_edge' in row and row['leading_edge']:
                    genes = row['leading_edge'].split(',')
                elif 'overlap_genes' in row and row['overlap_genes']:
                    genes = row['overlap_genes'].split(',')
                else:
                    genes = []
                
                for gene in genes:
                    if not gene:
                        continue
                        
                    # Add gene node if not already in the graph
                    if gene not in G:
                        G.add_node(gene, type='gene', size=50, color='gray')
                    
                    # Add edge between term and gene
                    # Edge weight based on significance
                    if is_gsea:
                        weight = abs(row['nes']) / 4
                    else:
                        weight = -np.log10(row['p_value']) / 10
                    
                    if weight >= min_edge_weight:
                        G.add_edge(term_id, gene, weight=weight)
        
        # Calculate term-term edges based on shared genes
        if include_genes:
            # Get all terms
            terms = list(nx.get_node_attributes(G, 'type').keys())
            terms = [t for t in terms if G.nodes[t]['type'] == 'term']
            
            # Calculate shared genes between terms
            for i, term1 in enumerate(terms):
                term1_neighbors = set(nx.neighbors(G, term1))
                term1_genes = [n for n in term1_neighbors if G.nodes[n]['type'] == 'gene']
                
                for j in range(i+1, len(terms)):
                    term2 = terms[j]
                    term2_neighbors = set(nx.neighbors(G, term2))
                    term2_genes = [n for n in term2_neighbors if G.nodes[n]['type'] == 'gene']
                    
                    # Calculate Jaccard similarity between gene sets
                    shared_genes = set(term1_genes).intersection(set(term2_genes))
                    
                    if shared_genes:
                        jaccard = len(shared_genes) / len(set(term1_genes).union(set(term2_genes)))
                        
                        # Add edge if similarity is above threshold
                        if jaccard >= min_edge_weight:
                            G.add_edge(term1, term2, weight=jaccard, style='dashed')
        
        # Draw the network
        plt.figure(figsize=(12, 10))
        
        # Get node positions using spring layout
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Draw term nodes
        term_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'term']
        term_sizes = [G.nodes[n]['size'] for n in term_nodes]
        term_colors = [G.nodes[n]['color'] for n in term_nodes]
        
        nx.draw_networkx_nodes(G, pos, nodelist=term_nodes, 
                            node_size=term_sizes, 
                            node_color=term_colors, 
                            alpha=0.8)
        
        # Draw gene nodes if included
        if include_genes:
            gene_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'gene']
            
            if gene_nodes:
                gene_sizes = [G.nodes[n]['size'] for n in gene_nodes]
                gene_colors = [G.nodes[n]['color'] for n in gene_nodes]
                
                nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, 
                                    node_size=gene_sizes, 
                                    node_color=gene_colors, 
                                    alpha=0.6)
        
        # Draw edges
        edges = G.edges(data=True)
        
        # Scale edge widths based on weight
        edge_widths = [e[2]['weight'] * 2 for e in edges]
        
        # Edge styles
        edge_styles = [e[2].get('style', 'solid') for e in edges]
        solid_edges = [(e[0], e[1]) for e, style in zip(edges, edge_styles) if style == 'solid']
        dashed_edges = [(e[0], e[1]) for e, style in zip(edges, edge_styles) if style == 'dashed']
        
        # Draw solid edges
        if solid_edges:
            solid_widths = [G[u][v]['weight'] * 2 for u, v in solid_edges]
            nx.draw_networkx_edges(G, pos, edgelist=solid_edges, 
                                width=solid_widths, 
                                alpha=0.6)
        
        # Draw dashed edges
        if dashed_edges:
            dashed_widths = [G[u][v]['weight'] * 2 for u, v in dashed_edges]
            nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, 
                                width=dashed_widths,
                                style='dashed',
                                alpha=0.4)
        
        # Add labels
        term_labels = {n: n for n in term_nodes}
        nx.draw_networkx_labels(G, pos, labels=term_labels, font_size=8)
        
        if include_genes:
            gene_labels = {n: n for n in gene_nodes}
            nx.draw_networkx_labels(G, pos, labels=gene_labels, font_size=6)
        
        plt.title('Term-Gene Network')
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def export_results(self, 
                     results: pd.DataFrame, 
                     output_path: str,
                     format: str = 'csv') -> str:
        """
        Export enrichment results to file
        
        Parameters:
        -----------
        results: pd.DataFrame
            Enrichment results from run_enrichment_analysis or run_gsea
        output_path: str
            Path to save the results
        format: str
            Export format ('csv', 'tsv', 'excel', 'json')
            
        Returns:
        --------
        str: Path to the exported file
        """
        if results.empty:
            self.logger.warning("No results to export.")
            return ""
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Export in the specified format
        format = format.lower()
        
        if format == 'csv':
            results.to_csv(output_path, index=False)
        elif format == 'tsv':
            results.to_csv(output_path, sep='\t', index=False)
        elif format == 'excel':
            results.to_excel(output_path, index=False)
        elif format == 'json':
            results.to_json(output_path, orient='records')
        else:
            self.logger.warning(f"Unknown format: {format}. Exporting as CSV.")
            results.to_csv(output_path, index=False)
        
        self.logger.info(f"Results exported to {output_path}")
        return output_path


