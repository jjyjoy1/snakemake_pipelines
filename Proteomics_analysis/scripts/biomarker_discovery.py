#!/usr/bin/env python3
"""
Biomarker Discovery and Machine Learning for Proteomics Data

This script:
1. Processes proteomics quantification data
2. Performs feature selection to identify potential biomarkers
3. Builds and evaluates various machine learning models
4. Generates visualizations and interpretability analyses
5. Creates a comprehensive report of findings

Author: Jiyang Jiang
Date: March 11, 2025
"""

import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set, Optional, Union
import json
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import io
import base64
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(snakemake.log[0]),
        logging.StreamHandler()
    ]
)

class BiomarkerDiscovery:
    """Class for biomarker discovery and machine learning analysis"""
    
    def __init__(self, random_state=42):
        """
        Initialize the biomarker discovery tool
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility (default: 42)
        """
        self.random_state = random_state
        self.data = None
        self.features = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.feature_importance = {}
        self.selected_features = []
        self.best_model = None
        
    def load_data(self, quantification_file, metadata_file, sample_col="sample_id", group_col="condition"):
        """
        Load and prepare proteomics data for analysis
        
        Parameters:
        -----------
        quantification_file : str
            Path to protein quantification file
        metadata_file : str
            Path to sample metadata file with condition information
        sample_col : str
            Column name for sample IDs in metadata (default: "sample_id")
        group_col : str
            Column name for experimental conditions in metadata (default: "condition")
            
        Returns:
        --------
        pandas.DataFrame
            Processed data frame with samples, features, and target variable
        """
        logging.info(f"Loading proteomics data from {quantification_file}")
        
        try:
            # Load quantification data
            quant_data = pd.read_csv(quantification_file, sep='\t', index_col=0)
            
            # Load metadata
            metadata = pd.read_csv(metadata_file, sep='\t')
            
            # Check if required columns exist in metadata
            if sample_col not in metadata.columns:
                raise ValueError(f"Sample column '{sample_col}' not found in metadata")
            if group_col not in metadata.columns:
                raise ValueError(f"Group column '{group_col}' not found in metadata")
            
            # Check if all samples in quantification data are in metadata
            quant_samples = quant_data.columns
            metadata_samples = set(metadata[sample_col])
            
            missing_samples = [s for s in quant_samples if s not in metadata_samples]
            if missing_samples:
                logging.warning(f"The following samples are missing from metadata: {missing_samples}")
                # Filter out missing samples
                quant_data = quant_data[[s for s in quant_samples if s in metadata_samples]]
            
            # Create a mapping from sample to group
            sample_to_group = dict(zip(metadata[sample_col], metadata[group_col]))
            
            # Transpose quantification data to have samples as rows and proteins as columns
            quant_data_t = quant_data.T
            
            # Add group information
            quant_data_t['group'] = quant_data_t.index.map(lambda x: sample_to_group.get(x, None))
            
            # Remove samples with missing group information
            if quant_data_t['group'].isna().any():
                logging.warning("Removing samples with missing group information")
                quant_data_t = quant_data_t.dropna(subset=['group'])
            
            logging.info(f"Loaded data with {quant_data_t.shape[0]} samples and {quant_data_t.shape[1]-1} proteins")
            
            # Store processed data
            self.data = quant_data_t
            self.features = quant_data_t.columns[:-1]  # All columns except 'group'
            self.target = 'group'
            
            return self.data
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, test_size=0.2, impute_method='knn', scale_method='robust'):
        """
        Preprocess data for machine learning
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing (default: 0.2)
        impute_method : str
            Method for imputing missing values ('knn', 'mean', 'median', 'none')
        scale_method : str
            Method for scaling features ('standard', 'robust', 'none')
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        logging.info("Preprocessing data for ML analysis")
        
        if self.data is None:
            logging.error("No data loaded. Call load_data() first.")
            return None, None, None, None
        
        try:
            # Split features and target
            X = self.data[self.features]
            y = self.data[self.target]
            
            # Convert categorical target to numeric
            unique_groups = y.unique()
            group_to_num = {group: i for i, group in enumerate(unique_groups)}
            y_numeric = y.map(group_to_num)
            
            logging.info(f"Target classes: {dict(zip(group_to_num.values(), group_to_num.keys()))}")
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_numeric, test_size=test_size, random_state=self.random_state, stratify=y_numeric
            )
            
            logging.info(f"Split data into {X_train.shape[0]} training and {X_test.shape[0]} testing samples")
            
            # Handle missing values
            if impute_method != 'none':
                if impute_method == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
                elif impute_method == 'mean':
                    X_train = X_train.fillna(X_train.mean())
                    X_test = X_test.fillna(X_train.mean())  # Use training mean for test data
                elif impute_method == 'median':
                    X_train = X_train.fillna(X_train.median())
                    X_test = X_test.fillna(X_train.median())  # Use training median for test data
                
                logging.info(f"Imputed missing values using {impute_method} method")
            
            # Scale features
            if scale_method != 'none':
                if scale_method == 'standard':
                    scaler = StandardScaler()
                elif scale_method == 'robust':
                    scaler = RobustScaler()
                
                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
                
                logging.info(f"Scaled features using {scale_method} scaler")
            
            # Store processed data
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            
            # Store class mapping for later use
            self.class_mapping = {i: group for group, i in group_to_num.items()}
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def select_features(self, method='rfe', n_features=20, rfe_estimator=None):
        """
        Perform feature selection to identify potential biomarkers
        
        Parameters:
        -----------
        method : str
            Feature selection method ('anova', 'rfe', 'rfecv')
        n_features : int
            Number of features to select (default: 20)
        rfe_estimator : estimator object
            Base estimator for RFE/RFECV (default: RandomForestClassifier)
            
        Returns:
        --------
        list
            Selected feature names
        """
        logging.info(f"Performing feature selection using {method} method")
        
        if self.X_train is None or self.y_train is None:
            logging.error("No preprocessed data available. Call preprocess_data() first.")
            return []
        
        try:
            if method == 'anova':
                # ANOVA F-value between features and target
                selector = SelectKBest(f_classif, k=n_features)
                selector.fit(self.X_train, self.y_train)
                
                # Get selected feature indices
                feature_indices = selector.get_support(indices=True)
                
                # Get feature names
                selected_features = self.X_train.columns[feature_indices].tolist()
                
                # Store feature importance scores
                self.feature_importance['anova'] = {
                    'features': selected_features,
                    'scores': selector.scores_[feature_indices]
                }
                
            elif method in ['rfe', 'rfecv']:
                # Set default estimator if not provided
                if rfe_estimator is None:
                    rfe_estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                
                if method == 'rfe':
                    # Recursive Feature Elimination
                    selector = RFE(estimator=rfe_estimator, n_features_to_select=n_features, step=1)
                    selector.fit(self.X_train, self.y_train)
                else:
                    # Recursive Feature Elimination with Cross-Validation
                    selector = RFECV(estimator=rfe_estimator, step=1, cv=5, scoring='accuracy')
                    selector.fit(self.X_train, self.y_train)
                    n_features = selector.n_features_  # Update with optimal number of features
                
                # Get selected feature indices
                feature_indices = selector.get_support(indices=True)
                
                # Get feature names
                selected_features = self.X_train.columns[feature_indices].tolist()
                
                # Store feature importance
                if hasattr(rfe_estimator, 'feature_importances_'):
                    # For tree-based methods
                    self.feature_importance[method] = {
                        'features': selected_features,
                        'scores': rfe_estimator.feature_importances_[feature_indices]
                    }
            else:
                logging.error(f"Unknown feature selection method: {method}")
                return []
            
            logging.info(f"Selected {len(selected_features)} features")
            
            # Store selected features
            self.selected_features = selected_features
            
            return selected_features
            
        except Exception as e:
            logging.error(f"Error in feature selection: {str(e)}")
            return []
    
    def train_models(self, models=None):
        """
        Train multiple machine learning models
        
        Parameters:
        -----------
        models : list
            List of model names to train
            Default: ['rf', 'svm', 'lr', 'gb', 'xgb']
            
        Returns:
        --------
        dict
            Dictionary of trained models and their performances
        """
        logging.info("Training machine learning models")
        
        if self.X_train is None or self.y_train is None:
            logging.error("No preprocessed data available. Call preprocess_data() first.")
            return {}
        
        if not self.selected_features:
            logging.warning("No features selected. Using all features.")
            X_train = self.X_train
            X_test = self.X_test
        else:
            # Use only selected features
            X_train = self.X_train[self.selected_features]
            X_test = self.X_test[self.selected_features]
        
        # Define default models if not provided
        if models is None:
            models = ['rf', 'svm', 'lr', 'gb', 'xgb']
        
        # Dictionary to store trained models and their performances
        trained_models = {}
        
        try:
            # Check if class imbalance exists
            class_counts = np.bincount(self.y_train)
            min_class = np.min(class_counts)
            max_class = np.max(class_counts)
            imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
            
            # Use SMOTE for imbalanced data
            use_smote = imbalance_ratio > 1.5 and min_class >= 5
            
            if use_smote:
                logging.info(f"Detected class imbalance (ratio: {imbalance_ratio:.2f}). Applying SMOTE.")
                smote = SMOTE(random_state=self.random_state)
                X_train_res, y_train_res = smote.fit_resample(X_train, self.y_train)
                
                # Update class counts after SMOTE
                class_counts_after = np.bincount(y_train_res)
                logging.info(f"Class distribution after SMOTE: {dict(zip(range(len(class_counts_after)), class_counts_after))}")
            else:
                X_train_res, y_train_res = X_train, self.y_train
            
            # Train each model
            for model_name in models:
                logging.info(f"Training {model_name} model")
                
                # Initialize model
                if model_name == 'rf':
                    model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                elif model_name == 'svm':
                    model = SVC(probability=True, random_state=self.random_state)
                elif model_name == 'lr':
                    model = LogisticRegression(max_iter=1000, random_state=self.random_state)
                elif model_name == 'gb':
                    model = GradientBoostingClassifier(random_state=self.random_state)
                elif model_name == 'xgb':
                    model = XGBClassifier(random_state=self.random_state)
                else:
                    logging.warning(f"Unknown model type: {model_name}. Skipping.")
                    continue
                
                # Train model
                model.fit(X_train_res, y_train_res)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate performance metrics
                metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred, average='weighted'),
                    'recall': recall_score(self.y_test, y_pred, average='weighted'),
                    'f1': f1_score(self.y_test, y_pred, average='weighted')
                }
                
                # Calculate ROC AUC if applicable (for binary classification)
                if y_prob is not None and len(np.unique(self.y_test)) == 2:
                    fpr, tpr, _ = roc_curve(self.y_test, y_prob[:, 1])
                    metrics['roc_auc'] = auc(fpr, tpr)
                
                # Store model and metrics
                trained_models[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_prob,
                    'feature_names': X_train.columns.tolist()
                }
                
                logging.info(f"{model_name} performance: {metrics}")
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = {
                        'features': X_train.columns.tolist(),
                        'scores': model.feature_importances_
                    }
            
            # Store trained models
            self.models = trained_models
            
            # Identify best model based on F1 score
            best_model_name = max(trained_models.keys(), key=lambda k: trained_models[k]['metrics']['f1'])
            self.best_model = {
                'name': best_model_name,
                'model': trained_models[best_model_name]['model'],
                'metrics': trained_models[best_model_name]['metrics'],
                'feature_names': trained_models[best_model_name]['feature_names']
            }
            
            logging.info(f"Best model: {best_model_name} with F1 score: {self.best_model['metrics']['f1']:.4f}")
            
            return trained_models
            
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
            return {}
    
    def visualize_feature_importance(self, output_file, top_n=20):
        """
        Visualize feature importance from models
        
        Parameters:
        -----------
        output_file : str
            Path to save the visualization
        top_n : int
            Number of top features to display (default: 20)
            
        Returns:
        --------
        None
        """
        logging.info(f"Visualizing feature importance for top {top_n} features")
        
        if not self.feature_importance:
            logging.error("No feature importance data available.")
            return
        
        try:
            # Create figure with subplots for each model
            n_models = len(self.feature_importance)
            fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
            
            # Handle case with only one model
            if n_models == 1:
                axes = [axes]
            
            # Process each model
            for i, (model_name, importance) in enumerate(self.feature_importance.items()):
                features = importance['features']
                scores = importance['scores']
                
                # Create DataFrame for easier sorting
                importance_df = pd.DataFrame({'Feature': features, 'Importance': scores})
                
                # Sort by importance and get top N
                importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
                
                # Plot horizontal bar chart
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=axes[i])
                
                # Customize plot
                axes[i].set_title(f"Top {top_n} Features from {model_name.upper()}", fontsize=14)
                axes[i].set_xlabel('Importance', fontsize=12)
                axes[i].set_ylabel('Feature', fontsize=12)
                axes[i].tick_params(axis='y', labelsize=10)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Feature importance visualization saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Error visualizing feature importance: {str(e)}")
    
    def visualize_predictions(self, output_file):
        """
        Visualize model predictions and confusion matrices
        
        Parameters:
        -----------
        output_file : str
            Path to save the visualization
            
        Returns:
        --------
        None
        """
        logging.info("Visualizing model predictions")
        
        if not self.models:
            logging.error("No trained models available.")
            return
        
        try:
            # Create figure with subplots for each model
            n_models = len(self.models)
            fig, axes = plt.subplots(n_models, 2, figsize=(15, 5 * n_models))
            
            # For each model
            for i, (model_name, model_data) in enumerate(self.models.items()):
                predictions = model_data['predictions']
                true_labels = self.y_test
                
                # Get class names
                class_names = [self.class_mapping.get(j, f"Class {j}") for j in range(len(self.class_mapping))]
                
                # Plot confusion matrix
                cm = confusion_matrix(true_labels, predictions)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                           yticklabels=class_names, ax=axes[i, 0])
                axes[i, 0].set_title(f"{model_name.upper()} - Confusion Matrix", fontsize=14)
                axes[i, 0].set_xlabel('Predicted Label', fontsize=12)
                axes[i, 0].set_ylabel('True Label', fontsize=12)
                
                # Plot ROC curve if applicable (binary classification)
                if 'probabilities' in model_data and model_data['probabilities'] is not None and len(class_names) == 2:
                    probs = model_data['probabilities'][:, 1]
                    fpr, tpr, _ = roc_curve(true_labels, probs)
                    roc_auc = auc(fpr, tpr)
                    
                    axes[i, 1].plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    axes[i, 1].plot([0, 1], [0, 1], 'k--', lw=2)
                    axes[i, 1].set_xlim([0.0, 1.0])
                    axes[i, 1].set_ylim([0.0, 1.05])
                    axes[i, 1].set_xlabel('False Positive Rate', fontsize=12)
                    axes[i, 1].set_ylabel('True Positive Rate', fontsize=12)
                    axes[i, 1].set_title(f"{model_name.upper()} - ROC Curve", fontsize=14)
                    axes[i, 1].legend(loc="lower right")
                else:
                    # For multiclass, show metrics
                    metrics = model_data['metrics']
                    text = (f"Accuracy: {metrics['accuracy']:.4f}\n"
                           f"Precision: {metrics['precision']:.4f}\n"
                           f"Recall: {metrics['recall']:.4f}\n"
                           f"F1 Score: {metrics['f1']:.4f}")
                    
                    axes[i, 1].text(0.5, 0.5, text, ha='center', va='center', fontsize=14)
                    axes[i, 1].set_title(f"{model_name.upper()} - Performance Metrics", fontsize=14)
                    axes[i, 1].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Prediction visualization saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Error visualizing predictions: {str(e)}")
    
    def visualize_data_distribution(self, output_file):
        """
        Visualize data distribution and dimensionality reduction
        
        Parameters:
        -----------
        output_file : str
            Path to save the visualization
            
        Returns:
        --------
        None
        """
        logging.info("Visualizing data distribution")
        
        if self.X_train is None or self.y_train is None:
            logging.error("No preprocessed data available.")
            return
        
        try:
            # Use selected features if available, otherwise use all features
            if self.selected_features:
                X = self.data[self.selected_features]
            else:
                X = self.data[self.features]
            
            y = self.data[self.target]
            
            # Create figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 1. Class distribution
            class_counts = y.value_counts()
            sns.barplot(x=class_counts.index, y=class_counts.values, ax=axes[0])
            axes[0].set_title("Class Distribution", fontsize=14)
            axes[0].set_xlabel("Class", fontsize=12)
            axes[0].set_ylabel("Count", fontsize=12)
            
            # 2. PCA visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X)
            pca_df = pd.DataFrame({'PCA1': pca_result[:, 0], 'PCA2': pca_result[:, 1], 'Class': y})
            
            sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Class', ax=axes[1])
            axes[1].set_title(f"PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.2f})", fontsize=14)
            axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f})", fontsize=12)
            axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f})", fontsize=12)
            
            # 3. t-SNE visualization
            tsne = TSNE(n_components=2, random_state=self.random_state)
            tsne_result = tsne.fit_transform(X)
            tsne_df = pd.DataFrame({'t-SNE1': tsne_result[:, 0], 't-SNE2': tsne_result[:, 1], 'Class': y})
            
            sns.scatterplot(data=tsne_df, x='t-SNE1', y='t-SNE2', hue='Class', ax=axes[2])
            axes[2].set_title("t-SNE Projection", fontsize=14)
            axes[2].set_xlabel("t-SNE1", fontsize=12)
            axes[2].set_ylabel("t-SNE2", fontsize=12)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Data distribution visualization saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Error visualizing data distribution: {str(e)}")
    
    def generate_shap_explanations(self, output_file):
        """
        Generate SHAP explanations for model interpretability
        
        Parameters:
        -----------
        output_file : str
            Path to save the visualizations
            
        Returns:
        --------
        None
        """
        logging.info("Generating SHAP explanations for model interpretability")
        
        if not self.best_model:
            logging.error("No best model available.")
            return
        
        try:
            model_name = self.best_model['name']
            model = self.best_model['model']
            feature_names = self.best_model['feature_names']
            
            # Use selected features if available, otherwise use all features
            if feature_names:
                X_test = self.X_test[feature_names]
            else:
                X_test = self.X_test
            
            # Create SHAP explainer
            if model_name in ['rf', 'gb', 'xgb']:
                # For tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                
                # For multi-class classification, shap_values is a list of arrays
                if isinstance(shap_values, list):
                    # Create figure with multiple subplots for multi-class
                    n_classes = len(shap_values)
                    fig, axes = plt.subplots(n_classes, 1, figsize=(12, 5 * n_classes))
                    
                    # Handle case with only one class
                    if n_classes == 1:
                        axes = [axes]
                    
                    for i in range(n_classes):
                        class_name = self.class_mapping.get(i, f"Class {i}")
                        
                        # Create a new figure for SHAP summary plot
                        plt.figure(figsize=(10, 8))
                        shap.summary_plot(shap_values[i], X_test, plot_type="bar", show=False)
                        plt.title(f"SHAP Feature Importance for {class_name}", fontsize=14)
                        plt.tight_layout()
                        
                        # Save class-specific plot
                        class_output_file = output_file.replace('.png', f'_class{i}.png')
                        plt.savefig(class_output_file, dpi=300, bbox_inches='tight')
                        plt.close()
                else:
                    # For binary classification
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                    plt.title("SHAP Feature Importance", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
            else:
                # For non-tree models
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 100))
                shap_values = explainer.shap_values(X_test.iloc[:100])
                
                # For multi-class classification, shap_values is a list of arrays
                if isinstance(shap_values, list):
                    # Create separate plots for each class
                    for i in range(len(shap_values)):
                        class_name = self.class_mapping.get(i, f"Class {i}")
                        
                        plt.figure(figsize=(10, 8))
                        shap.summary_plot(shap_values[i], X_test.iloc[:100], plot_type="bar", show=False)
                        plt.title(f"SHAP Feature Importance for {class_name}", fontsize=14)
                        plt.tight_layout()
                        
                        # Save class-specific plot
                        class_output_file = output_file.replace('.png', f'_class{i}.png')
                        plt.savefig(class_output_file, dpi=300, bbox_inches='tight')
                        plt.close()
                else:
                    # For binary classification
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_test.iloc[:100], plot_type="bar", show=False)
                    plt.title("SHAP Feature Importance", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
            
            logging.info(f"SHAP explanations saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Error generating SHAP explanations: {str(e)}")
    
    def generate_biomarker_report(self, output_file):
        """
        Generate a comprehensive HTML report for biomarker discovery
        
        Parameters:
        -----------
        output_file : str
            Path to save the HTML report
            
        Returns:
        --------
        None
        """
        logging.info("Generating biomarker discovery report")
        
        if not self.models or not self.feature_importance:
            logging.error("No models or feature importance data available.")
            return
        
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            
            # Get base filename for images
            base_filename = os.path.splitext(os.path.basename(output_file))[0]
            
            # Collect top biomarkers from all models
            all_biomarkers = {}
            for model_name, importance in self.feature_importance.items():
                features = importance['features']
                scores = importance['scores']
                
                # Create DataFrame for sorting
                importance_df = pd.DataFrame({'Feature': features, 'Importance': scores})
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Store top 10 biomarkers for each model
                all_biomarkers[model_name] = importance_df.head(10).to_dict(orient='records')
            
            # Identify consensus biomarkers
            biomarker_counts = {}
            for model_biomarkers in all_biomarkers.values():
                for biomarker in model_biomarkers:
                    feature = biomarker['Feature']
                    if feature in biomarker_counts:
                        biomarker_counts[feature] += 1
                    else:
                        biomarker_counts[feature] = 1
            
            # Sort by frequency
            consensus_biomarkers = [{'Feature': k, 'Count': v} for k, v in biomarker_counts.items()]
            consensus_biomarkers = sorted(consensus_biomarkers, key=lambda x: x['Count'], reverse=True)
            
            # Create HTML content
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Biomarker Discovery and Machine Learning Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2C3E50; }
                    h2 { color: #2980B9; }
                    h3 { color: #3498DB; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .summary { background-color: #eef6f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                    img { max-width: 100%; height: auto; margin: 10px 0; }
                    .feature-box { display: inline-block; margin: 5px; padding: 5px; border-radius: 3px; background-color: #3498db; color: white; }
                    .model-performance { display: flex; flex-wrap: wrap; }
                    .model-card { border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 10px; width: 300px; }
                    .model-card h3 { margin-top: 0; }
                    .metric { margin: 5px 0; }
                    .metric-name { font-weight: bold; }
                    .chart-container { display: flex; flex-wrap: wrap; justify-content: center; }
                    .chart { margin: 10px; max-width: 45%; }
                </style>
            </head>
            <body>
                <h1>Biomarker Discovery and Machine Learning Report</h1>
                
                <div class="summary">
                    <h2>Analysis Summary</h2>
                    <p>This report summarizes the results of biomarker discovery and machine learning analysis.</p>
            """
            
            # Add data summary
            if self.data is not None:
                html_content += f"""
                    <p>Number of samples: {self.data.shape[0]}</p>
                    <p>Number of features (proteins): {len(self.features)}</p>
                    <p>Number of classes: {len(self.data[self.target].unique())}</p>
                """
            
            html_content += """
                </div>
                
                <h2>Data Distribution and Visualization</h2>
                <div class="chart-container">
                    <img src="{}_data_distribution.png" alt="Data Distribution">
                </div>
                
                <h2>Identified Biomarkers</h2>
                <h3>Consensus Biomarkers</h3>
                <p>These biomarkers were identified as important by multiple models.</p>
                <table>
                    <tr>
                        <th>Biomarker</th>
                        <th>Number of Models</th>
                    </tr>
            """.format(base_filename)
            
            # Add consensus biomarkers
            for biomarker in consensus_biomarkers[:20]:  # Top 20 consensus biomarkers
                html_content += f"""
                    <tr>
                        <td>{biomarker['Feature']}</td>
                        <td>{biomarker['Count']}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h3>Model-Specific Biomarkers</h3>
                <div class="chart-container">
                    <img src="{}_feature_importance.png" alt="Feature Importance">
                </div>
                
                <h2>Model Performance</h2>
                <div class="model-performance">
            """.format(base_filename)
            
            # Add model performance
            for model_name, model_data in self.models.items():
                metrics = model_data['metrics']
                
                html_content += f"""
                    <div class="model-card">
                        <h3>{model_name.upper()}</h3>
                        <div class="metric"><span class="metric-name">Accuracy:</span> {metrics['accuracy']:.4f}</div>
                        <div class="metric"><span class="metric-name">Precision:</span> {metrics['precision']:.4f}</div>
                        <div class="metric"><span class="metric-name">Recall:</span> {metrics['recall']:.4f}</div>
                        <div class="metric"><span class="metric-name">F1 Score:</span> {metrics['f1']:.4f}</div>
                """
                
                if 'roc_auc' in metrics:
                    html_content += f'<div class="metric"><span class="metric-name">ROC AUC:</span> {metrics["roc_auc"]:.4f}</div>'
                
                html_content += """
                    </div>
                """
            
            html_content += """
                </div>
                
                <h2>Prediction Performance</h2>
                <div class="chart-container">
                    <img src="{}_predictions.png" alt="Model Predictions">
                </div>
                
                <h2>Model Interpretability (SHAP Analysis)</h2>
                <div class="chart-container">
                    <img src="{}_shap.png" alt="SHAP Analysis">
                </div>
                
                <h2>Conclusions and Recommendations</h2>
                <div class="summary">
                    <p>Based on the analysis, the following biomarkers show the strongest association with the target condition:</p>
                    <div>
            """.format(base_filename, base_filename)
            
            # Add top consensus biomarkers
            for biomarker in consensus_biomarkers[:5]:
                html_content += f'<span class="feature-box">{biomarker["Feature"]}</span>'
            
            # Add best model information
            if self.best_model:
                best_model_name = self.best_model['name']
                best_model_metrics = self.best_model['metrics']
                
                html_content += f"""
                    </div>
                    <p>The best performing model was <strong>{best_model_name.upper()}</strong> with an F1 score of {best_model_metrics['f1']:.4f}.</p>
                    <p>These biomarkers and model could be used for:</p>
                    <ul>
                        <li>Diagnostic applications to differentiate between conditions</li>
                        <li>Prognostic applications to predict disease outcomes</li>
                        <li>Therapeutic target identification</li>
                        <li>Patient stratification for personalized treatment</li>
                    </ul>
                """
            
            # Close HTML
            html_content += """
                </div>
                
                <hr>
                <p><em>Report generated by Proteomics Biomarker Discovery Pipeline</em></p>
            </body>
            </html>
            """
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            logging.info(f"Biomarker discovery report generated: {output_file}")
            
        except Exception as e:
            logging.error(f"Error generating biomarker report: {str(e)}")
    
    def discover_biomarkers(self, output_dir):
        """
        Run the complete biomarker discovery pipeline
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output files
            
        Returns:
        --------
        dict
            Dictionary with results and identified biomarkers
        """
        logging.info("Running complete biomarker discovery pipeline")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Preprocess data (if not already done)
            if self.X_train is None or self.y_train is None:
                self.preprocess_data()
            
            # Select features
            self.select_features(method='rfe', n_features=20)
            
            # Train models
            self.train_models()
            
            # Generate visualizations
            self.visualize_feature_importance(os.path.join(output_dir, "feature_importance.png"))
            self.visualize_predictions(os.path.join(output_dir, "predictions.png"))
            self.visualize_data_distribution(os.path.join(output_dir, "data_distribution.png"))
            self.generate_shap_explanations(os.path.join(output_dir, "shap.png"))
            
            # Generate comprehensive report
            self.generate_biomarker_report(os.path.join(output_dir, "biomarker_report.html"))
            
            # Collect results
            results = {
                'biomarkers': self.selected_features,
                'best_model': self.best_model['name'] if self.best_model else None,
                'model_metrics': {name: data['metrics'] for name, data in self.models.items()},
                'feature_importance': self.feature_importance
            }
            
            logging.info("Biomarker discovery pipeline completed successfully")
            
            return results
            
        except Exception as e:
            logging.error(f"Error in biomarker discovery pipeline: {str(e)}")
            raise

def main():
    try:
        # Get inputs and outputs from Snakemake
        quant_file = snakemake.input.quantification
        metadata_file = snakemake.input.metadata
        output_dir = snakemake.output.output_dir
        report_file = snakemake.output.report
        
        # Get parameters
        sample_col = snakemake.params.get("sample_col", "sample_id")
        group_col = snakemake.params.get("group_col", "condition")
        feature_selection_method = snakemake.params.get("feature_selection", "rfe")
        n_features = int(snakemake.params.get("n_features", 20))
        
        # Initialize biomarker discovery tool
        biomarker_tool = BiomarkerDiscovery()
        
        # Load data
        biomarker_tool.load_data(quant_file, metadata_file, sample_col=sample_col, group_col=group_col)
        
        # Run complete pipeline
        biomarker_tool.discover_biomarkers(output_dir)
        
        logging.info("Biomarker discovery workflow completed successfully")
        
    except Exception as e:
        logging.error(f"Error in biomarker discovery workflow: {str(e)}")
        raise

if __name__ == "__main__":
    main()

