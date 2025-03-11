#!/usr/bin/env python3
"""
Protein Structure and Domain Analysis Module for Proteomics Data

This script:
1. Analyzes protein domains, motifs, and structural features
2. Predicts protein structure using AlphaFold API (if available)
3. Maps post-translational modifications to structural regions
4. Visualizes protein structures and domains
5. Provides functional insights based on structural analysis

Author: Jiyang Jiang
Date: March 11, 2025
"""

import pandas as pd
import numpy as np
import os
import logging
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set, Optional, Union
from pathlib import Path
import subprocess
import io
import re
from collections import defaultdict
import warnings
from Bio import SeqIO, Seq, SeqRecord, ExPASy, SwissProt
from Bio.PDB import *
from Bio.SeqUtils import ProtParam
import tempfile
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(snakemake.log[0]),
        logging.StreamHandler()
    ]
)

class ProteinStructureAnalyzer:
    """Class for protein structure and domain analysis"""
    
    def __init__(self, organism="human"):
        """
        Initialize the protein structure analyzer
        
        Parameters:
        -----------
        organism : str
            Organism name (default: "human")
        """
        self.organism = organism
        self.uniprot_api_url = "https://rest.uniprot.org/uniprotkb"
        self.interpro_api_url = "https://www.ebi.ac.uk/interpro/api"
        self.alphafold_api_url = "https://alphafold.ebi.ac.uk/api"
        self.protein_data = {}
        self.structure_data = {}
        self.domain_data = {}
        self.ptm_data = {}
        
    def fetch_protein_sequences(self, protein_ids):
        """
        Fetch protein sequences from UniProt
        
        Parameters:
        -----------
        protein_ids : list
            List of UniProt protein IDs
            
        Returns:
        --------
        dict
            Dictionary mapping protein IDs to their sequences
        """
        logging.info(f"Fetching protein sequences for {len(protein_ids)} proteins")
        
        sequences = {}
        
        for protein_id in protein_ids:
            try:
                # Query UniProt API
                url = f"{self.uniprot_api_url}/{protein_id}.fasta"
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Parse FASTA
                    fasta_io = io.StringIO(response.text)
                    for record in SeqIO.parse(fasta_io, "fasta"):
                        sequences[protein_id] = str(record.seq)
                    logging.info(f"Retrieved sequence for {protein_id}")
                else:
                    logging.warning(f"Failed to retrieve sequence for {protein_id}")
                    
            except Exception as e:
                logging.error(f"Error fetching sequence for {protein_id}: {str(e)}")
        
        logging.info(f"Retrieved sequences for {len(sequences)} out of {len(protein_ids)} proteins")
        return sequences
    
    def fetch_protein_features(self, protein_ids):
        """
        Fetch protein features (domains, motifs, etc.) from UniProt
        
        Parameters:
        -----------
        protein_ids : list
            List of UniProt protein IDs
            
        Returns:
        --------
        dict
            Dictionary mapping protein IDs to their features
        """
        logging.info(f"Fetching protein features for {len(protein_ids)} proteins")
        
        features = {}
        
        for protein_id in protein_ids:
            try:
                # Query UniProt API for JSON format
                url = f"{self.uniprot_api_url}/{protein_id}.json"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract features
                    protein_features = {
                        'domains': [],
                        'motifs': [],
                        'ptms': [],
                        'secondary_structure': [],
                        'binding_sites': []
                    }
                    
                    if 'features' in data:
                        for feature in data['features']:
                            feature_type = feature.get('type', '')
                            
                            # Extract feature location
                            location = feature.get('location', {})
                            start = location.get('start', {}).get('value', None)
                            end = location.get('end', {}).get('value', None)
                            
                            # Skip features without positions
                            if start is None or end is None:
                                continue
                            
                            # Categorize feature
                            if feature_type in ['DOMAIN', 'REGION']:
                                protein_features['domains'].append({
                                    'name': feature.get('description', feature_type),
                                    'start': start,
                                    'end': end
                                })
                            elif feature_type in ['MOTIF', 'SITE']:
                                protein_features['motifs'].append({
                                    'name': feature.get('description', feature_type),
                                    'start': start,
                                    'end': end
                                })
                            elif feature_type in ['MODIFIED', 'CROSSLNK', 'LIPID', 'CARBOHYD']:
                                protein_features['ptms'].append({
                                    'type': feature_type,
                                    'description': feature.get('description', ''),
                                    'position': start
                                })
                            elif feature_type in ['HELIX', 'STRAND', 'TURN']:
                                protein_features['secondary_structure'].append({
                                    'type': feature_type,
                                    'start': start,
                                    'end': end
                                })
                            elif feature_type in ['BINDING', 'ACT_SITE', 'METAL', 'NP_BIND']:
                                protein_features['binding_sites'].append({
                                    'type': feature_type,
                                    'description': feature.get('description', ''),
                                    'start': start,
                                    'end': end if end is not None else start
                                })
                    
                    features[protein_id] = protein_features
                    logging.info(f"Retrieved features for {protein_id}")
                else:
                    logging.warning(f"Failed to retrieve features for {protein_id}")
                    
            except Exception as e:
                logging.error(f"Error fetching features for {protein_id}: {str(e)}")
        
        logging.info(f"Retrieved features for {len(features)} out of {len(protein_ids)} proteins")
        self.domain_data = features
        return features
    
    def fetch_protein_3d_structures(self, protein_ids, use_alphafold=True):
        """
        Fetch 3D structures from PDB and/or AlphaFold
        
        Parameters:
        -----------
        protein_ids : list
            List of UniProt protein IDs
        use_alphafold : bool
            Whether to fetch AlphaFold predicted structures if experimental structures are not available
            
        Returns:
        --------
        dict
            Dictionary mapping protein IDs to structure information
        """
        logging.info(f"Fetching 3D structures for {len(protein_ids)} proteins")
        
        structures = {}
        
        for protein_id in protein_ids:
            try:
                # First check if experimental structures exist in PDB
                url = f"{self.uniprot_api_url}/{protein_id}.json"
                response = requests.get(url)
                
                pdb_ids = []
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract PDB cross-references
                    if 'uniProtKBCrossReferences' in data:
                        for xref in data['uniProtKBCrossReferences']:
                            if xref.get('database') == 'PDB':
                                pdb_id = xref.get('id')
                                if pdb_id:
                                    pdb_ids.append(pdb_id)
                
                structure_info = {
                    'experimental': pdb_ids,
                    'predicted': None,
                    'source': None,
                    'path': None
                }
                
                # If experimental structures found, use the first one
                if pdb_ids:
                    structure_info['source'] = 'PDB'
                    logging.info(f"Found {len(pdb_ids)} experimental structures for {protein_id}")
                    
                # If no experimental structures or we want to use AlphaFold regardless
                elif use_alphafold:
                    # Check AlphaFold DB
                    alphafold_url = f"{self.alphafold_api_url}/prediction/{protein_id}"
                    af_response = requests.get(alphafold_url)
                    
                    if af_response.status_code == 200:
                        af_data = af_response.json()
                        
                        if 'pdbUrl' in af_data:
                            structure_info['predicted'] = protein_id
                            structure_info['source'] = 'AlphaFold'
                            logging.info(f"Found AlphaFold structure for {protein_id}")
                    else:
                        logging.warning(f"No AlphaFold structure found for {protein_id}")
                
                structures[protein_id] = structure_info
                
            except Exception as e:
                logging.error(f"Error fetching structures for {protein_id}: {str(e)}")
        
        logging.info(f"Retrieved structure information for {len(structures)} proteins")
        self.structure_data = structures
        return structures
    
    def fetch_interpro_domains(self, protein_ids):
        """
        Fetch domain information from InterPro
        
        Parameters:
        -----------
        protein_ids : list
            List of UniProt protein IDs
            
        Returns:
        --------
        dict
            Dictionary mapping protein IDs to InterPro domains
        """
        logging.info(f"Fetching InterPro domains for {len(protein_ids)} proteins")
        
        interpro_domains = {}
        
        for protein_id in protein_ids:
            try:
                # Query InterPro API
                url = f"{self.interpro_api_url}/protein/reviewed/{protein_id}"
                headers = {"Accept": "application/json"}
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract domains
                    domains = []
                    
                    if 'entry_subset' in data:
                        for entry in data['entry_subset']:
                            entry_acc = entry.get('metadata', {}).get('accession')
                            entry_name = entry.get('metadata', {}).get('name')
                            entry_type = entry.get('metadata', {}).get('type')
                            
                            if 'proteins' in entry and protein_id in entry['proteins']:
                                protein_data = entry['proteins'][protein_id]
                                
                                if 'entry_protein_locations' in protein_data:
                                    for location in protein_data['entry_protein_locations']:
                                        fragments = location.get('fragments', [])
                                        
                                        for fragment in fragments:
                                            start = fragment.get('start')
                                            end = fragment.get('end')
                                            
                                            if start is not None and end is not None:
                                                domains.append({
                                                    'id': entry_acc,
                                                    'name': entry_name,
                                                    'type': entry_type,
                                                    'start': start,
                                                    'end': end
                                                })
                    
                    interpro_domains[protein_id] = domains
                    logging.info(f"Retrieved {len(domains)} InterPro domains for {protein_id}")
                else:
                    logging.warning(f"Failed to retrieve InterPro domains for {protein_id}")
                    
            except Exception as e:
                logging.error(f"Error fetching InterPro domains for {protein_id}: {str(e)}")
        
        logging.info(f"Retrieved InterPro domains for {len(interpro_domains)} proteins")
        
        # Merge with existing domain data
        for protein_id, domains in interpro_domains.items():
            if protein_id in self.domain_data:
                # Create a new 'interpro' category in the domain data
                self.domain_data[protein_id]['interpro'] = domains
            else:
                self.domain_data[protein_id] = {'interpro': domains}
        
        return interpro_domains
    
    def analyze_protein_properties(self, sequences):
        """
        Analyze basic protein properties
        
        Parameters:
        -----------
        sequences : dict
            Dictionary mapping protein IDs to their sequences
            
        Returns:
        --------
        dict
            Dictionary mapping protein IDs to their properties
        """
        logging.info(f"Analyzing properties for {len(sequences)} proteins")
        
        properties = {}
        
        for protein_id, sequence in sequences.items():
            try:
                # Use BioPython's ProtParam
                prot_param = ProtParam.ProteinAnalysis(sequence)
                
                # Calculate basic properties
                props = {
                    'length': len(sequence),
                    'molecular_weight': prot_param.molecular_weight(),
                    'isoelectric_point': prot_param.isoelectric_point(),
                    'aromaticity': prot_param.aromaticity(),
                    'instability_index': prot_param.instability_index(),
                    'gravy': prot_param.gravy(),
                    'secondary_structure_fraction': prot_param.secondary_structure_fraction()
                }
                
                # Calculate amino acid composition
                aa_composition = prot_param.get_amino_acids_percent()
                props['aa_composition'] = aa_composition
                
                properties[protein_id] = props
                
            except Exception as e:
                logging.error(f"Error analyzing properties for {protein_id}: {str(e)}")
        
        logging.info(f"Analyzed properties for {len(properties)} proteins")
        return properties
    
    def map_ptms_to_domains(self, protein_id):
        """
        Map post-translational modifications to domains
        
        Parameters:
        -----------
        protein_id : str
            UniProt protein ID
            
        Returns:
        --------
        dict
            Dictionary mapping domains to PTMs
        """
        if protein_id not in self.domain_data:
            logging.warning(f"No domain data available for {protein_id}")
            return {}
        
        domain_data = self.domain_data[protein_id]
        
        if 'ptms' not in domain_data or 'domains' not in domain_data:
            logging.warning(f"No PTM or domain data available for {protein_id}")
            return {}
        
        ptms = domain_data['ptms']
        domains = domain_data['domains']
        interpro_domains = domain_data.get('interpro', [])
        
        # Combine all domains
        all_domains = domains + interpro_domains
        
        # Map PTMs to domains
        ptm_domain_map = defaultdict(list)
        
        for ptm in ptms:
            ptm_pos = ptm['position']
            
            for domain in all_domains:
                if domain['start'] <= ptm_pos <= domain['end']:
                    # Format domain name
                    domain_name = f"{domain.get('name', 'Domain')} ({domain['start']}-{domain['end']})"
                    
                    # Add PTM to this domain
                    ptm_domain_map[domain_name].append(ptm)
        
        return dict(ptm_domain_map)
    
    def visualize_protein_domains(self, protein_id, output_file):
        """
        Create a visualization of protein domains and features
        
        Parameters:
        -----------
        protein_id : str
            UniProt protein ID
        output_file : str
            Path to save the visualization
            
        Returns:
        --------
        None
        """
        logging.info(f"Creating domain visualization for {protein_id}")
        
        if protein_id not in self.domain_data:
            logging.warning(f"No domain data available for {protein_id}")
            return
        
        domain_data = self.domain_data[protein_id]
        
        # Get protein length from UniProt if not already in data
        protein_length = 0
        for domain_type in ['domains', 'interpro']:
            if domain_type in domain_data and domain_data[domain_type]:
                for domain in domain_data[domain_type]:
                    protein_length = max(protein_length, domain.get('end', 0))
        
        if protein_length == 0:
            # Try to get sequence to determine length
            sequences = self.fetch_protein_sequences([protein_id])
            if protein_id in sequences:
                protein_length = len(sequences[protein_id])
            else:
                logging.warning(f"Could not determine protein length for {protein_id}")
                return
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Track current y-position
        y_pos = 0
        
        # Define colors for different feature types
        colors = {
            'domains': '#3498db',       # Blue
            'interpro': '#2ecc71',     # Green
            'motifs': '#f1c40f',       # Yellow
            'secondary_structure': '#95a5a6',  # Gray
            'binding_sites': '#e74c3c'  # Red
        }
        
        # Draw protein backbone
        plt.plot([0, protein_length], [y_pos, y_pos], 'k-', linewidth=2)
        
        # Add tick marks every 100 amino acids
        tick_positions = np.arange(0, protein_length + 100, 100)
        tick_positions = tick_positions[tick_positions <= protein_length]
        plt.xticks(tick_positions)
        
        # Label for backbone
        plt.text(-50, y_pos, 'Protein', verticalalignment='center')
        
        # Step size for different feature types
        step = 1
        
        # Draw domains
        if 'domains' in domain_data and domain_data['domains']:
            y_pos -= step
            for domain in domain_data['domains']:
                start = domain['start']
                end = domain['end']
                name = domain.get('name', 'Domain')
                
                plt.gca().add_patch(
                    Rectangle((start, y_pos - 0.3), end - start, 0.6, 
                             facecolor=colors['domains'], alpha=0.7)
                )
                
                # Add domain name in the middle of the domain
                mid_point = (start + end) / 2
                plt.text(mid_point, y_pos, name, 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        fontsize=8)
        
        # Draw InterPro domains
        if 'interpro' in domain_data and domain_data['interpro']:
            y_pos -= step
            for domain in domain_data['interpro']:
                start = domain['start']
                end = domain['end']
                name = domain.get('name', 'InterPro')
                
                plt.gca().add_patch(
                    Rectangle((start, y_pos - 0.3), end - start, 0.6, 
                             facecolor=colors['interpro'], alpha=0.7)
                )
                
                # Add domain name in the middle of the domain
                mid_point = (start + end) / 2
                plt.text(mid_point, y_pos, name, 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        fontsize=8)
        
        # Draw motifs
        if 'motifs' in domain_data and domain_data['motifs']:
            y_pos -= step
            for motif in domain_data['motifs']:
                start = motif['start']
                end = motif['end']
                name = motif.get('name', 'Motif')
                
                plt.gca().add_patch(
                    Rectangle((start, y_pos - 0.3), end - start, 0.6, 
                             facecolor=colors['motifs'], alpha=0.7)
                )
                
                # Add motif name in the middle
                mid_point = (start + end) / 2
                plt.text(mid_point, y_pos, name, 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        fontsize=8)
        
        # Draw secondary structure
        if 'secondary_structure' in domain_data and domain_data['secondary_structure']:
            y_pos -= step
            for ss in domain_data['secondary_structure']:
                start = ss['start']
                end = ss['end']
                ss_type = ss.get('type', 'SS')
                
                plt.gca().add_patch(
                    Rectangle((start, y_pos - 0.3), end - start, 0.6, 
                             facecolor=colors['secondary_structure'], alpha=0.7)
                )
                
                # Add secondary structure type in the middle
                mid_point = (start + end) / 2
                plt.text(mid_point, y_pos, ss_type, 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        fontsize=8)
        
        # Draw binding sites
        if 'binding_sites' in domain_data and domain_data['binding_sites']:
            y_pos -= step
            for site in domain_data['binding_sites']:
                start = site['start']
                end = site['end']
                site_type = site.get('type', 'Binding')
                
                plt.gca().add_patch(
                    Rectangle((start, y_pos - 0.3), end - start, 0.6, 
                             facecolor=colors['binding_sites'], alpha=0.7)
                )
                
                # Add binding site type in the middle
                mid_point = (start + end) / 2
                plt.text(mid_point, y_pos, site_type, 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        fontsize=8)
        
        # Draw PTMs as points
        if 'ptms' in domain_data and domain_data['ptms']:
            y_pos -= step
            
            for ptm in domain_data['ptms']:
                position = ptm['position']
                ptm_type = ptm.get('type', 'PTM')
                
                plt.plot(position, y_pos, 'ro', markersize=6)
                
                # Add PTM type above the point
                plt.text(position, y_pos + 0.3, ptm_type, 
                        horizontalalignment='center', 
                        verticalalignment='bottom',
                        fontsize=8, rotation=90)
        
        # Set plot limits
        plt.xlim(-100, protein_length + 100)
        plt.ylim(y_pos - 1, 1)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='k', lw=2, label='Protein Backbone'),
            plt.Rectangle((0, 0), 1, 1, facecolor=colors['domains'], alpha=0.7, label='Domains'),
            plt.Rectangle((0, 0), 1, 1, facecolor=colors['interpro'], alpha=0.7, label='InterPro Domains'),
            plt.Rectangle((0, 0), 1, 1, facecolor=colors['motifs'], alpha=0.7, label='Motifs'),
            plt.Rectangle((0, 0), 1, 1, facecolor=colors['secondary_structure'], alpha=0.7, label='Secondary Structure'),
            plt.Rectangle((0, 0), 1, 1, facecolor=colors['binding_sites'], alpha=0.7, label='Binding Sites'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='PTMs')
        ]
        
        plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        
        # Set title and labels
        plt.title(f"Domain Architecture of {protein_id}", fontsize=14)
        plt.xlabel('Amino Acid Position', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Domain visualization saved to {output_file}")
    
    def generate_domain_report(self, protein_ids, output_file):
        """
        Generate a comprehensive HTML report for domain analysis
        
        Parameters:
        -----------
        protein_ids : list
            List of UniProt protein IDs
        output_file : str
            Path to save the HTML report
            
        Returns:
        --------
        None
        """
        logging.info(f"Generating domain analysis report for {len(protein_ids)} proteins")
        
        # Get domain data for all proteins
        if not self.domain_data:
            self.fetch_protein_features(protein_ids)
            self.fetch_interpro_domains(protein_ids)
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Create HTML content
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Protein Domain Analysis Report</title>
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
                    .feature-box { display: inline-block; margin: 5px; padding: 5px; border-radius: 3px; }
                    .domain { background-color: #3498db; color: white; }
                    .motif { background-color: #f1c40f; color: black; }
                    .ptm { background-color: #e74c3c; color: white; }
                    .binding { background-color: #9b59b6; color: white; }
                </style>
            </head>
            <body>
                <h1>Protein Domain Analysis Report</h1>
                
                <div class="summary">
                    <h2>Analysis Summary</h2>
                    <p>This report provides an analysis of protein domains, motifs, and other structural features.</p>
                    <p>Number of proteins analyzed: {}</p>
                </div>
            """.format(len(protein_ids))
            
            # Add sections for each protein
            for protein_id in protein_ids:
                if protein_id not in self.domain_data:
                    continue
                
                domain_data = self.domain_data[protein_id]
                
                html_content += f"""
                <h2>Protein: {protein_id}</h2>
                
                <h3>Domain Architecture</h3>
                <img src="{protein_id}_domains.png" alt="Domain architecture for {protein_id}">
                """
                
                # Add domain table
                html_content += """
                <h3>Domains and Features</h3>
                <table>
                    <tr>
                        <th>Feature Type</th>
                        <th>Name/Description</th>
                        <th>Start</th>
                        <th>End</th>
                    </tr>
                """
                
                # Add domains from UniProt
                if 'domains' in domain_data and domain_data['domains']:
                    for domain in domain_data['domains']:
                        html_content += f"""
                        <tr>
                            <td>Domain (UniProt)</td>
                            <td>{domain.get('name', 'Domain')}</td>
                            <td>{domain['start']}</td>
                            <td>{domain['end']}</td>
                        </tr>
                        """
                
                # Add domains from InterPro
                if 'interpro' in domain_data and domain_data['interpro']:
                    for domain in domain_data['interpro']:
                        html_content += f"""
                        <tr>
                            <td>Domain (InterPro)</td>
                            <td>{domain.get('name', 'InterPro')} [{domain.get('id', '')}]</td>
                            <td>{domain['start']}</td>
                            <td>{domain['end']}</td>
                        </tr>
                        """
                
                # Add motifs
                if 'motifs' in domain_data and domain_data['motifs']:
                    for motif in domain_data['motifs']:
                        html_content += f"""
                        <tr>
                            <td>Motif</td>
                            <td>{motif.get('name', 'Motif')}</td>
                            <td>{motif['start']}</td>
                            <td>{motif['end']}</td>
                        </tr>
                        """
                
                # Add binding sites
                if 'binding_sites' in domain_data and domain_data['binding_sites']:
                    for site in domain_data['binding_sites']:
                        html_content += f"""
                        <tr>
                            <td>Binding Site</td>
                            <td>{site.get('type', 'Binding')} - {site.get('description', '')}</td>
                            <td>{site['start']}</td>
                            <td>{site['end']}</td>
                        </tr>
                        """
                
                html_content += "</table>"
                
                # Add PTM table
                if 'ptms' in domain_data and domain_data['ptms']:
                    html_content += """
                    <h3>Post-Translational Modifications</h3>
                    <table>
                        <tr>
                            <th>Type</th>
                            <th>Position</th>
                            <th>Description</th>
                        </tr>
                    """
                    
                    for ptm in domain_data['ptms']:
                        html_content += f"""
                        <tr>
                            <td>{ptm.get('type', 'PTM')}</td>
                            <td>{ptm['position']}</td>
                            <td>{ptm.get('description', '')}</td>
                        </tr>
                        """
                    
                    html_content += "</table>"
                
                # Add PTM-to-domain mapping
                ptm_domain_map = self.map_ptms_to_domains(protein_id)
                
                if ptm_domain_map:
                    html_content += """
                    <h3>PTMs Mapped to Domains</h3>
                    <table>
                        <tr>
                            <th>Domain</th>
                            <th>PTMs</th>
                        </tr>
                    """
                    
                    for domain_name, ptms in ptm_domain_map.items():
                        ptm_descriptions = []
                        for ptm in ptms:
                            ptm_desc = f"{ptm.get('type', 'PTM')} at position {ptm['position']}"
                            if 'description' in ptm and ptm['description']:
                                ptm_desc += f": {ptm['description']}"
                            ptm_descriptions.append(ptm_desc)
                        
                        html_content += f"""
                        <tr>
                            <td>{domain_name}</td>
                            <td>{', '.join(ptm_descriptions)}</td>
                        </tr>
                        """
                    
                    html_content += "</table>"
                
                # Add 3D structure information if available
                if protein_id in self.structure_data:
                    structure_info = self.structure_data[protein_id]
                    
                    html_content += """
                    <h3>3D Structure Information</h3>
                    <table>
                        <tr>
                            <th>Source</th>
                            <th>IDs</th>
                        </tr>
                    """
                    
                    if structure_info['experimental']:
                        html_content += f"""
                        <tr>
                            <td>Experimental (PDB)</td>
                            <td>{', '.join(structure_info['experimental'])}</td>
                        </tr>
                        """
                    
                    if structure_info['predicted']:
                        html_content += f"""
                        <tr>
                            <td>Predicted (AlphaFold)</td>
                            <td>{structure_info['predicted']}</td>
                        </tr>
                        """
                    
                    html_content += "</table>"
            
            # Close HTML
            html_content += """
                <hr>
                <p><em>Report generated by Protein Domain Analysis Pipeline</em></p>
            </body>
            </html>
            """
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            logging.info(f"Domain analysis report generated: {output_file}")
            
        except Exception as e:
            logging.error(f"Error generating domain report: {str(e)}")
    
    def analyze_proteins(self, protein_ids, output_dir):
        """
        Perform comprehensive protein structure analysis
        
        Parameters:
        -----------
        protein_ids : list
            List of UniProt protein IDs
        output_dir : str
            Directory to save output files
            
        Returns:
        --------
        None
        """
        logging.info(f"Performing comprehensive analysis for {len(protein_ids)} proteins")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Fetch protein sequences
        sequences = self.fetch_protein_sequences(protein_ids)
        
        # Analyze protein properties
        properties = self.analyze_protein_properties(sequences)
        
        # Fetch protein features (domains, motifs, PTMs)
        self.fetch_protein_features(protein_ids)
        
        # Fetch additional domain information from InterPro
        self.fetch_interpro_domains(protein_ids)
        
        # Fetch protein 3D structures
        self.fetch_protein_3d_structures(protein_ids)
        
        # Create domain visualizations for each protein
        for protein_id in protein_ids:
            domain_viz_file = os.path.join(output_dir, f"{protein_id}_domains.png")
            self.visualize_protein_domains(protein_id, domain_viz_file)
        
        # Generate comprehensive report
        report_file = os.path.join(output_dir, "domain_analysis_report.html")
        self.generate_domain_report(protein_ids, report_file)
        
        logging.info("Protein structure analysis completed successfully")

def main():
    try:
        # Get inputs and outputs from Snakemake
        diff_expr_file = snakemake.input.diff_expr
        output_dir = snakemake.output.output_dir
        report_file = snakemake.output.report
        
        # Get parameters
        organism = snakemake.params.get("organism", "human")
        log2fc_cutoff = float(snakemake.params.get("log2fc_cutoff", 1.0))
        pval_cutoff = float(snakemake.params.get("pval_cutoff", 0.05))
        max_proteins = int(snakemake.params.get("max_proteins", 10))
        
        # Load differential expression data
        diff_expr = pd.read_csv(diff_expr_file, sep='\t', index_col=0)
        
        # Filter for significant proteins
        sig_proteins = diff_expr[(abs(diff_expr['log2FC']) >= log2fc_cutoff) & 
                                (diff_expr['padj'] <= pval_cutoff)].copy()
        
        logging.info(f"Found {len(sig_proteins)} significant proteins for structure analysis")
        
        # Select top proteins for analysis (to avoid API limits)
        selected_proteins = sig_proteins.sort_values('padj').head(max_proteins).index.tolist()
        
        # Initialize protein structure analyzer
        structure_analyzer = ProteinStructureAnalyzer(organism=organism)
        
        # Perform comprehensive analysis
        structure_analyzer.analyze_proteins(selected_proteins, output_dir)
        
        logging.info("Protein structure analysis workflow completed successfully")
        
    except Exception as e:
        logging.error(f"Error in protein structure analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()

