# scripts/hail_rare_variant_test.py
import hail as hl
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(filename=snakemake.log[0], level=logging.INFO,
                   format='%(asctime)s %(levelname)s %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

def main():
    # Initialize Hail
    hl.init(log=snakemake.log[0])
    
    # Read the VEP-annotated MatrixTable
    logging.info(f"Reading VEP-annotated MatrixTable from {snakemake.input.mt}")
    mt = hl.read_matrix_table(snakemake.input.mt)
    
    # Import phenotype data
    logging.info(f"Importing phenotype data: {snakemake.input.phenotype}")
    phenotype_table = hl.import_table(snakemake.input.phenotype, 
                                      impute=True, 
                                      key='sample_id')
    
    # Ensure phenotype is properly annotated
    if 'phenotype' not in mt.col:
        logging.info("Annotating with phenotype data")
        mt = mt.annotate_cols(phenotype = phenotype_table[mt.s])
    
    # Filter for rare variants based on MAF threshold
    maf_threshold = snakemake.params.maf_threshold
    logging.info(f"Filtering for rare variants (MAF < {maf_threshold})")
    
    # Calculate allele frequencies
    mt = hl.variant_qc(mt)
    mt = mt.annotate_rows(maf = mt.variant_qc.AF[1])
    mt = mt.filter_rows(mt.maf < maf_threshold)
    
    # Optional: Filter for specific gene list if provided
    if snakemake.input.gene_list is not None:
        logging.info(f"Filtering for genes in gene list: {snakemake.input.gene_list}")
        gene_list = pd.read_csv(snakemake.input.gene_list, header=None)[0].tolist()
        mt = mt.filter_rows(mt.vep_parsed.gene_symbol.isin(gene_list))
    
    # Group variants by gene for burden testing
    logging.info("Grouping variants by gene for burden testing")
    
    # Create gene-based variant groups
    mt = mt.annotate_rows(gene_key = mt.vep_parsed.gene_symbol)
    
    # Convert case/control phenotype to binary for burden testing
    # Assuming the phenotype column contains 1 for cases and 0 for controls
    mt = mt.annotate_cols(binary_phenotype = hl.int(mt.phenotype.is_case))
    
    # Run the specified burden test
    burden_test = snakemake.params.burden_test.lower()
    
    logging.info(f"Running {burden_test} burden test")
    
    if burden_test == "skat":
        # Run SKAT test
        skat_results = hl.skat(
            mt.binary_phenotype,
            mt.GT.n_alt_alleles(),
            key=mt.gene_key
        )
        
        # Format results
        results_table = skat_results.select(
            gene = skat_results.key,
            p_value = skat_results.p_value,
            n_variants = skat_results.size
        )
        
    elif burden_test == "burden":
        # Run burden test
        burden_results = hl.burden(
            mt.binary_phenotype,
            mt.GT.n_alt_alleles(),
            key=mt.gene_key
        )
        
        # Format results
        results_table = burden_results.select(
            gene = burden_results.key,
            p_value = burden_results.p_value,
            n_variants = burden_results.size
        )
        
    elif burden_test == "skat-o":
        # Run SKAT-O test (this is a simplified example, as Hail doesn't directly support SKAT-O)
        # In practice, you might need to implement SKAT-O or call it via R
        logging.warning("SKAT-O not directly implemented in Hail, falling back to SKAT")
        skat_results = hl.skat(
            mt.binary_phenotype,
            mt.GT.n_alt_alleles(),
            key=mt.gene_key
        )
        
        # Format results
        results_table = skat_results.select(
            gene = skat_results.key,
            p_value = skat_results.p_value,
            n_variants = skat_results.size,
            test_method = "SKAT (fallback from SKAT-O)"
        )
    
    else:
        logging.error(f"Unsupported burden test: {burden_test}")
        raise ValueError(f"Unsupported burden test: {burden_test}. Supported tests are: skat, burden, skat-o")
    
    # Add genomic annotations
    results_table = results_table.annotate(
        chromosome = hl.str(results_table.gene.contig),
        position = results_table.gene.position,
        gene_id = results_table.gene.gene_id
    )
    
    # Sort by p-value
    results_table = results_table.order_by(results_table.p_value)
    
    # Export results to TSV
    logging.info(f"Exporting results to {snakemake.output.results}")
    results_table.export(snakemake.output.results)
    
    # Write the filtered MatrixTable with rare variants
    logging.info(f"Writing rare variants MatrixTable to {snakemake.output.mt}")
    mt.write(snakemake.output.mt, overwrite=True)
    
    logging.info("Rare variant burden testing completed successfully")

if __name__ == "__main__":
    main()



