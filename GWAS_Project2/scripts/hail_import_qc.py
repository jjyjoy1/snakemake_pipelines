# scripts/hail_import_qc.py
import hail as hl
import os
import pandas as pd
from bokeh.plotting import output_file, save
from bokeh.layouts import layout
import logging

# Configure logging
logging.basicConfig(filename=snakemake.log[0], level=logging.INFO,
                   format='%(asctime)s %(levelname)s %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

def main():
    # Initialize Hail
    hl.init(log=snakemake.log[0])
    
    # Import VCF file
    logging.info(f"Importing VCF file: {snakemake.input.vcf}")
    mt = hl.import_vcf(snakemake.input.vcf, force_bgz=True, reference_genome='GRCh38')
    
    # Import phenotype data
    logging.info(f"Importing phenotype data: {snakemake.input.phenotype}")
    phenotype_table = hl.import_table(snakemake.input.phenotype, 
                                      impute=True, 
                                      key='sample_id')
    
    # Annotate with phenotype
    mt = mt.annotate_cols(phenotype = phenotype_table[mt.s])
    
    # Sample QC
    logging.info("Running sample QC")
    mt = hl.sample_qc(mt)
    
    # Filter samples based on QC metrics
    logging.info("Filtering samples based on QC metrics")
    sample_qc_stats = mt.cols().select(mt.sample_qc)
    
    # Calculate sample QC thresholds
    qc_thresholds = {
        'call_rate': 0.95,  # Samples with less than 95% call rate are filtered
        'dp_stats_mean': 15,  # Samples with mean depth < 15 are filtered
        'r_ti_tv': (1.8, 2.2),  # Acceptable range for Ti/Tv ratio
        'r_het_hom_var': (1.0, 3.0)  # Acceptable range for het/hom ratio
    }
    
    # Apply sample filters
    mt = mt.filter_cols(
        (mt.sample_qc.call_rate >= qc_thresholds['call_rate']) &
        (mt.sample_qc.dp_stats.mean >= qc_thresholds['dp_stats_mean']) &
        (mt.sample_qc.r_ti_tv >= qc_thresholds['r_ti_tv'][0]) &
        (mt.sample_qc.r_ti_tv <= qc_thresholds['r_ti_tv'][1]) &
        (mt.sample_qc.r_het_hom_var >= qc_thresholds['r_het_hom_var'][0]) &
        (mt.sample_qc.r_het_hom_var <= qc_thresholds['r_het_hom_var'][1])
    )
    
    # Variant QC
    logging.info("Running variant QC")
    mt = hl.variant_qc(mt)
    
    # Filter variants based on QC metrics
    logging.info("Filtering variants based on QC metrics")
    
    # QC thresholds for variants
    variant_qc_thresholds = {
        'call_rate': 0.95,  # Variants with less than 95% call rate are filtered
        'p_value_hwe': 1e-6,  # Filter variants with HWE p-value < 1e-6
        'AC': 0  # Filter variants with AC=0 (monomorphic)
    }
    
    # Apply variant filters
    mt = mt.filter_rows(
        (mt.variant_qc.call_rate >= variant_qc_thresholds['call_rate']) &
        (mt.variant_qc.p_value_hwe >= variant_qc_thresholds['p_value_hwe']) &
        (mt.variant_qc.AC > variant_qc_thresholds['AC'])
    )
    
    # Save QC metrics for reporting
    logging.info("Saving QC metrics")
    sample_qc_metrics = mt.cols().select('sample_qc', 'phenotype').flatten()
    variant_qc_metrics = mt.rows().select('variant_qc').flatten()
    
    # Export sample QC metrics
    sample_qc_file = os.path.join(os.path.dirname(snakemake.output.qc_report), "sample_qc_metrics.tsv")
    sample_qc_metrics.export(sample_qc_file)
    
    # Export variant QC metrics
    variant_qc_file = os.path.join(os.path.dirname(snakemake.output.qc_report), "variant_qc_metrics.tsv")
    variant_qc_metrics.export(variant_qc_file)
    
    # Create QC visualizations
    logging.info("Creating QC visualizations")
    output_file(snakemake.output.qc_report)
    
    # Plot sample QC metrics
    p_call_rate = hl.plot.histogram(mt.sample_qc.call_rate, legend='Call Rate')
    p_dp = hl.plot.histogram(mt.sample_qc.dp_stats.mean, legend='Mean Depth')
    p_gq = hl.plot.histogram(mt.sample_qc.gq_stats.mean, legend='Mean GQ')
    p_ti_tv = hl.plot.histogram(mt.sample_qc.r_ti_tv, legend='Ti/Tv Ratio')
    p_het_hom = hl.plot.histogram(mt.sample_qc.r_het_hom_var, legend='Het/HomVar Ratio')
    
    # Plot variant QC metrics
    p_variant_call_rate = hl.plot.histogram(mt.variant_qc.call_rate, legend='Variant Call Rate')
    p_allele_count = hl.plot.histogram(mt.variant_qc.AC, legend='Allele Count')
    p_hwe = hl.plot.histogram(mt.variant_qc.p_value_hwe, legend='HWE p-value')
    
    # Create a layout for the plots
    plots = layout([
        [p_call_rate, p_dp, p_gq],
        [p_ti_tv, p_het_hom],
        [p_variant_call_rate, p_allele_count, p_hwe]
    ])
    
    # Save the plots to the output HTML file
    save(plots)
    
    # Write out the MatrixTable
    logging.info(f"Writing MatrixTable to {snakemake.output.mt}")
    mt.write(snakemake.output.mt, overwrite=True)
    
    logging.info("Hail import and QC completed successfully")

if __name__ == "__main__":
    main()


