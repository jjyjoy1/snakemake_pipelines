# scripts/hail_vep_annotate.py
import hail as hl
import os
import logging

# Configure logging
logging.basicConfig(filename=snakemake.log[0], level=logging.INFO,
                   format='%(asctime)s %(levelname)s %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

def main():
    # Initialize Hail
    hl.init(log=snakemake.log[0])
    
    # Read the MatrixTable
    logging.info(f"Reading MatrixTable from {snakemake.input.mt}")
    mt = hl.read_matrix_table(snakemake.input.mt)
    
    # Get VEP configuration from the config file
    vep_config = snakemake.input.vep_config
    
    # Run VEP
    logging.info("Running VEP annotation")
    mt = hl.vep(mt, config=vep_config)
    
    # Parse VEP annotations
    logging.info("Parsing VEP annotations")
    
    # Create structured fields from VEP annotations
    mt = mt.annotate_rows(
        vep_parsed = hl.struct(
            most_severe_consequence = mt.vep.most_severe_consequence,
            gene_symbol = mt.vep.transcript_consequences.map(lambda tc: tc.gene_symbol).find(lambda x: x.length() > 0),
            gene_id = mt.vep.transcript_consequences.map(lambda tc: tc.gene_id).find(lambda x: x.length() > 0),
            impact = mt.vep.transcript_consequences.map(lambda tc: tc.impact).min(),
            lof = mt.vep.transcript_consequences.map(lambda tc: tc.lof).find(lambda x: x == "HC"),
            lof_filter = mt.vep.transcript_consequences.map(lambda tc: tc.lof_filter).find(lambda x: x.length() > 0),
            lof_flags = mt.vep.transcript_consequences.map(lambda tc: tc.lof_flags).find(lambda x: x.length() > 0)
        )
    )
    
    # Filter for variants with HIGH or MODERATE impact
    logging.info("Filtering for variants with HIGH or MODERATE impact")
    mt = mt.filter_rows(
        (mt.vep_parsed.impact == "HIGH") | 
        (mt.vep_parsed.impact == "MODERATE")
    )
    
    # Add a derived annotation for loss-of-function variants
    mt = mt.annotate_rows(
        is_lof = hl.or_else(mt.vep_parsed.lof == "HC", False)
    )
    
    # Export a list of affected genes for reference
    affected_genes = mt.rows().select('vep_parsed').distinct()
    affected_genes_file = os.path.join(os.path.dirname(snakemake.output.mt), "affected_genes.tsv")
    affected_genes.export(affected_genes_file)
    
    # Write out the annotated MatrixTable
    logging.info(f"Writing VEP-annotated MatrixTable to {snakemake.output.mt}")
    mt.write(snakemake.output.mt, overwrite=True)
    
    logging.info("VEP annotation completed successfully")

if __name__ == "__main__":
    main()



