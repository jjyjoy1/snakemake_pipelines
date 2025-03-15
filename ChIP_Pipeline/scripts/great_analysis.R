#!/usr/bin/env Rscript
# This script prepares ChIP-seq peak files for GREAT analysis
# and processes the results from GREAT API

# Load required libraries
library(rGREAT)
library(GenomicRanges)
library(rtracklayer)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(jsonlite)
library(httr)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
peak_file <- args[1]  # Path to narrowPeak file
output_dir <- args[2]  # Output directory
species <- args[3]     # Species (default: hg38)
background_type <- args[4]  # Background type (default: whole genome)

# Create output directory if it doesn't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Generate name for output files based on input file
sample_name <- gsub(".narrowPeak$|.bed$", "", basename(peak_file))

# Load peak file
print("Loading peak file...")
if (grepl(".narrowPeak$", peak_file)) {
  peaks <- read.table(peak_file, header = FALSE)
  peaks <- GRanges(
    seqnames = peaks$V1,
    ranges = IRanges(start = peaks$V2, end = peaks$V3),
    strand = "*",
    score = peaks$V5
  )
} else {
  peaks <- import(peak_file)
}

# Ensure we have at least 500 peaks for GREAT analysis
# GREAT recommends a minimum of 500 regions
print(paste("Number of peaks:", length(peaks)))
if (length(peaks) < 500) {
  warning("Less than 500 peaks provided. GREAT analysis may not be reliable.")
}

# Prepare for GREAT analysis
print("Submitting to GREAT...")
# Set up species genome
genome <- switch(species,
                "hg38" = "hg38",
                "hg19" = "hg19",
                "mm10" = "mm10",
                "mm9" = "mm9",
                "hg38")  # default

# Set up background
bg <- switch(background_type,
           "wholeGenome" = "wholeGenome",
           "regulatoryDomains" = "regulatoryDomains",
           "wholeGenome")  # default

# Run GREAT analysis using rGREAT
job <- submitGreatJob(peaks, 
                     species = genome,
                     includeCuratedRegDoms = TRUE,
                     rule = "basalPlusExt",
                     adv_upstream = 5.0,  # 5kb upstream
                     adv_downstream = 1.0,  # 1kb downstream
                     adv_span = 1000.0,  # max 1000kb extension
                     bgChoice = bg)

# Wait for job to complete
print("Waiting for GREAT analysis to complete...")
tryCatch({
  # Try to get results with a timeout
  tb_go_bp <- getEnrichmentTables(job, category = "GO Biological Process")
  print("GREAT analysis completed!")
}, error = function(e) {
  print(paste("Error in GREAT analysis:", e$message))
  # Create a backup approach using direct API if needed (commented out for reference)
  # Would require additional implementation for direct API access
})

# Retrieve results
print("Retrieving GREAT results...")
# Get GO Biological Process results
tb_go_bp <- getEnrichmentTables(job, category = "GO Biological Process")
# Get GO Molecular Function results
tb_go_mf <- getEnrichmentTables(job, category = "GO Molecular Function")
# Get GO Cellular Component results
tb_go_cc <- getEnrichmentTables(job, category = "GO Cellular Component")
# Get MSigDB Pathway results
tb_msigdb <- getEnrichmentTables(job, category = "MSigDB Pathway")
# Get PANTHER Pathway results
tb_panther <- getEnrichmentTables(job, category = "PANTHER Pathway")

# Save results to files
write.csv(tb_go_bp, file.path(output_dir, paste0(sample_name, "_GREAT_GO_BP.csv")), row.names = FALSE)
write.csv(tb_go_mf, file.path(output_dir, paste0(sample_name, "_GREAT_GO_MF.csv")), row.names = FALSE)
write.csv(tb_go_cc, file.path(output_dir, paste0(sample_name, "_GREAT_GO_CC.csv")), row.names = FALSE)
write.csv(tb_msigdb, file.path(output_dir, paste0(sample_name, "_GREAT_MSigDB.csv")), row.names = FALSE)
write.csv(tb_panther, file.path(output_dir, paste0(sample_name, "_GREAT_PANTHER.csv")), row.names = FALSE)

# Plot top results
pdf(file.path(output_dir, paste0(sample_name, "_GREAT_results.pdf")), width = 12, height = 10)

# Function to create a dotplot from GREAT results
create_great_dotplot <- function(data, title, top_n = 20) {
  if (nrow(data) > 0) {
    # Take top results
    data <- data[1:min(top_n, nrow(data)),]
    
    # Create a ggplot2 dotplot
    data$Term <- factor(data$name, levels = rev(data$name))
    
    ggplot(data, aes(x = -log10(Hyper_Adjp_BH), y = Term)) +
      geom_point(aes(size = Hyper_Total_Hits / Hyper_Total_Genes, color = -log10(Hyper_Adjp_BH))) +
      scale_color_gradient(low = "blue", high = "red") +
      labs(title = title,
           x = "-log10(Adjusted p-value)",
           y = "",
           size = "Region Fold Enrichment") +
      theme_minimal() +
      theme(axis.text.y = element_text(size = 8))
  }
}

# Create and print plots
if (nrow(tb_go_bp) > 0) {
  print(create_great_dotplot(tb_go_bp, "GREAT GO Biological Process"))
}
if (nrow(tb_go_mf) > 0) {
  print(create_great_dotplot(tb_go_mf, "GREAT GO Molecular Function"))
}
if (nrow(tb_go_cc) > 0) {
  print(create_great_dotplot(tb_go_cc, "GREAT GO Cellular Component"))
}
if (nrow(tb_msigdb) > 0) {
  print(create_great_dotplot(tb_msigdb, "GREAT MSigDB Pathway"))
}
if (nrow(tb_panther) > 0) {
  print(create_great_dotplot(tb_panther, "GREAT PANTHER Pathway"))
}

# Plot region-gene associations
assoc <- plotRegionGeneAssociationGraphs(job)
print(assoc$dist_graph)
print(assoc$assoc_graph)

dev.off()

# Get genes associated with regions
print("Retrieving GREAT gene associations...")
gene_assoc <- getRegionGeneAssociations(job)

# Save gene associations to file
gene_assoc_df <- as.data.frame(gene_assoc)
write.csv(gene_assoc_df, file.path(output_dir, paste0(sample_name, "_GREAT_gene_associations.csv")), row.names = FALSE)

# Create a BED file of regions and associated genes for visualization
bed_output <- data.frame(
  chrom = as.character(seqnames(gene_assoc)),
  start = start(gene_assoc) - 1,  # Convert to 0-based for BED format
  end = end(gene_assoc),
  name = gene_assoc$gene
)

# Write BED file
write.table(bed_output, 
           file.path(output_dir, paste0(sample_name, "_GREAT_region_gene_associations.bed")),
           quote = FALSE, sep = "\t", row.names = FALSE, col.names = FALSE)

print("GREAT analysis completed successfully!")

