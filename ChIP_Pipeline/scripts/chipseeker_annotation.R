#!/usr/bin/env Rscript
# This script performs ChIP-seq peak annotation using ChIPseeker
# and conducts downstream GO and pathway enrichment analysis

# Load required libraries
library(ChIPseeker)
library(TxDb.Hsapiens.UCSC.hg38.knownGene)  # Use appropriate TxDb for your organism
library(org.Hs.eg.db)  # Use appropriate org.db for your organism
library(clusterProfiler)
library(ReactomePA)
library(DOSE)
library(ggplot2)
library(dplyr)
library(tidyr)
library(GenomicRanges)
library(rtracklayer)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
peak_file <- args[1]  # Path to narrowPeak file
output_dir <- args[2]  # Output directory
organism <- args[3]   # Organism (default: human)
genome <- args[4]     # Genome version (default: hg38)

# Create output directory if it doesn't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Set up TxDb based on organism and genome
txdb <- NULL
orgdb <- NULL

if (organism == "human") {
  if (genome == "hg38") {
    library(TxDb.Hsapiens.UCSC.hg38.knownGene)
    txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
    orgdb <- "org.Hs.eg.db"
  } else if (genome == "hg19") {
    library(TxDb.Hsapiens.UCSC.hg19.knownGene)
    txdb <- TxDb.Hsapiens.UCSC.hg19.knownGene
    orgdb <- "org.Hs.eg.db"
  }
} else if (organism == "mouse") {
  if (genome == "mm10") {
    library(TxDb.Mmusculus.UCSC.mm10.knownGene)
    txdb <- TxDb.Mmusculus.UCSC.mm10.knownGene
    orgdb <- "org.Mm.eg.db"
  } else if (genome == "mm9") {
    library(TxDb.Mmusculus.UCSC.mm9.knownGene)
    txdb <- TxDb.Mmusculus.UCSC.mm9.knownGene
    orgdb <- "org.Mm.eg.db"
  }
}

if (is.null(txdb)) {
  stop("Unsupported organism/genome combination")
}

# Load peak file
peaks <- readPeakFile(peak_file)

# Generate name for output files based on input file
sample_name <- gsub(".narrowPeak$|.bed$", "", basename(peak_file))

# 1. Basic Peak Statistics
print("Generating basic peak statistics...")
pdf(file.path(output_dir, paste0(sample_name, "_peak_statistics.pdf")), width = 10, height = 8)
covplot(peaks, weightCol="V5")
print(plotCoverage(peaks, weightCol="V5"))
dev.off()

# 2. Peak Annotation with ChIPseeker
print("Annotating peaks with ChIPseeker...")
peakAnno <- annotatePeak(peaks, 
                        tssRegion=c(-3000, 3000),
                        TxDb=txdb, 
                        annoDb=orgdb)

# Save annotation results
anno_df <- as.data.frame(peakAnno)
write.csv(anno_df, file.path(output_dir, paste0(sample_name, "_peak_annotation.csv")), row.names = FALSE)

# Generate annotation plots
pdf(file.path(output_dir, paste0(sample_name, "_annotation_plots.pdf")), width = 10, height = 10)
# Annotation bar plot
print(plotAnnoBar(peakAnno))
# Venn pie chart
print(vennpie(peakAnno))
# Distance to TSS
print(plotDistToTSS(peakAnno))
dev.off()

# 3. GO Enrichment Analysis
print("Performing GO enrichment analysis...")
# Extract gene IDs from annotation
genes <- unique(anno_df$geneId)
genes <- genes[!is.na(genes)]

# Perform GO enrichment for Biological Process, Molecular Function, and Cellular Component
go_bp <- enrichGO(gene = genes,
                 OrgDb = orgdb,
                 keyType = "ENTREZID",
                 ont = "BP",
                 pAdjustMethod = "BH",
                 pvalueCutoff = 0.05,
                 qvalueCutoff = 0.1)

go_mf <- enrichGO(gene = genes,
                 OrgDb = orgdb,
                 keyType = "ENTREZID",
                 ont = "MF",
                 pAdjustMethod = "BH",
                 pvalueCutoff = 0.05,
                 qvalueCutoff = 0.1)

go_cc <- enrichGO(gene = genes,
                 OrgDb = orgdb,
                 keyType = "ENTREZID",
                 ont = "CC",
                 pAdjustMethod = "BH",
                 pvalueCutoff = 0.05,
                 qvalueCutoff = 0.1)

# Save GO results
if (nrow(go_bp@result) > 0) {
  write.csv(go_bp@result, file.path(output_dir, paste0(sample_name, "_GO_BP.csv")), row.names = FALSE)
}
if (nrow(go_mf@result) > 0) {
  write.csv(go_mf@result, file.path(output_dir, paste0(sample_name, "_GO_MF.csv")), row.names = FALSE)
}
if (nrow(go_cc@result) > 0) {
  write.csv(go_cc@result, file.path(output_dir, paste0(sample_name, "_GO_CC.csv")), row.names = FALSE)
}

# Plot GO results
pdf(file.path(output_dir, paste0(sample_name, "_GO_enrichment.pdf")), width = 12, height = 8)
if (nrow(go_bp@result) > 0) {
  print(dotplot(go_bp, title = "GO Biological Process Enrichment", showCategory = 20))
  print(barplot(go_bp, title = "GO Biological Process Enrichment", showCategory = 20))
  print(emapplot(go_bp, title = "GO Biological Process Network"))
}
if (nrow(go_mf@result) > 0) {
  print(dotplot(go_mf, title = "GO Molecular Function Enrichment", showCategory = 20))
  print(barplot(go_mf, title = "GO Molecular Function Enrichment", showCategory = 20))
}
if (nrow(go_cc@result) > 0) {
  print(dotplot(go_cc, title = "GO Cellular Component Enrichment", showCategory = 20))
  print(barplot(go_cc, title = "GO Cellular Component Enrichment", showCategory = 20))
}
dev.off()

# 4. KEGG Pathway Analysis
print("Performing KEGG pathway analysis...")
kegg <- enrichKEGG(gene = genes,
                  organism = ifelse(organism == "human", "hsa", "mmu"),
                  pAdjustMethod = "BH",
                  pvalueCutoff = 0.05,
                  qvalueCutoff = 0.1)

# Save KEGG results
if (!is.null(kegg) && nrow(kegg@result) > 0) {
  write.csv(kegg@result, file.path(output_dir, paste0(sample_name, "_KEGG_pathways.csv")), row.names = FALSE)
  
  # Plot KEGG results
  pdf(file.path(output_dir, paste0(sample_name, "_KEGG_enrichment.pdf")), width = 12, height = 8)
  print(dotplot(kegg, title = "KEGG Pathway Enrichment", showCategory = 20))
  print(barplot(kegg, title = "KEGG Pathway Enrichment", showCategory = 20))
  if (nrow(kegg@result) >= 5) {
    print(emapplot(kegg, title = "KEGG Pathway Network"))
  }
  dev.off()
}

# 5. Reactome Pathway Analysis
print("Performing Reactome pathway analysis...")
reactome <- enrichPathway(gene = genes,
                         organism = ifelse(organism == "human", "human", "mouse"),
                         pAdjustMethod = "BH",
                         pvalueCutoff = 0.05,
                         qvalueCutoff = 0.1)

# Save Reactome results
if (!is.null(reactome) && nrow(reactome@result) > 0) {
  write.csv(reactome@result, file.path(output_dir, paste0(sample_name, "_Reactome_pathways.csv")), row.names = FALSE)
  
  # Plot Reactome results
  pdf(file.path(output_dir, paste0(sample_name, "_Reactome_enrichment.pdf")), width = 12, height = 8)
  print(dotplot(reactome, title = "Reactome Pathway Enrichment", showCategory = 20))
  print(barplot(reactome, title = "Reactome Pathway Enrichment", showCategory = 20))
  if (nrow(reactome@result) >= 5) {
    print(emapplot(reactome, title = "Reactome Pathway Network"))
  }
  dev.off()
}

# 6. Prepare ChIPseeker visualization tag cloud
print("Generating visualization tag cloud...")
genes_df <- bitr(genes, fromType = "ENTREZID", toType = c("SYMBOL", "GENENAME"), OrgDb = orgdb)
if (!is.null(genes_df) && nrow(genes_df) > 0) {
  # Save gene list with symbols and descriptions
  write.csv(genes_df, file.path(output_dir, paste0(sample_name, "_target_genes.csv")), row.names = FALSE)
}

print("ChIPseeker annotation and functional analysis completed successfully!")


