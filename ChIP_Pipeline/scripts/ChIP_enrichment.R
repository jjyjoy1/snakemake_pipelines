#!/usr/bin/env Rscript
# Script for functional enrichment analysis of ChIP-seq peaks
# Takes peak files and annotation results as input

# Load required libraries
suppressPackageStartupMessages({
  library(clusterProfiler)
  library(ReactomePA)
  library(DOSE)
  library(enrichplot)
  library(ggplot2)
  library(org.Hs.eg.db)
  library(org.Mm.eg.db)
  library(pathview)
  library(dplyr)
  library(tidyr)
  library(rtracklayer)
  library(GenomicRanges)
})

# Get command line arguments from Snakemake
peak_file <- snakemake@input[["peaks"]]  # Path to peaks file
anno_file <- snakemake@input[["anno"]]   # Path to ChIPseeker annotation file
output_dir <- snakemake@params[["outdir"]]  # Output directory
organism <- snakemake@params[["organism"]]  # Organism (human or mouse)

# Create output directory if it doesn't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Set sample name for output files
sample_name <- gsub(".narrowPeak$|.bed$", "", basename(peak_file))

# Load annotation data
anno_data <- read.csv(anno_file)

# Get gene IDs from annotation file
gene_ids <- unique(anno_data$geneId)
gene_ids <- gene_ids[!is.na(gene_ids)]

# Check if we have enough genes for analysis
if (length(gene_ids) < 10) {
  warning("Less than 10 genes found. Functional enrichment may not be reliable.")
}

# Set the appropriate organism database
org_db <- switch(organism,
                human = org.Hs.eg.db,
                mouse = org.Mm.eg.db,
                org.Hs.eg.db)  # default to human

org_kegg <- switch(organism,
                  human = "hsa",
                  mouse = "mmu",
                  "hsa")  # default to human

# Convert gene IDs to gene symbols for better interpretation
gene_symbols <- NA
tryCatch({
  gene_symbols <- mapIds(org_db, 
                       keys = gene_ids, 
                       column = "SYMBOL", 
                       keytype = "ENTREZID", 
                       multiVals = "first")
}, error = function(e) {
  warning("Error converting gene IDs to symbols: ", e$message)
})

# Save gene list
genes_df <- data.frame(
  ENTREZID = gene_ids,
  SYMBOL = gene_symbols
)
write.csv(genes_df, file.path(output_dir, paste0(sample_name, "_target_genes.csv")), row.names = FALSE)

##########################
# GO Enrichment Analysis #
##########################
print("Performing GO enrichment analysis...")

# Biological Process
go_bp <- enrichGO(
  gene = gene_ids,
  OrgDb = org_db,
  ont = "BP",
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.1
)

# Simplify GO terms to reduce redundancy
if (!is.null(go_bp) && nrow(go_bp@result) > 10) {
  go_bp <- simplify(go_bp, cutoff = 0.7, by = "p.adjust", select_fun = min)
}

# Save results
write.csv(go_bp@result, file.path(output_dir, paste0(sample_name, "_GO_BP.csv")), row.names = FALSE)

# Molecular Function
go_mf <- enrichGO(
  gene = gene_ids,
  OrgDb = org_db,
  ont = "MF",
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.1
)

# Simplify GO terms
if (!is.null(go_mf) && nrow(go_mf@result) > 10) {
  go_mf <- simplify(go_mf, cutoff = 0.7, by = "p.adjust", select_fun = min)
}

# Save results
write.csv(go_mf@result, file.path(output_dir, paste0(sample_name, "_GO_MF.csv")), row.names = FALSE)

# Cellular Component
go_cc <- enrichGO(
  gene = gene_ids,
  OrgDb = org_db,
  ont = "CC",
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.1
)

# Simplify GO terms
if (!is.null(go_cc) && nrow(go_cc@result) > 10) {
  go_cc <- simplify(go_cc, cutoff = 0.7, by = "p.adjust", select_fun = min)
}

# Save results
write.csv(go_cc@result, file.path(output_dir, paste0(sample_name, "_GO_CC.csv")), row.names = FALSE)

# Plot GO results
print("Creating GO enrichment plots...")
pdf(file.path(output_dir, paste0(sample_name, "_GO_enrichment.pdf")), width = 12, height = 10)

# Plot for Biological Process
if (!is.null(go_bp) && nrow(go_bp@result) > 0) {
  print(dotplot(go_bp, showCategory = 15, title = "GO Biological Process"))
  print(barplot(go_bp, showCategory = 15, title = "GO Biological Process"))
  
  # Create GO network if enough terms
  if (nrow(go_bp@result) >= 5) {
    print(emapplot(go_bp, showCategory = 30))
    print(cnetplot(go_bp, categorySize = "pvalue", foldChange = gene_symbols))
  }
}

# Plot for Molecular Function
if (!is.null(go_mf) && nrow(go_mf@result) > 0) {
  print(dotplot(go_mf, showCategory = 15, title = "GO Molecular Function"))
  print(barplot(go_mf, showCategory = 15, title = "GO Molecular Function"))
  
  # Create GO network if enough terms
  if (nrow(go_mf@result) >= 5) {
    print(emapplot(go_mf, showCategory = 30))
    print(cnetplot(go_mf, categorySize = "pvalue", foldChange = gene_symbols))
  }
}

# Plot for Cellular Component
if (!is.null(go_cc) && nrow(go_cc@result) > 0) {
  print(dotplot(go_cc, showCategory = 15, title = "GO Cellular Component"))
  print(barplot(go_cc, showCategory = 15, title = "GO Cellular Component"))
  
  # Create GO network if enough terms
  if (nrow(go_cc@result) >= 5) {
    print(emapplot(go_cc, showCategory = 30))
    print(cnetplot(go_cc, categorySize = "pvalue", foldChange = gene_symbols))
  }
}

dev.off()

##########################
# KEGG Pathway Analysis  #
##########################
print("Performing KEGG pathway analysis...")
kegg <- enrichKEGG(
  gene = gene_ids,
  organism = org_kegg,
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.1
)

# Save KEGG results
if (!is.null(kegg) && nrow(kegg@result) > 0) {
  write.csv(kegg@result, file.path(output_dir, paste0(sample_name, "_KEGG_pathways.csv")), row.names = FALSE)
  
  # Plot KEGG results
  pdf(file.path(output_dir, paste0(sample_name, "_KEGG_enrichment.pdf")), width = 12, height = 8)
  print(dotplot(kegg, showCategory = 15, title = "KEGG Pathway"))
  print(barplot(kegg, showCategory = 15, title = "KEGG Pathway"))
  
  # Create pathway network if enough terms
  if (nrow(kegg@result) >= 5) {
    print(emapplot(kegg, showCategory = 30))
    print(cnetplot(kegg, categorySize = "pvalue", foldChange = gene_symbols))
  }
  
  # Generate pathway diagrams for top pathways
  if (nrow(kegg@result) > 0) {
    top_pathways <- head(kegg@result$ID, 5)
    for (pathway in top_pathways) {
      tryCatch({
        pathview(gene.data = gene_ids, pathway.id = pathway, species = org_kegg)
        # Copy the generated PNG file to output directory
        pathway_file <- paste0(pathway, ".pathview.png")
        if (file.exists(pathway_file)) {
          file.copy(pathway_file, file.path(output_dir, paste0(sample_name, "_", pathway_file)))
          file.remove(pathway_file)
        }
      }, error = function(e) {
        warning("Error generating pathway view for ", pathway, ": ", e$message)
      })
    }
  }
  dev.off()
}

#############################
# Reactome Pathway Analysis #
#############################
print("Performing Reactome pathway analysis...")
reactome <- enrichPathway(
  gene = gene_ids,
  organism = ifelse(organism == "human", "human", "mouse"),
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.1
)

# Save Reactome results
if (!is.null(reactome) && nrow(reactome@result) > 0) {
  write.csv(reactome@result, file.path(output_dir, paste0(sample_name, "_Reactome_pathways.csv")), row.names = FALSE)
  
  # Plot Reactome results
  pdf(file.path(output_dir, paste0(sample_name, "_Reactome_enrichment.pdf")), width = 12, height = 8)
  print(dotplot(reactome, showCategory = 15, title = "Reactome Pathway"))
  print(barplot(reactome, showCategory = 15, title = "Reactome Pathway"))
  
  # Create pathway network if enough terms
  if (nrow(reactome@result) >= 5) {
    print(emapplot(reactome, showCategory = 30))
    print(cnetplot(reactome, categorySize = "pvalue", foldChange = gene_symbols))
  }
  dev.off()
}

#############################
# Disease Ontology Analysis #
#############################
print("Performing Disease Ontology analysis...")
do <- enrichDO(
  gene = gene_ids,
  ont = "DO",
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.1,
  readable = TRUE
)

# Save Disease Ontology results
if (!is.null(do) && nrow(do@result) > 0) {
  write.csv(do@result, file.path(output_dir, paste0(sample_name, "_Disease_Ontology.csv")), row.names = FALSE)
  
  # Plot Disease Ontology results
  pdf(file.path(output_dir, paste0(sample_name, "_Disease_Ontology.pdf")), width = 12, height = 8)
  print(dotplot(do, showCategory = 15, title = "Disease Ontology"))
  print(barplot(do, showCategory = 15, title = "Disease Ontology"))
  
  # Create disease network if enough terms
  if (nrow(do@result) >= 5) {
    print(emapplot(do, showCategory = 30))
    print(cnetplot(do, categorySize = "pvalue", foldChange = gene_symbols))
  }
  dev.off()
}

#######################################
# Transcription Factor Enrichment     #
#######################################
if (organism == "human") {
  print("Performing transcription factor enrichment analysis...")
  tryCatch({
    # Try using enricher with MSigDB C3 collection if available
    msigdb_file <- file.path(output_dir, "c3.tft.v7.4.entrez.gmt")
    
    # If MSigDB file doesn't exist locally, try to download it
    if (!file.exists(msigdb_file)) {
      msigdb_url <- "http://software.broadinstitute.org/gsea/msigdb/download_file.jsp?filePath=/resources/msigdb/7.4/c3.tft.v7.4.entrez.gmt"
      tryCatch({
        download.file(msigdb_url, msigdb_file)
      }, error = function(e) {
        warning("Could not download MSigDB file: ", e$message)
        msigdb_file <- NULL
      })
    }
    
    # If we have the MSigDB file, use it for TF enrichment
    if (!is.null(msigdb_file) && file.exists(msigdb_file)) {
      msigdb <- read.gmt(msigdb_file)
      tf_enrich <- enricher(
        gene = gene_ids,
        TERM2GENE = msigdb,
        pAdjustMethod = "BH",
        pvalueCutoff = 0.05,
        qvalueCutoff = 0.1
      )
      
      if (!is.null(tf_enrich) && nrow(tf_enrich@result) > 0) {
        write.csv(tf_enrich@result, file.path(output_dir, paste0(sample_name, "_TF_enrichment.csv")), row.names = FALSE)
        
        # Plot TF enrichment results
        pdf(file.path(output_dir, paste0(sample_name, "_TF_enrichment.pdf")), width = 12, height = 8)
        print(dotplot(tf_enrich, showCategory = 15, title = "Transcription Factor Enrichment"))
        print(barplot(tf_enrich, showCategory = 15, title = "Transcription Factor Enrichment"))
        dev.off()
      }
    } else {
      warning("MSigDB file not available for TF enrichment")
    }
  }, error = function(e) {
    warning("Error in transcription factor enrichment analysis: ", e$message)
  })
}

#################################
# Comprehensive Summary Report  #
#################################
print("Creating summary report...")
summary_file <- file.path(output_dir, paste0(sample_name, "_functional_analysis_summary.txt"))

sink(summary_file)
cat("Functional Analysis Summary for ", sample_name, "\n")
cat("==============================================\n\n")

cat("Number of target genes: ", length(gene_ids), "\n\n")

cat("GO Biological Process Enrichment:\n")
if (!is.null(go_bp) && nrow(go_bp@result) > 0) {
  cat("Top 10 enriched terms:\n")
  print(head(go_bp@result[, c("ID", "Description", "pvalue", "p.adjust", "Count")], 10))
} else {
  cat("No significant enrichment found.\n")
}
cat("\n")

cat("GO Molecular Function Enrichment:\n")
if (!is.null(go_mf) && nrow(go_mf@result) > 0) {
  cat("Top 10 enriched terms:\n")
  print(head(go_mf@result[, c("ID", "Description", "pvalue", "p.adjust", "Count")], 10))
} else {
  cat("No significant enrichment found.\n")
}
cat("\n")

cat("GO Cellular Component Enrichment:\n")
if (!is.null(go_cc) && nrow(go_cc@result) > 0) {
  cat("Top 10 enriched terms:\n")
  print(head(go_cc@result[, c("ID", "Description", "pvalue", "p.adjust", "Count")], 10))
} else {
  cat("No significant enrichment found.\n")
}
cat("\n")

cat("KEGG Pathway Enrichment:\n")
if (!is.null(kegg) && nrow(kegg@result) > 0) {
  cat("Top 10 enriched pathways:\n")
  print(head(kegg@result[, c("ID", "Description", "pvalue", "p.adjust", "Count")], 10))
} else {
  cat("No significant enrichment found.\n")
}
cat("\n")

cat("Reactome Pathway Enrichment:\n")
if (!is.null(reactome) && nrow(reactome@result) > 0) {
  cat("Top 10 enriched pathways:\n")
  print(head(reactome@result[, c("ID", "Description", "pvalue", "p.adjust", "Count")], 10))
} else {
  cat("No significant enrichment found.\n")
}
cat("\n")

cat("Disease Ontology Enrichment:\n")
if (!is.null(do) && nrow(do@result) > 0) {
  cat("Top 10 enriched terms:\n")
  print(head(do@result[, c("ID", "Description", "pvalue", "p.adjust", "Count")], 10))
} else {
  cat("No significant enrichment found.\n")
}
cat("\n")

sink()

print("Functional analysis completed successfully!")
