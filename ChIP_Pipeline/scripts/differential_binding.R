# scripts/differential_binding.R - Script for differential binding analysis

library(DiffBind)
library(ggplot2)
library(tidyverse)
library(ComplexHeatmap)

# Create output directory for plots
dir.create(snakemake@output[["plots"]], showWarnings = FALSE, recursive = TRUE)

# Read manifest file
manifest <- read.csv(snakemake@params[["manifest"]], sep="\t")

# Create sample sheet for DiffBind
samples <- data.frame(
  SampleID = manifest$sample_id,
  Condition = manifest$condition,
  Replicate = manifest$replicate,
  bamReads = file.path("results/filtered", paste0(manifest$sample_id, ".filtered.bam")),
  Peaks = NA,
  PeakCaller = "macs",
  ScoreCol = "qvalue",
  LowerBetter = TRUE
)

# Add peak files for treatment samples
for (i in 1:nrow(samples)) {
  if (manifest$sample_type[i] == "treatment") {
    # Find corresponding control
    control <- manifest$sample_id[manifest$condition == manifest$condition[i] & 
                                 manifest$sample_type == "control" & 
                                 manifest$replicate == manifest$replicate[i]]
    
    if (length(control) > 0) {
      samples$Peaks[i] <- file.path("results/peaks", 
                                   paste0(manifest$sample_id[i], "_vs_", 
                                         control, "_peaks.narrowPeak"))
    }
  }
}

# Remove rows with NA peaks
samples <- samples[!is.na(samples$Peaks),]

# Create DiffBind object
dba_obj <- dba(sampleSheet = samples)

# Count reads
dba_obj <- dba.count(dba_obj, summits = 250)

# Normalize
dba_obj <- dba.normalize(dba_obj)

# Plot correlation heatmap
pdf(file.path(snakemake@output[["plots"]], "correlation_heatmap.pdf"), width = 8, height = 8)
dba.plotHeatmap(dba_obj)
dev.off()

# PCA plot
pdf(file.path(snakemake@output[["plots"]], "pca_plot.pdf"), width = 8, height = 6)
dba.plotPCA(dba_obj, DBA_CONDITION, label = DBA_ID)
dev.off()

# Define contrasts based on conditions
unique_conditions <- unique(samples$Condition)
if (length(unique_conditions) > 1) {
  for (i in 1:(length(unique_conditions)-1)) {
    for (j in (i+1):length(unique_conditions)) {
      dba_obj <- dba.contrast(dba_obj, 
                             group1 = unique_conditions[i],
                             group2 = unique_conditions[j],
                             name1 = unique_conditions[i],
                             name2 = unique_conditions[j])
    }
  }
  
  # Perform differential binding analysis
  dba_obj <- dba.analyze(dba_obj, method = DBA_DESEQ2)
  
  # Extract results
  all_results <- list()
  for (i in 1:length(dba_obj$contrasts)) {
    report <- dba.report(dba_obj, contrast = i, th = snakemake@params[["fdr"]])
    
    if (nrow(report) > 0) {
      df <- as.data.frame(report)
      df$contrast <- paste0(dba_obj$contrasts[[i]]$name1, "_vs_", dba_obj$contrasts[[i]]$name2)
      all_results[[i]] <- df
      
      # MA plot
      pdf(file.path(snakemake@output[["plots"]], 
                   paste0("ma_plot_", dba_obj$contrasts[[i]]$name1, "_vs_", 
                         dba_obj$contrasts[[i]]$name2, ".pdf")), 
          width = 8, height = 6)
      dba.plotMA(dba_obj, contrast = i)
      dev.off()
      
      # Volcano plot
      pdf(file.path(snakemake@output[["plots"]], 
                   paste0("volcano_plot_", dba_obj$contrasts[[i]]$name1, "_vs_", 
                         dba_obj$contrasts[[i]]$name2, ".pdf")), 
          width = 8, height = 6)
      dba.plotVolcano(dba_obj, contrast = i)
      dev.off()
    }
  }
  
  # Combine all results
  if (length(all_results) > 0) {
    combined_results <- do.call(rbind, all_results)
    write.csv(combined_results, snakemake@output[["csv"]], row.names = FALSE)
  } else {
    # Create empty results file if no significant differences
    empty_df <- data.frame(Chr = character(), Start = integer(), End = integer(), 
                          Fold = numeric(), p.value = numeric(), FDR = numeric(),
                          contrast = character())
    write.csv(empty_df, snakemake@output[["csv"]], row.names = FALSE)
  }
} else {
  # Create empty results file if there's only one condition
  empty_df <- data.frame(Chr = character(), Start = integer(), End = integer(), 
                        Fold = numeric(), p.value = numeric(), FDR = numeric(),
                        contrast = character())
  write.csv(empty_df, snakemake@output[["csv"]], row.names = FALSE)
}

# Save DiffBind object
save(dba_obj, file = snakemake@output[["rdata"]])
