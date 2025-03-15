# methylation_analysis.R
# Script for differential methylation analysis using methylKit

# Load required libraries
suppressPackageStartupMessages({
  library(methylKit)
  library(genomation)
  library(GenomicRanges)
  library(ggplot2)
  library(reshape2)
  library(rmarkdown)
})

# Get file paths from snakemake
covFiles <- snakemake@input[["covs"]]
sample_groups <- snakemake@params[["sample_groups"]]
dmr_params <- snakemake@params[["dmr_parameters"]]
outdir <- snakemake@params[["outdir"]]
dmrsFile <- snakemake@output[["dmrs"]]
reportFile <- snakemake@output[["report"]]

# Create output directory if it doesn't exist
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# Extract sample names from file paths
sample_names <- gsub("_bismark_bt2_pe.deduplicated.bismark.cov.gz", "", basename(covFiles))

# Determine treatment groups
treatment <- rep(0, length(sample_names))
for (i in 1:length(sample_names)) {
  for (group_name in names(sample_groups)) {
    if (sample_names[i] %in% sample_groups[[group_name]]) {
      if (group_name == "treatment") {
        treatment[i] <- 1
      }
    }
  }
}

# Read methylation data
methylData <- methRead(
  covFiles,
  sample.id = sample_names,
  treatment = treatment,
  assembly = "genome",
  context = "CpG",
  mincov = dmr_params$min_coverage
)

# Filter low coverage sites
methylFiltered <- filterByCoverage(
  methylData,
  lo.count = dmr_params$min_coverage,
  lo.perc = NULL,
  hi.count = NULL,
  hi.perc = 99.9
)

# Normalize coverage
methylNormalized <- normalizeCoverage(methylFiltered)

# Merge samples
methylMerged <- unite(methylNormalized)

# Calculate differential methylation
myDiff <- calculateDiffMeth(
  methylMerged,
  overdispersion = "MN",
  test = "fisher"
)

# Select significantly differential methylated regions
myDiffSig <- getMethylDiff(
  myDiff,
  difference = dmr_params$min_diff,
  qvalue = dmr_params$qvalue
)

# Extract hypermethylated and hypomethylated regions
hyper <- getMethylDiff(
  myDiff,
  difference = dmr_params$min_diff,
  qvalue = dmr_params$qvalue,
  type = "hyper"
)

hypo <- getMethylDiff(
  myDiff,
  difference = dmr_params$min_diff,
  qvalue = dmr_params$qvalue,
  type = "hypo"
)

# Write differential methylation results to CSV
write.csv(getData(myDiffSig), file = dmrsFile, row.names = FALSE)

# Create annotation objects for visualization
diffAnn <- annotateWithGeneParts(
  as(myDiffSig, "GRanges"),
  gene.obj = NULL  # You would provide a TxDb object here for a real annotation
)

# Create markdown report content
report_content <- c(
  "---",
  "title: \"Differential Methylation Analysis Report\"",
  "date: \"`r Sys.Date()`\"",
  "output: html_document",
  "---",
  "",
  "```{r setup, include=FALSE}",
  "knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)",
  "```",
  "",
  "## Summary of Differential Methylation Analysis",
  "",
  "This report summarizes the differential methylation analysis results.",
  "",
  "### Sample Information",
  "",
  paste("Total samples analyzed:", length(sample_names)),
  "",
  "```{r}",
  "# Create a table of sample groups",
  "sample_df <- data.frame(",
  "  Sample = sample_names,",
  "  Group = ifelse(treatment == 1, 'Treatment', 'Control')",
  ")",
  "knitr::kable(sample_df)",
  "```",
  "",
  "### Differential Methylation Results",
  "",
  paste("Minimum coverage threshold:", dmr_params$min_coverage),
  paste("Minimum methylation difference (%):", dmr_params$min_diff),
  paste("Q-value threshold:", dmr_params$qvalue),
  "",
  paste("Total differentially methylated CpGs:", nrow(myDiffSig)),
  paste("Hypermethylated CpGs:", nrow(hyper)),
  paste("Hypomethylated CpGs:", nrow(hypo)),
  "",
  "```{r}",
  "# Create a plot of methylation percentages",
  "percMethylTable <- percMethylation(methylMerged)",
  "percMethylMatrix <- as.matrix(percMethylTable)",
  "heatmap_data <- percMethylMatrix[sample(1:nrow(percMethylMatrix), min(1000, nrow(percMethylMatrix))),]",
  "heatmap(heatmap_data, col = colorRampPalette(c('blue', 'white', 'red'))(100),",
  "        main = 'Methylation Percentage Heatmap (Subset)',",
  "        Colv = NA, labRow = FALSE)",
  "```",
  "",
  "```{r}",
  "# Plot methylation level distribution",
  "methylData_melt <- melt(percMethylTable)",
  "ggplot(methylData_melt, aes(x = value, fill = variable)) +",
  "  geom_density(alpha = 0.5) +",
  "  theme_minimal() +",
  "  labs(x = 'Methylation %', y = 'Density', title = 'Methylation Level Distribution')",
  "```",
  "",
  "```{r}",
  "# Plot the number of differentially methylated CpGs",
  "dmr_counts <- data.frame(",
  "  Type = c('Hypermethylated', 'Hypomethylated'),",
  "  Count = c(nrow(hyper), nrow(hypo))",
  ")",
  "ggplot(dmr_counts, aes(x = Type, y = Count, fill = Type)) +",
  "  geom_bar(stat = 'identity') +",
  "  theme_minimal() +",
  "  labs(title = 'Differentially Methylated CpGs')",
  "```",
  "",
  "### Top Differentially Methylated CpGs",
  "",
  "```{r}",
  "# Show the top differential methylation positions",
  "top_dmrs <- getData(myDiffSig)[order(abs(getData(myDiffSig)$meth.diff), decreasing = TRUE), ]",
  "top_n_dmrs <- head(top_dmrs, 20)",
  "knitr::kable(top_n_dmrs)",
  "```"
)

# Write the report Rmd file
rmd_file <- file.path(outdir, "differential_methylation_report.Rmd")
writeLines(report_content, rmd_file)

# Render the report
rmarkdown::render(rmd_file, output_file = reportFile)

# Print completion message
cat("Differential methylation analysis complete\n")


