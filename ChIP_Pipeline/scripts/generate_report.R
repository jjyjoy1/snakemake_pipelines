# scripts/generate_report.R - Script to generate final report

library(rmarkdown)
library(knitr)
library(DT)
library(ggplot2)
library(dplyr)

# Create a temporary R Markdown file for the report
report_rmd <- tempfile(fileext = ".Rmd")

cat('---
title: "ChIP-seq Analysis Report"
date: "`r format(Sys.time(), "%d %B, %Y")`"
output:
  html_document:
    theme: cosmo
    toc: true
    toc_float: true
    code_folding: hide
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)
library(DT)
library(ggplot2)
library(dplyr)
```

## Overview

This report summarizes the results of ChIP-seq data analysis performed using a Snakemake pipeline.

## Quality Control

The quality of raw reads was assessed using FastQC. Below is a summary of key metrics:

```{r qc, echo=FALSE}
# Parse FastQC results
fastqc_files <- list.files("results/fastqc", pattern = "*_fastqc.zip", full.names = TRUE)
qc_summary <- data.frame(Sample = character(),
                         TotalReads = integer(),
                         PercentDuplication = numeric(),
                         PercentGC = numeric(),
                         stringsAsFactors = FALSE)

for(file in fastqc_files) {
  # Extract sample name
  sample_name <- gsub("_fastqc.zip", "", basename(file))
  
  # Unzip the FastQC file to a temporary directory
  temp_dir <- tempdir()
  unzip(file, exdir = temp_dir)
  
  # Read the FastQC data
  fqdata <- readLines(file.path(temp_dir, paste0(sample_name, "_fastqc"), "fastqc_data.txt"))
  
  # Extract total sequences
  total_seq_line <- grep("Total Sequences", fqdata, value = TRUE)
  total_reads <- as.integer(strsplit(total_seq_line, "\\t")[[1]][2])
  
  # Extract duplication rate
  dup_line <- grep("Total Duplicate Percentage", fqdata, value = TRUE)
  dup_percent <- as.numeric(strsplit(dup_line, "\\t")[[1]][2])
  
  # Extract GC content
  gc_line <- grep("%GC", fqdata, value = TRUE)
  gc_percent <- as.numeric(strsplit(gc_line, "\\t")[[1]][2])
  
  # Add to summary dataframe
  qc_summary <- rbind(qc_summary, 
                      data.frame(Sample = sample_name,
                                 TotalReads = total_reads,
                                 PercentDuplication = dup_percent,
                                 PercentGC = gc_percent,
                                 stringsAsFactors = FALSE))
}

# Display QC summary table
datatable(qc_summary, options = list(pageLength = 10))

# Plot read counts
ggplot(qc_summary, aes(x = Sample, y = TotalReads / 1e6, fill = Sample)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Sample", y = "Total Reads (millions)", title = "Total Reads per Sample")
```

## Mapping Statistics

The following shows the mapping statistics for each sample:

```{r mapping, echo=FALSE}
# Read mapping statistics
stats_files <- list.files("results/mapped", pattern = "*.stats.txt", full.names = TRUE)
mapping_summary <- data.frame(Sample = character(),
                             TotalReads = integer(),
                             MappedReads = integer(),
                             MappingRate = numeric(),
                             stringsAsFactors = FALSE)

for(file in stats_files) {
  # Extract sample name
  sample_name <- gsub(".stats.txt", "", basename(file))
  
  # Read stats file
  stats <- readLines(file)
  
  # Extract mapping statistics
  total_line <- grep("in total", stats, value = TRUE)
  total_reads <- as.integer(strsplit(total_line, " ")[[1]][1])
  
  mapped_line <- grep("mapped (", stats, value = TRUE)
  mapped_reads <- as.integer(strsplit(mapped_line, " ")[[1]][1])
  
  mapping_rate <- as.numeric(gsub("[^0-9\\.]", "", 
                                 regmatches(mapped_line, 
                                           regexpr("\\([0-9\\.]+%\\)", mapped_line))))
  
  # Add to summary dataframe
  mapping_summary <- rbind(mapping_summary, 
                          data.frame(Sample = sample_name,
                                    TotalReads = total_reads,
                                    MappedReads = mapped_reads,
                                    MappingRate = mapping_rate,
                                    stringsAsFactors = FALSE))
}

# Display mapping summary table
datatable(mapping_summary, options = list(pageLength = 10))

# Plot mapping rate
ggplot(mapping_summary, aes(x = Sample, y = MappingRate, fill = Sample)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Sample", y = "Mapping Rate (%)", title = "Mapping Rate per Sample")
```

## Peak Calling Results

Summary of peaks called for each comparison:

```{r peaks, echo=FALSE}
# List peak files
peak_files <- list.files("results/peaks", pattern = "*_peaks.narrowPeak", full.names = TRUE)
peak_summary <- data.frame(Comparison = character(),
                          PeakCount = integer(),
                          stringsAsFactors = FALSE)

for(file in peak_files) {
  # Extract comparison name
  comparison <- gsub("_peaks.narrowPeak", "", basename(file))
  
  # Count peaks
  peaks <- read.table(file)
  peak_count <- nrow(peaks)
  
  # Add to summary dataframe
  peak_summary <- rbind(peak_summary, 
                       data.frame(Comparison = comparison,
                                 PeakCount = peak_count,
                                 stringsAsFactors = FALSE))
}

# Display peak summary table
datatable(peak_summary, options = list(pageLength = 10))

# Plot peak counts
ggplot(peak_summary, aes(x = Comparison, y = PeakCount, fill = Comparison)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Comparison", y = "Number of Peaks", title = "Peak Count per Comparison")
```

## Differential Binding Analysis

The following table shows differentially bound regions with FDR < 0.05:

```{r diffbind, echo=FALSE}
# Read differential binding results
diff_results <- read.csv("results/differential/differential_binding_results.csv")

if(nrow(diff_results) > 0) {
  # Display results table
  datatable(diff_results, options = list(pageLength = 10))
  
  # Volcano plot of results
  ggplot(diff_results, aes(x = Fold, y = -log10(FDR), color = FDR < 0.05)) +
    geom_point(alpha = 0.7) +
    scale_color_manual(values = c("grey", "red")) +
    theme_minimal() +
    facet_wrap(~contrast) +
    labs(x = "Log2 Fold Change", y = "-log10(FDR)", 
         title = "Differential Binding Volcano Plot",
         color = "Significant")
} else {
  cat("No significant differentially bound regions were found.")
}
```

## Conclusion

This report provides a comprehensive overview of the ChIP-seq analysis results. For detailed exploration, please refer to the individual files in the results directory.

', file = report_rmd)

# Render the report
rmarkdown::render(report_rmd, output_file = snakemake@output[[1]])
