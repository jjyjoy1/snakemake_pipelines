## Peak Annotation Analysis

The following shows the genomic annotation of peaks for each comparison:

```{r peaks_annotation, echo=FALSE}
# Check for annotation files
anno_files <- list.files("results/annotation", pattern = "*_ChIPseeker_annotation.csv", full.names = TRUE)
if (length(anno_files) > 0) {
  # Read annotation summaries
  anno_summary <- data.frame(Comparison = character(),
                          PromoterPeaks = integer(),
                          ExonicPeaks = integer(),
                          IntronicPeaks = integer(),
                          IntergenicPeaks = integer(),
                          stringsAsFactors = FALSE)
  
  for (file in anno_files) {
    # Extract comparison name
    comparison <- gsub("_ChIPseeker_annotation.csv", "", basename(file))
    
    # Read annotation file
    anno <- read.csv(file)
    
    # Count peaks by annotation type
    promoter_count <- sum(grepl("Promoter", anno$annotation))
    exon_count <- sum(grepl("Exon", anno$annotation))
    intron_count <- sum(grepl("Intron", anno$annotation))
    intergenic_count <- sum(grepl("Intergenic", anno$annotation))
    
    # Add to summary
    anno_summary <- rbind(anno_summary,
                          data.frame(Comparison = comparison,
                                    PromoterPeaks = promoter_count,
                                    ExonicPeaks = exon_count,
                                    IntronicPeaks = intron_count,
                                    IntergenicPeaks = intergenic_count,
                                    stringsAsFactors = FALSE))
  }
  
  # Display annotation summary table
  datatable(anno_summary, options = list(pageLength = 10))
  
  # Create stacked bar chart of annotation distribution
  anno_long <- reshape2::melt(anno_summary, id.vars = "Comparison", 
                             variable.name = "AnnotationType", value.name = "PeakCount")
  
  ggplot(anno_long, aes(x = Comparison, y = PeakCount, fill = AnnotationType)) +
    geom_bar(stat = "identity", position = "stack") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(x = "Comparison", y = "Number of Peaks", title = "Distribution of Peak Annotations")
}
```

## Functional Enrichment Analysis

The following shows the functional enrichment analysis results for each comparison:

```{r functional, echo=FALSE}
# Check for GO enrichment files
go_files <- list.files("results/functional", pattern = "*_GO_BP.csv", full.names = TRUE)
if (length(go_files) > 0) {
  # Create a summary of GO enrichment
  go_summary <- data.frame(Comparison = character(),
                          TopGOTerm = character(),
                          AdjustedPValue = numeric(),
                          GeneCount = integer(),
                          stringsAsFactors = FALSE)
  
  for (file in go_files) {
    # Extract comparison name
    comparison <- gsub("_GO_BP.csv", "", basename(file))
    
    # Read GO file
    go_data <- read.csv(file)
    
    if (nrow(go_data) > 0) {
      # Get top term
      top_term <- go_data[1, ]
      
      # Add to summary
      go_summary <- rbind(go_summary,
                         data.frame(Comparison = comparison,
                                   TopGOTerm = top_term$Description,
                                   AdjustedPValue = top_term$p.adjust,
                                   GeneCount = top_term$Count,
                                   stringsAsFactors = FALSE))
    }
  }
  
  # Display GO summary table
  if (nrow(go_summary) > 0) {
    datatable(go_summary, options = list(pageLength = 10))
    
    # Create bubble plot for top GO terms
    ggplot(go_summary, aes(x = Comparison, y = TopGOTerm, size = GeneCount, color = -log10(AdjustedPValue))) +
      geom_point() +
      scale_color_gradient(low = "blue", high = "red") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(x = "Comparison", y = "Top GO Term", size = "Gene Count", color = "-log10(Adj.P)", 
           title = "Top GO Biological Process Terms")
  }
}

# Check for KEGG pathway files
kegg_files <- list.files("results/functional", pattern = "*_KEGG_pathways.csv", full.names = TRUE)
if (length(kegg_files) > 0) {
  # Create a summary of KEGG enrichment
  kegg_summary <- data.frame(Comparison = character(),
                            TopPathway = character(),
                            AdjustedPValue = numeric(),
                            GeneCount = integer(),
                            stringsAsFactors = FALSE)
  
  for (file in kegg_files) {
    # Extract comparison name
    comparison <- gsub("_KEGG_pathways.csv", "", basename(file))
    
    # Read KEGG file
    kegg_data <- read.csv(file)
    
    if (nrow(kegg_data) > 0) {
      # Get top pathway
      top_pathway <- kegg_data[1, ]
      
      # Add to summary
      kegg_summary <- rbind(kegg_summary,
                          data.frame(Comparison = comparison,
                                    TopPathway = top_pathway$Description,
                                    AdjustedPValue = top_pathway$p.adjust,
                                    GeneCount = top_pathway$Count,
                                    stringsAsFactors = FALSE))
    }
  }
  
  # Display KEGG summary table
  if (nrow(kegg_summary) > 0) {
    datatable(kegg_summary, options = list(pageLength = 10))
    
    # Create bubble plot for top KEGG pathways
    ggplot(kegg_summary, aes(x = Comparison, y = TopPathway, size = GeneCount, color = -log10(AdjustedPValue))) +
      geom_point() +
      scale_color_gradient(low = "blue", high = "red") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(x = "Comparison", y = "Top KEGG Pathway", size = "Gene Count", color = "-log10(Adj.P)", 
           title = "Top KEGG Pathways")
  }
}
```

## GREAT Analysis Results

The following shows results from GREAT (Genomic Regions Enrichment of Annotations Tool) analysis:

```{r great, echo=FALSE}
# Check for GREAT analysis files
great_files <- list.files("results/annotation", pattern = "*_GREAT_GO_BP.csv", full.names = TRUE)
if (length(great_files) > 0) {
  # Create a summary of GREAT enrichment
  great_summary <- data.frame(Comparison = character(),
                             TopTerm = character(),
                             AdjustedPValue = numeric(),
                             RegionFoldEnrichment = numeric(),
                             stringsAsFactors = FALSE)
  
  for (file in great_files) {
    # Extract comparison name
    comparison <- gsub("_GREAT_GO_BP.csv", "", basename(file))
    
    # Read GREAT file
    great_data <- read.csv(file)
    
    if (nrow(great_data) > 0) {
      # Get top term
      top_term <- great_data[1, ]
      
      # Add to summary
      great_summary <- rbind(great_summary,
                            data.frame(Comparison = comparison,
                                      TopTerm = top_term$name,
                                      AdjustedPValue = top_term$Hyper_Adjp_BH,
                                      RegionFoldEnrichment = top_term$Hyper_Fold_Enrichment,
                                      stringsAsFactors = FALSE))
    }
  }
  
  # Display GREAT summary table
  if (nrow(great_summary) > 0) {
    datatable(great_summary, options = list(pageLength = 10))
    
    # Create bubble plot for top GREAT terms
    ggplot(great_summary, aes(x = Comparison, y = TopTerm, size = RegionFoldEnrichment, 
                            color = -log10(AdjustedPValue))) +
      geom_point() +
      scale_color_gradient(low = "blue", high = "red") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(x = "Comparison", y = "Top GREAT Term", size = "Fold Enrichment", 
           color = "-log10(Adj.P)", title = "Top GREAT Enrichment Terms")
  }
}

# Check for GREAT gene association files
assoc_files <- list.files("results/annotation", pattern = "*_GREAT_gene_associations.csv", full.names = TRUE)
if (length(assoc_files) > 0) {
  # Create a summary of gene associations
  assoc_summary <- data.frame(Comparison = character(),
                             AssociatedGenes = integer(),
                             AvgDistance = numeric(),
                             stringsAsFactors = FALSE)
  
  for (file in assoc_files) {
    # Extract comparison name
    comparison <- gsub("_GREAT_gene_associations.csv", "", basename(file))
    
    # Read association file
    assoc_data <- read.csv(file)
    
    if (nrow(assoc_data) > 0) {
      # Calculate summary statistics
      n_genes <- length(unique(assoc_data$gene))
      
      # Add distance column if it exists
      if ("distTSS" %in% colnames(assoc_data)) {
        avg_dist <- mean(abs(assoc_data$distTSS), na.rm = TRUE)
      } else {
        avg_dist <- NA
      }
      
      # Add to summary
      assoc_summary <- rbind(assoc_summary,
                            data.frame(Comparison = comparison,
                                      AssociatedGenes = n_genes,
                                      AvgDistance = avg_dist,
                                      stringsAsFactors = FALSE))
    }
  }
  
  # Display association summary
  if (nrow(assoc_summary) > 0) {
    datatable(assoc_summary, options = list(pageLength = 10))
  }
}
```

## Reactome Pathway Analysis

```{r reactome, echo=FALSE}
# Check for Reactome pathway files
reactome_files <- list.files("results/functional", pattern = "*_Reactome_pathways.csv", full.names = TRUE)
if (length(reactome_files) > 0) {
  # Create a summary of Reactome enrichment
  reactome_summary <- data.frame(Comparison = character(),
                               TopPathway = character(),
                               AdjustedPValue = numeric(),
                               GeneCount = integer(),
                               stringsAsFactors = FALSE)
  
  for (file in reactome_files) {
    # Extract comparison name
    comparison <- gsub("_Reactome_pathways.csv", "", basename(file))
    
    # Read Reactome file
    reactome_data <- read.csv(file)
    
    if (nrow(reactome_data) > 0) {
      # Get top pathway
      top_pathway <- reactome_data[1, ]
      
      # Add to summary
      reactome_summary <- rbind(reactome_summary,
                              data.frame(Comparison = comparison,
                                        TopPathway = top_pathway$Description,
                                        AdjustedPValue = top_pathway$p.adjust,
                                        GeneCount = top_pathway$Count,
                                        stringsAsFactors = FALSE))
    }
  }
  
  # Display Reactome summary table
  if (nrow(reactome_summary) > 0) {
    datatable(reactome_summary, options = list(pageLength = 10))
    
    # Create bubble plot for top Reactome pathways
    ggplot(reactome_summary, aes(x = Comparison, y = TopPathway, size = GeneCount, 
                               color = -log10(AdjustedPValue))) +
      geom_point() +
      scale_color_gradient(low = "blue", high = "red") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(x = "Comparison", y = "Top Reactome Pathway", size = "Gene Count", 
           color = "-log10(Adj.P)", title = "Top Reactome Pathways")
  }
}
```

## Disease Ontology Analysis

```{r disease, echo=FALSE}
# Check for Disease Ontology files
do_files <- list.files("results/functional", pattern = "*_Disease_Ontology.csv", full.names = TRUE)
if (length(do_files) > 0) {
  # Create a summary of Disease Ontology enrichment
  do_summary <- data.frame(Comparison = character(),
                          TopDisease = character(),
                          AdjustedPValue = numeric(),
                          GeneCount = integer(),
                          stringsAsFactors = FALSE)
  
  for (file in do_files) {
    # Extract comparison name
    comparison <- gsub("_Disease_Ontology.csv", "", basename(file))
    
    # Read DO file
    do_data <- read.csv(file)
    
    if (nrow(do_data) > 0) {
      # Get top disease
      top_disease <- do_data[1, ]
      
      # Add to summary
      do_summary <- rbind(do_summary,
                         data.frame(Comparison = comparison,
                                   TopDisease = top_disease$Description,
                                   AdjustedPValue = top_disease$p.adjust,
                                   GeneCount = top_disease$Count,
                                   stringsAsFactors = FALSE))
    }
  }
  
  # Display DO summary table
  if (nrow(do_summary) > 0) {
    datatable(do_summary, options = list(pageLength = 10))
    
    # Create bubble plot for top disease terms
    ggplot(do_summary, aes(x = Comparison, y = TopDisease, size = GeneCount, 
                          color = -log10(AdjustedPValue))) +
      geom_point() +
      scale_color_gradient(low = "blue", high = "red") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(x = "Comparison", y = "Top Disease Term", size = "Gene Count", 
           color = "-log10(Adj.P)", title = "Top Disease Ontology Terms")
  }
}
```# scripts/generate_report.R - Script to generate final report

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
