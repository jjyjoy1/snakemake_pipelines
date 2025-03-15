# ChIP-seq and Gene Expression Integration Analysis
# This script integrates ChIP-seq peak data with gene expression data
# to identify relationships between binding/histone marks and gene expression

library(tidyverse)
library(GenomicRanges)
library(rtracklayer)
library(ChIPseeker)
library(TxDb.Hsapiens.UCSC.hg38.knownGene)  # Replace with appropriate genome
library(org.Hs.eg.db)  # Replace with appropriate organism
library(ggplot2)
library(pheatmap)
library(glmnet)
library(caret)
library(corrplot)

# Set this to your project directory
project_dir <- "/path/to/your/project"
setwd(project_dir)

# Create output directory
dir.create("results/integration", showWarnings = FALSE, recursive = TRUE)

###########################################
# Step 1: Load and process ChIP-seq data #
###########################################

# Load peak data (assuming narrowPeak format from MACS2)
# Modify file paths as needed for your project structure
peak_files <- list.files("results/peaks", pattern = "*_peaks.narrowPeak", full.names = TRUE)
names(peak_files) <- gsub("_peaks.narrowPeak", "", basename(peak_files))

# Create a list to store all peak data
peak_list <- list()

# Load each peak file and convert to GRanges
for (i in seq_along(peak_files)) {
  file_name <- names(peak_files)[i]
  peaks <- import(peak_files[i], format = "narrowPeak")
  
  # If you need to include peak score information
  if ("signalValue" %in% names(mcols(peaks))) {
    mcols(peaks)$score <- mcols(peaks)$signalValue
  }
  
  peak_list[[file_name]] <- peaks
  cat("Loaded", length(peaks), "peaks from", file_name, "\n")
}

# Load gene annotation (using TxDb for human hg38, replace as needed)
txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
genes <- genes(txdb)

# Add gene symbols
gene_symbols <- mapIds(org.Hs.eg.db, 
                      keys = genes$gene_id, 
                      column = "SYMBOL", 
                      keytype = "ENTREZID", 
                      multiVals = "first")
genes$symbol <- gene_symbols

#################################################
# Step 2: Associate peaks with genomic features #
#################################################

# Create a list to store annotated peaks
annotated_peaks <- list()

# Annotate peaks with genomic features for each ChIP sample
for (i in seq_along(peak_list)) {
  sample_name <- names(peak_list)[i]
  peaks <- peak_list[[sample_name]]
  
  # Annotate peaks
  anno <- annotatePeak(peaks, TxDb = txdb, annoDb = "org.Hs.eg.db")
  
  # Save the annotation
  annotated_peaks[[sample_name]] <- anno
  
  # Plot the annotation
  pdf(paste0("results/integration/", sample_name, "_peak_annotation.pdf"), width = 10, height = 8)
  print(plotAnnoBar(anno))
  print(plotDistToTSS(anno))
  dev.off()
  
  cat("Annotated peaks for", sample_name, "\n")
}

#################################################
# Step 3: Create peak-to-gene mapping matrix    #
#################################################

# Function to create a matrix of peak signals per gene
create_peak_gene_matrix <- function(annotated_peaks, 
                                   promoter_only = TRUE, 
                                   upstream = 2000, 
                                   downstream = 500) {
  
  # Define promoter regions if promoter_only is TRUE
  if (promoter_only) {
    promoters <- promoters(txdb, upstream = upstream, downstream = downstream)
    promoter_genes <- unique(mcols(promoters)$gene_id)
  }
  
  # Initialize a list to store data frames
  matrix_list <- list()
  
  # Process each annotated peak file
  for (sample_name in names(annotated_peaks)) {
    anno <- annotated_peaks[[sample_name]]
    anno_df <- as.data.frame(anno@anno)
    
    # Filter to promoter regions if requested
    if (promoter_only) {
      anno_df <- anno_df[anno_df$geneId %in% promoter_genes & 
                        anno_df$annotation %in% c("Promoter", "5' UTR"), ]
    }
    
    # If we have peak scores, use them, otherwise use 1 for presence
    if ("score" %in% colnames(anno_df)) {
      peak_scores <- aggregate(score ~ geneId, data = anno_df, FUN = max)
    } else {
      # Create a presence/absence matrix (1 if a peak exists near the gene)
      unique_genes <- unique(anno_df$geneId)
      peak_scores <- data.frame(geneId = unique_genes, score = 1)
    }
    
    # Rename the score column to the sample name
    colnames(peak_scores)[2] <- sample_name
    
    # Add to the list
    matrix_list[[sample_name]] <- peak_scores
  }
  
  # Merge all data frames by gene ID
  peak_gene_matrix <- Reduce(function(x, y) merge(x, y, by = "geneId", all = TRUE), matrix_list)
  
  # Replace NA with 0 (genes with no peaks)
  peak_gene_matrix[is.na(peak_gene_matrix)] <- 0
  
  # Add gene symbols
  gene_symbols <- mapIds(org.Hs.eg.db, 
                        keys = peak_gene_matrix$geneId, 
                        column = "SYMBOL", 
                        keytype = "ENTREZID", 
                        multiVals = "first")
  
  peak_gene_matrix$symbol <- gene_symbols
  
  return(peak_gene_matrix)
}

# Create the peak-gene matrix (default: promoter regions only)
peak_gene_matrix <- create_peak_gene_matrix(annotated_peaks)

# Write the matrix to file
write.csv(peak_gene_matrix, "results/integration/peak_gene_matrix.csv", row.names = FALSE)

cat("Created peak-gene matrix with", nrow(peak_gene_matrix), "genes and", 
    ncol(peak_gene_matrix) - 2, "ChIP samples.\n")

#################################################
# Step 4: Load and process gene expression data #
#################################################

# Load expression data (modify path as needed)
# This assumes your expression data is in a CSV file with gene IDs or symbols
expression_data <- read.csv("data/gene_expression_matrix.csv")

# Ensure we have a gene identifier column for merging
# This example assumes the expression data has a column named 'gene_id' or 'gene_symbol'
# Modify as needed to match your data

# Check if we need to convert identifiers for merging
if ("gene_id" %in% colnames(expression_data) && "geneId" %in% colnames(peak_gene_matrix)) {
  # Both matrices have Entrez IDs, we can merge directly
  merge_col_expr <- "gene_id"
  merge_col_peak <- "geneId"
} else if ("gene_symbol" %in% colnames(expression_data) && "symbol" %in% colnames(peak_gene_matrix)) {
  # Both matrices have gene symbols, we can merge directly
  merge_col_expr <- "gene_symbol"
  merge_col_peak <- "symbol"
} else {
  # Need to convert identifiers
  if ("gene_id" %in% colnames(expression_data) && !("geneId" %in% colnames(peak_gene_matrix))) {
    # Convert expression data's Entrez IDs to symbols
    expression_data$gene_symbol <- mapIds(org.Hs.eg.db, 
                                         keys = expression_data$gene_id, 
                                         column = "SYMBOL", 
                                         keytype = "ENTREZID", 
                                         multiVals = "first")
    merge_col_expr <- "gene_symbol"
    merge_col_peak <- "symbol"
  } else if ("gene_symbol" %in% colnames(expression_data) && !("symbol" %in% colnames(peak_gene_matrix))) {
    # Convert expression data's symbols to Entrez IDs
    expression_data$gene_id <- mapIds(org.Hs.eg.db, 
                                     keys = expression_data$gene_symbol, 
                                     column = "ENTREZID", 
                                     keytype = "SYMBOL", 
                                     multiVals = "first")
    merge_col_expr <- "gene_id"
    merge_col_peak <- "geneId"
  } else {
    stop("Cannot determine how to merge expression data with peak data. Please ensure gene identifiers are present.")
  }
}

##################################################
# Step 5: Integrate ChIP-seq and expression data #
##################################################

# Merge the peak-gene matrix with expression data
integrated_data <- merge(peak_gene_matrix, expression_data, 
                        by.x = merge_col_peak, by.y = merge_col_expr)

# Write the integrated data to file
write.csv(integrated_data, "results/integration/integrated_peak_expression_data.csv", row.names = FALSE)

cat("Created integrated dataset with", nrow(integrated_data), "genes.\n")

##################################################
# Step 6: Correlation analysis                   #
##################################################

# Identify the expression columns (modify based on your data format)
# This assumes expression columns start with "expr_" - adapt to your naming convention
expr_cols <- grep("^expr_", colnames(integrated_data), value = TRUE)

# Identify the ChIP peak columns (all numeric columns except gene IDs and expression)
peak_cols <- setdiff(
  colnames(integrated_data)[sapply(integrated_data, is.numeric)],
  c(merge_col_peak, "geneId", "symbol", expr_cols)
)

# Create a correlation matrix between peak scores and expression
corr_matrix <- matrix(NA, nrow = length(peak_cols), ncol = length(expr_cols))
rownames(corr_matrix) <- peak_cols
colnames(corr_matrix) <- expr_cols

# Fill the correlation matrix
for (i in seq_along(peak_cols)) {
  for (j in seq_along(expr_cols)) {
    corr_matrix[i, j] <- cor(integrated_data[[peak_cols[i]]], 
                           integrated_data[[expr_cols[j]]], 
                           method = "spearman",
                           use = "pairwise.complete.obs")
  }
}

# Plot the correlation heatmap
pdf("results/integration/peak_expression_correlation_heatmap.pdf", width = 10, height = 8)
pheatmap(corr_matrix, 
        main = "Correlation between ChIP-seq Peaks and Gene Expression",
        color = colorRampPalette(c("blue", "white", "red"))(100),
        breaks = seq(-1, 1, length.out = 101),
        cluster_rows = TRUE, 
        cluster_cols = TRUE)
dev.off()

# Write correlation matrix to file
write.csv(corr_matrix, "results/integration/peak_expression_correlation.csv")

##################################################
# Step 7: Predictive modeling                    #
##################################################

# Function to build a predictive model for gene expression based on ChIP-seq data
build_predictive_model <- function(integrated_data, expr_col, peak_cols, 
                                 method = "glmnet", alpha = 0.5, 
                                 train_prop = 0.7) {
  
  # Prepare the modeling dataset
  model_data <- integrated_data[, c(expr_col, peak_cols)]
  model_data <- model_data[complete.cases(model_data), ]
  
  # Split into training and testing sets
  set.seed(42)
  train_idx <- createDataPartition(model_data[[expr_col]], p = train_prop, list = FALSE)
  train_data <- model_data[train_idx, ]
  test_data <- model_data[-train_idx, ]
  
  # Feature matrix and response vector for training
  x_train <- as.matrix(train_data[, peak_cols])
  y_train <- train_data[[expr_col]]
  
  # For testing
  x_test <- as.matrix(test_data[, peak_cols])
  y_test <- test_data[[expr_col]]
  
  if (method == "glmnet") {
    # Train elastic net model (alpha=1 for LASSO, alpha=0 for ridge, alpha=0.5 for elastic net)
    cv_model <- cv.glmnet(x_train, y_train, alpha = alpha, nfolds = 5)
    
    # Best lambda
    best_lambda <- cv_model$lambda.min
    
    # Make predictions
    predictions <- predict(cv_model, s = best_lambda, newx = x_test)
    
    # Calculate model performance
    rmse <- sqrt(mean((predictions - y_test)^2))
    r_squared <- cor(predictions, y_test)^2
    
    # Get feature importance
    coef_lambda_min <- coef(cv_model, s = best_lambda)
    feature_importance <- data.frame(
      feature = rownames(coef_lambda_min),
      importance = as.numeric(coef_lambda_min)
    )
    feature_importance <- feature_importance[feature_importance$feature != "(Intercept)", ]
    feature_importance <- feature_importance[order(abs(feature_importance$importance), decreasing = TRUE), ]
    
    # Create result object
    model_result <- list(
      model = cv_model,
      method = method,
      rmse = rmse,
      r_squared = r_squared,
      feature_importance = feature_importance,
      pred_vs_actual = data.frame(predicted = as.numeric(predictions), actual = y_test)
    )
    
    return(model_result)
  } else {
    # For other methods (can be expanded)
    stop(paste("Method", method, "not implemented"))
  }
}

# Build a predictive model for each expression column
model_results <- list()

for (expr_col in expr_cols) {
  cat("Building predictive model for", expr_col, "\n")
  model_results[[expr_col]] <- build_predictive_model(integrated_data, expr_col, peak_cols)
  
  # Plot predicted vs actual values
  result <- model_results[[expr_col]]
  p <- ggplot(result$pred_vs_actual, aes(x = actual, y = predicted)) +
    geom_point(alpha = 0.5) +
    geom_smooth(method = "lm", color = "red") +
    labs(title = paste("Predicted vs Actual Expression -", expr_col),
         subtitle = paste("R² =", round(result$r_squared, 3), "RMSE =", round(result$rmse, 3)),
         x = "Actual Expression",
         y = "Predicted Expression") +
    theme_minimal()
  
  ggsave(paste0("results/integration/prediction_model_", expr_col, ".pdf"), p, width = 8, height = 6)
  
  # Save feature importance
  write.csv(result$feature_importance, 
           paste0("results/integration/feature_importance_", expr_col, ".csv"), 
           row.names = FALSE)
}

##################################################
# Step 8: Visualization of key findings          #
##################################################

# Identify top genes with strong correlation between ChIP and expression
top_correlated_genes <- function(integrated_data, peak_col, expr_col, top_n = 20) {
  data <- integrated_data[, c("symbol", peak_col, expr_col)]
  data <- data[complete.cases(data), ]
  
  # Calculate correlation for each gene
  correlations <- data.frame(
    symbol = data$symbol,
    correlation = apply(data[, c(peak_col, expr_col)], 1, function(x) cor(x[1], x[2]))
  )
  
  # Sort and get top positive and negative correlated genes
  correlations <- correlations[order(correlations$correlation), ]
  top_negative <- head(correlations, top_n)
  top_positive <- tail(correlations, top_n)
  
  return(list(top_positive = top_positive, top_negative = top_negative))
}

# For the first expression column and peak column, get top correlated genes
if (length(expr_cols) > 0 && length(peak_cols) > 0) {
  top_genes <- top_correlated_genes(integrated_data, peak_cols[1], expr_cols[1])
  
  # Write to file
  write.csv(rbind(top_genes$top_positive, top_genes$top_negative),
           "results/integration/top_correlated_genes.csv", row.names = FALSE)
  
  # Create scatter plots for these genes
  for (i in 1:min(5, nrow(top_genes$top_positive))) {
    gene <- top_genes$top_positive$symbol[i]
    gene_data <- integrated_data[integrated_data$symbol == gene, ]
    
    if (nrow(gene_data) > 0) {
      p <- ggplot(gene_data, aes_string(x = peak_cols[1], y = expr_cols[1])) +
        geom_point(color = "blue", size = 3) +
        labs(title = paste("Positive Correlation for", gene),
             x = "ChIP-seq Peak Score",
             y = "Gene Expression Level") +
        theme_minimal()
      
      ggsave(paste0("results/integration/scatter_positive_", i, "_", gene, ".pdf"), p, width = 6, height = 6)
    }
  }
  
  for (i in 1:min(5, nrow(top_genes$top_negative))) {
    gene <- top_genes$top_negative$symbol[i]
    gene_data <- integrated_data[integrated_data$symbol == gene, ]
    
    if (nrow(gene_data) > 0) {
      p <- ggplot(gene_data, aes_string(x = peak_cols[1], y = expr_cols[1])) +
        geom_point(color = "red", size = 3) +
        labs(title = paste("Negative Correlation for", gene),
             x = "ChIP-seq Peak Score",
             y = "Gene Expression Level") +
        theme_minimal()
      
      ggsave(paste0("results/integration/scatter_negative_", i, "_", gene, ".pdf"), p, width = 6, height = 6)
    }
  }
}

##################################################
# Step 9: Pathway enrichment analysis            #
##################################################

# If available, perform pathway enrichment on genes with strong correlations
if (require(clusterProfiler) && require(DOSE)) {
  # For the first expression column and peak column
  if (length(expr_cols) > 0 && length(peak_cols) > 0) {
    # Get correlation between peak score and expression for each gene
    gene_data <- integrated_data[, c("geneId", peak_cols[1], expr_cols[1])]
    gene_data <- gene_data[complete.cases(gene_data), ]
    
    # Calculate correlation for each gene
    correlations <- data.frame(
      geneId = gene_data$geneId,
      correlation = sapply(1:nrow(gene_data), function(i) {
        cor(gene_data[i, peak_cols[1]], gene_data[i, expr_cols[1]])
      })
    )
    
    # Sort by correlation
    correlations <- correlations[order(correlations$correlation, decreasing = TRUE), ]
    
    # Select top positively correlated genes for enrichment analysis
    top_pos_genes <- head(correlations$geneId, 200)
    
    # Convert gene IDs to Entrez IDs (if not already)
    if (!all(grepl("^\\d+$", top_pos_genes))) {
      top_pos_genes <- mapIds(org.Hs.eg.db, 
                             keys = top_pos_genes, 
                             column = "ENTREZID", 
                             keytype = "SYMBOL", 
                             multiVals = "first")
    }
    
    # Remove NA values
    top_pos_genes <- top_pos_genes[!is.na(top_pos_genes)]
    
    # Perform GO enrichment analysis
    go_enrich <- enrichGO(gene = top_pos_genes,
                         OrgDb = org.Hs.eg.db,
                         ont = "BP",
                         pAdjustMethod = "BH",
                         pvalueCutoff = 0.05,
                         qvalueCutoff = 0.2)
    
    # Save results
    if (!is.null(go_enrich) && nrow(go_enrich@result) > 0) {
      write.csv(go_enrich@result, "results/integration/go_enrichment_positive_correlation.csv", row.names = FALSE)
      
      # Plot
      pdf("results/integration/go_enrichment_positive_correlation.pdf", width = 10, height = 8)
      print(dotplot(go_enrich, showCategory = 20))
      dev.off()
    }
    
    # Select top negatively correlated genes
    correlations <- correlations[order(correlations$correlation), ]
    top_neg_genes <- head(correlations$geneId, 200)
    
    # Convert gene IDs to Entrez IDs (if not already)
    if (!all(grepl("^\\d+$", top_neg_genes))) {
      top_neg_genes <- mapIds(org.Hs.eg.db, 
                             keys = top_neg_genes, 
                             column = "ENTREZID", 
                             keytype = "SYMBOL", 
                             multiVals = "first")
    }
    
    # Remove NA values
    top_neg_genes <- top_neg_genes[!is.na(top_neg_genes)]
    
    # Perform GO enrichment analysis
    go_enrich_neg <- enrichGO(gene = top_neg_genes,
                            OrgDb = org.Hs.eg.db,
                            ont = "BP",
                            pAdjustMethod = "BH",
                            pvalueCutoff = 0.05,
                            qvalueCutoff = 0.2)
    
    # Save results
    if (!is.null(go_enrich_neg) && nrow(go_enrich_neg@result) > 0) {
      write.csv(go_enrich_neg@result, "results/integration/go_enrichment_negative_correlation.csv", row.names = FALSE)
      
      # Plot
      pdf("results/integration/go_enrichment_negative_correlation.pdf", width = 10, height = 8)
      print(dotplot(go_enrich_neg, showCategory = 20))
      dev.off()
    }
  }
}

# Save the session info for reproducibility
writeLines(capture.output(sessionInfo()), "results/integration/session_info.txt")

cat("Analysis completed successfully!\n")


