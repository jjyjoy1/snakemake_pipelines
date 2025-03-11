expression_data <- read.table("gene_expression.txt", header=TRUE, row.names=1)
gene_pos <- read.table("gene_positions.txt", header=TRUE)

# Create BED file header
bed_header <- c("#chr", "start", "end", "gene_id", colnames(expression_data))

# Create BED content
bed_content <- data.frame(
  chr = gene_pos$chr,
  start = gene_pos$start,
  end = gene_pos$end,
  gene_id = gene_pos$gene_id
)

# Add expression values
for (gene in bed_content$gene_id) {
  if (gene %in% rownames(expression_data)) {
    bed_content[bed_content$gene_id == gene, 5:ncol(bed_content)] <- expression_data[gene, ]
  }
}

# Write BED file
write.table(
  bed_content, 
  "expression.bed", 
  sep="\t", 
  quote=FALSE, 
  row.names=FALSE, 
  col.names=bed_header
)

# Compress and index
system("bgzip expression.bed")
system("tabix -p bed expression.bed.gz")

