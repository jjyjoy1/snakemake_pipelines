# Start with your expression data
expression_data <- read.table("gene_expression.txt", header=TRUE, row.names=1)

# Need gene position information
gene_positions <- read.table("gene_positions.txt", header=TRUE)
# This file should have columns: gene_id, chr, start, end

# Create BED format data frame
bed_format <- data.frame(
  chr = gene_positions$chr,
  start = gene_positions$start,
  end = gene_positions$end,
  gene_id = gene_positions$gene_id,
  stringsAsFactors = FALSE
)

# Add expression data (samples as columns)
for (sample in colnames(expression_data)) {
  bed_format[[sample]] <- expression_data[bed_format$gene_id, sample]
}

# Write out in BED format
write.table(bed_format, "expression.bed", quote=FALSE, sep="\t", row.names=FALSE, col.names=TRUE)

# Compress and index
system("bgzip expression.bed")
system("tabix -p bed expression.bed.gz")
