cwlVersion: v1.2
class: CommandLineTool
id: gatk_haplotypecaller
label: "GATK HaplotypeCaller Variant Calling"
baseCommand: ["bash", "-c"]
inputs:
  bam: File
  reference: File
outputs:
  vcf_output:
    type: File
    outputBinding:
      glob: "*.vcf.gz"
arguments:
  - "module load gatk/4.2.0.0 && gatk HaplotypeCaller -R $(inputs.reference.path) -I $(inputs.bam.path) \
      -O output_variants.vcf.gz && module unload gatk/4.2.0.0"

