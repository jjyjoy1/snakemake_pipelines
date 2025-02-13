cwlVersion: v1.2
class: CommandLineTool
id: vep_annotation
label: "Variant Effect Predictor (VEP) Annotation"
baseCommand: ["bash", "-c"]
inputs:
  vcf: File
outputs:
  annotated_vcf:
    type: File
    outputBinding:
      glob: "*.annotated.vcf.gz"
arguments:
  - "module load vep/104 && vep -i $(inputs.vcf.path) -o annotated_variants.vcf.gz --cache --everything && module unload vep/104"


