cwlVersion: v1.2
class: CommandLineTool
id: fastqc
label: "FastQC Quality Control"
baseCommand: ["bash", "-c"]
inputs:
  input_fastq:
    type: File
    inputBinding:
      position: 1
outputs:
  fastqc_report:
    type: File
    outputBinding:
      glob: "*.html"
stdout: fastqc_output.log
arguments:
  - "module load fastqc/1.1.0 && fastqc $(inputs.input_fastq.path) && module unload fastqc/1.1.0"


