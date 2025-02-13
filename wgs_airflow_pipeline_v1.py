from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

# Define default arguments for the DAG
default_args = {
    'owner': 'bioinformatics',
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'bioinformatics_pipeline',
    default_args=default_args,
    description='A bioinformatics pipeline from fastq.gz to VCF with module management',
    schedule_interval='@once',
)

# Define the tasks

# Task 1: Quality control with FastQC
fastqc_task = BashOperator(
    task_id='fastqc',
    bash_command='module load fastqc/0.11.9 && fastqc input.fastq.gz -o ./fastqc_output && module unload fastqc/0.11.9',
    dag=dag,
)

# Task 2: Trimming with Trimmomatic
trim_task = BashOperator(
    task_id='trim_reads',
    bash_command='module load trimmomatic/0.39 && java -jar $TRIMMOMATIC_HOME/trimmomatic.jar PE -phred33 input.fastq.gz output_forward_paired.fq.gz output_forward_unpaired.fq.gz output_reverse_paired.fq.gz output_reverse_unpaired.fq.gz ILLUMINACLIP:TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36 && module unload trimmomatic/0.39',
    dag=dag,
)

# Task 3: Alignment with BWA
align_task = BashOperator(
    task_id='align_reads',
    bash_command='module load bwa/0.7.17 && bwa mem -t 4 reference_genome.fa output_forward_paired.fq.gz output_reverse_paired.fq.gz > aligned_reads.sam && module unload bwa/0.7.17',
    dag=dag,
)

# Task 4: Convert SAM to BAM and sort
sam_to_bam_task = BashOperator(
    task_id='sam_to_bam',
    bash_command='module load samtools/1.12 && samtools view -Sb aligned_reads.sam | samtools sort -o sorted_reads.bam && module unload samtools/1.12',
    dag=dag,
)

# Task 5: Mark duplicates with Picard
mark_duplicates_task = BashOperator(
    task_id='mark_duplicates',
    bash_command='module load picard/2.25.0 && java -jar $PICARD_HOME/picard.jar MarkDuplicates I=sorted_reads.bam O=marked_duplicates.bam M=marked_dup_metrics.txt && module unload picard/2.25.0',
    dag=dag,
)

# Task 6: Variant calling with GATK HaplotypeCaller
variant_calling_task = BashOperator(
    task_id='variant_calling',
    bash_command='module load gatk/4.2.0.0 && gatk HaplotypeCaller -R reference_genome.fa -I marked_duplicates.bam -O raw_variants.vcf && module unload gatk/4.2.0.0',
    dag=dag,
)

# Task 7: Filter variants with GATK VariantFiltration
filter_variants_task = BashOperator(
    task_id='filter_variants',
    bash_command='module load gatk/4.2.0.0 && gatk VariantFiltration -R reference_genome.fa -V raw_variants.vcf -O filtered_variants.vcf --filter-expression "QD < 2.0 || FS > 60.0 || MQ < 40.0" --filter-name "my_filter" && module unload gatk/4.2.0.0',
    dag=dag,
)

# Task 8: Compress and index the final VCF
compress_index_task = BashOperator(
    task_id='compress_index_vcf',
    bash_command='module load htslib/1.12 && bgzip filtered_variants.vcf && tabix -p vcf filtered_variants.vcf.gz && module unload htslib/1.12',
    dag=dag,
)

# Define the task dependencies
fastqc_task >> trim_task >> align_task >> sam_to_bam_task >> mark_duplicates_task >> variant_calling_task >> filter_variants_task >> compress_index_task
