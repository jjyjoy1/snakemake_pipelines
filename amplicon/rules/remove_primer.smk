#The rule of "extract_reads" and "fit_region_classifer" are not necessary, but it you run and use it in downstream assay, it could save your running time

rule extract_reads:
    input:
        ful_ref = config["params"]["classifier"]["refseqs"]
    output:
        ref_seqs_redion = "ref-seqs-v4.qza"  
    params:
        forward_seq = config["params"]["remove"]["forward_sequence_515F"],
        reverse_seq = config["params"]["remove"]["reverse_sequence_806R"]
    shell:
        """
        qiime feature-classifier extract-reads \
         --i-sequences {input.ful_ref} \
         --p-f-primer {params.forward_seq} \
         --p-r-primer {params.reverse_seq} \
         --o-reads {output}
        """

rule fit_region_classifer:
    input:
        rules.extract_reads.output
        tax = config["params"]["classifier"]["reftax"],
    output:
        region_unif_clr = 
    shell:
        """
        qiime feature-classifier fit-classifier-naive-bayes \
         --i-reference-reads ref-seqs-v4.qza \
         --i-reference-taxonomy ref-taxonomy.qza \
         --o-classifier region-uniform-classifier.qza

        """        


rule remove_primer_pe:
    input:
        os.path.join(config["result"]["import"], "paired-end-demux.qza")
    params:
        forward_seq = config["params"]["remove"]["forward_sequence_515F"],
        reverse_seq = config["params"]["remove"]["reverse_sequence_806R"]
    threads: 
        config["params"]["remove"]["remove_primer_threads"]
    log:
        os.path.join(config["logs"]["remove"], "remove_primer.log")
    output:
        os.path.join(config["result"]["remove"], "paired-end-demux-trim.qza")
    shell:
        '''
        qiime cutadapt trim-paired \
            --i-demultiplexed-sequences {input} \
            --p-front-f {params.forward_seq} \
            --p-front-r {params.reverse_seq} \
            --p-cores {threads} \
            --o-trimmed-sequences {output} 2>{log}
        '''
