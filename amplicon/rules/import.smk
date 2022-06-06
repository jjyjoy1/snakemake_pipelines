rule import_ref_seqs:
    input:
        config["params"]["classifier"]["refseqs"]
    output:
        os.path.join(config["result"]["import"], "refseqs.qza")
    shell:
        '''
        qiime tools import \
            --type 'FeatureData[Sequence]' \
            --input-path {input} \
            --output-path {output}
        '''

rule import_ref_tax:
    input:
        config["params"]["classifier"]["reftax"]
    output:
        os.path.join(config["result"]["import"], "reftax.qza")
    shell:
        '''
        qiime tools import \
            --type 'FeatureData[Taxonomy]' \
            --input-format HeaderlessTSVTaxonomyFormat \
            --input-path {input} \
            --output-path {output}
        '''

rule build_naive-bayes_classifier:
    input:
        refseqs = os.path.join(config["result"]["import"], "refseqs.qza"),
        reftax  = os.path.join(config["result"]["import"], "reftax.qza")
    output:
        os.path.join(config["result"]["taxonomy"], "classifier.qza")
    shell:
        '''
        qiime feature-classifier fit-classifier-naive-bayes \
            --i-reference-reads {input.refseqs} \
            --i-reference-taxonomy {input.reftax} \
            --o-classifier {output}
        '''


rule import_fastq_demux_pe:
    input:
        config["manifest"]
    log:
        os.path.join(config["logs"]["import"], "fastq2qza.log")
    output:
        os.path.join(config["result"]["import"], "paired-end-demux.qza")
    shell:
        '''
        qiime tools import \
            --type 'SampleData[PairedEndSequencesWithQuality]' \
            --input-path {input} \
            --output-path {output} \
            --input-format PairedEndFastqManifestPhred33 2>{log}
        '''

"""
rule summarize_fastq_demux_pe:
    input:
        rules.import_fastq_demux_pe.output,
    output:
        out = os.path.join(config["result"]["import"], "stq_summary.qzv"),
    shell:
        "qiime demux summarize "
        "--i-data {input} "
        "--o-visualization {output}"


rule export_fastq_summary_to_counts:
    input:
        rules.summarize_fastq_demux_pe.output
    output:
        out = os.path.join(config["result"]["import"],"fastq_counts.tsv")
     shell:
        "unzip -qq -o {input} -d temp0; "
        "mv temp0/*/data/per-sample-fastq-counts.tsv {output}; "
        "/bin/rm -r temp0"



rule describe_fastq_counts:
    input:
        rules.export_fastq_summary_to_counts.output,
    output:
        out = os.path.join(config["result"]["import"],"fastq_counts_describe.md")
    run:
        s = pd.read_csv(input[0], index_col=0, sep='\t')
        t = s.describe()
        outstr = tabulate(pd.DataFrame(t.iloc[1:,0]), tablefmt="pipe", headers=['Statistic (n=%s)' % t.iloc[0,0].astype(int), 'Fastq sequences per sample'])
        with open(output[0], 'w') as target:
            target.write(outstr)
            target.write('\n')

"""

