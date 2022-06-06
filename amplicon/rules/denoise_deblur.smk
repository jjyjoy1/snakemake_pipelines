rule vsearch_join:
    input:
        f = os.path.join(config["result"]["remove"], "paired-end-demux-trim.qza")
    output:
        pair_merged = os.path.join(config["result"]["import"], "pair_merged.qza")
    params:
        minlen = ,
        maxdif = ,
        minmerge = , 
    shell:
        """
        qiime vsearch join-pairs \
      --i-demultiplexed-seqs {input} \
      --o-joined-sequences {output} \
      --p-minovlen {params.minlen} \
      --p-maxdiffs {params.maxdif} \
      --p-minmergelen {params.minmerge}

        """


rule filter_qscore: 
    input:
        rules.vsearch_join.output.pair_merged
    output:
        seqs = os.path.join(config["result"] "joined-filtered.qza"),
        stats = os.path.join(config["result"],"joined-filtered-stats.qza")
    params:
         minphred = 4,
         qualw = 3,
    shell:
        """
        qiime quality-filter q-score \
            --i-demux {input} \
            --o-filtered-sequences {output.seqs} \
            --p-min-quality {params.minphred]} \
            --p-quality-window {params.qualw} \
            --o-filter-stats {output.stats}
        """


rule deblur:
    input:
         rules.filter_qscore.output.seqs
    output:
         reps = os.path.join(config["result"],"rep-seqs.qza"), 
         tab = os.path.join(config["result"],"table.qza"),
         stats = os.path.join(config["result"],"deblur-stats.qza")
    params:
         trimlen = 250
    shell:
        """
        qiime deblur denoise-16S \
        --i-demultiplexed-seqs {input} \
        --p-trim-length {params.trimlen} \
        --o-representative-sequences {output.reps} \
        --o-table {output.tab} \
        --p-sample-stats \
        --o-stats {output.stats}
        """


rule feature_summary:
    input:
        rules.deblue.output.tab
    output:
        sum = os.path.join(config["result"], "deblur.tavle.qzv")
    shell:
        """
        qiime feature-table summarize \
        --i-table {input} \
        --o-visualization {output}
        """


