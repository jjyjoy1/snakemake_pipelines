#metadaa_val could be anywhere, which based on the project setting.
#It isn't possible to include in pipeline. 
#redbiom search metadata "where host_taxid==9606 and (sample_type=='stool' or sample_type=='Stool')" > sample_ids
#

matedata_val = "Animal surface"surface


rule build_weight_class:
    input:
        uniform_clr = os.path.join(config["result"]["taxonomy"], "classifier.qza"),
        ref = os.path.join(config["result"]["import"], "refseqs.qza"),
        tax = os.path.join(config["result"]["import"], "reftax.qza"),
    output:
        weight_clr = os.path.join(config["result"]["taxonomy"], "weighted_class.qza")
    params:
        key = "empo_3",
    shell:
        """
        qiime clawback assemble-weights-from-Qiita \
         --i-classifier {input.uniform_clr} \
         --i-reference-taxonomy {input.tax} \
         --i-reference-sequences {input.ref} \
         --p-metadata-key {params.key} \
         --p-metadata-value matedata_val \
         --p-context Deblur-Illumina-16S-V4-150nt-780653 \
         --o-class-weight {output}

        """

rule retrain_clr:
    input:
       ref = os.path.join(config["result"]["import"], "refseqs.qza"),
       tax = os.path.join(config["result"]["import"], "reftax.qza"),
       weight_class = os.path.join(config["result"]["taxonomy"], "weighted_class.qza"),
    output:
       weight_clr = os.path.join(config["result"]["taxonomy"], "weighted_classifier.qza")
    shell:
        """
        qiime feature-classifier fit-classifier-naive-bayes \
         --i-reference-reads {input.ref} \
         --i-reference-taxonomy {input.tax} \
         --i-class-weight {input.weight_class} \
         --o-classifier {output.weight_clr}

        """
        
