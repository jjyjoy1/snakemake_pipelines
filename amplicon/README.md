## Amplicon Sequence Analysis Workflow

Build this pipeline based on https://github.com/HuaZou/bioinformatics_pipeline. 
 
I have modified in several places. 
a. Use new python script to build manifest.csv file, I suppose the data was the paired fastq.gz file, and all files wase in same folder. 
b. I added deblur denoise step, in case it will be used it later.
c. I added "weighted_taxo_classifiers.smk", using "qiime clawback". This classifier can be build based on the project as needed. Because the custom weighted taxo classifier should be build based on project. I didn't add it in pipeline. 
d. QIIME2 provided statistical analysis, I prefer to use several of R packages to do down stream analysis, including Phyloseq, metacoder, taxa, ape, vegan...
e. So as well as the sample_metadata.csv, taxonomy.qza, rep_seqs.qza are available from QIIME2, import the output into phyloseq or other R packages could generate even better picture.
f. If there are the previous QIIME/QIIME2 results data which need to re-analysis , the previous results can be export as biom file, and then import into R as phyloseq object. 


For Downstream metagenomics analysis, there are several of factors need to be considered before process data. Usually the downstream data analysis depends on the experiment design and experiment hypothesis. That is main reason it is difficult to prepare one pipeline to process different data. 
    1. Filter the low abundance reads, including absolult lower number, filter based on group number, low frequence among groups, such as filter the feature which presents in less 2/3 groups.
    2. Select the intersting group to do the next step analysis
    3. Normalization. This is the first step of formal data analysis. 
          The most simply and frequently used mormalization is relative abundances and rarefaction. If the project only contains small samples, and the main purpose of research is detecting the difference between samples diversity. I think it is OK to use.
          The more sophisticated normalization techniques can be implemented in R packages, such as DEseq which consider size factor, edgeR which used trimmed mean of M-value(TMM). Weiss S, Xu ZZ, et al. Normalization and microbial differential abundance strategies depend upon data characteristics. Microbiome 2017;5:27.   
          However, the normalization techniques DEseq and edgR failed to address covariants in experiments, especially in the big experiment design, which contains either technical(batch effect), environmental or biological(phenotype) covariates. In order to handle smaple metadata covariants, the four R packages will be used successive procedures.  
	Principal Variance Component Analysis (PVCA) to explore how technical and biological factors correlate with the major components of variance in the data set. 
	Surrogate Variable Analysis (SVA) to identify major unwanted sources of variation. 
 	Supervised Normalization of Microarrays (SNM) to efficiently remove these sources of variation while retaining the biological factor(s) of interest.
        RUVSeq package will be used detect the hidden batch effects.
        So the data will be checked before normalization and after normalization. 
 
     Usually I prefor to use the normalized data in the next step.        

    4. Diversity analysis: Alpha diversity measures the variability of species within a sample while beta diversity accounts for the differences in compo- sition between samples. I profer to use phyloseq, vegan, metacoder
      Alpha diversity: richness, evenness
      beta diversity: ecological distances(Bray-Curtis, UniFrac and weighted UniFrac distances and Aitchison distances)
   5. Ordination: The goal of ordination plots is the visualization of beta diversity for identification of possible data structures, and retrieve significant knowledge hidden in the data. 
      Most commonly used include PCoA, NMD, and tSNE.
      PCA is a linear dimension technique while t-SNE is not linear. PCA can preserve the globe structure of data while t-SNE tries tp preserve the locao structure. t-SNE can handle outliers while PCA is highly affected. 
      It is interested in the more advanced feature selectin and feature clustering, I will build it later. As example, use PCA to reduce the number of features, in the meantime to keep 90% precent of total variance; and then use t-SNE to rcplore the local structure of data.
            

   6. Microbiome Statistical: There three big classes of statistical tests, which based on different hypothesis, and also related experiment design. 
	a. Multivariate differential abundance testing, which based on the hypothesis is there are statistical differences in microbial composition between two or more groups of experiment samples. Usually experiment contains treat/untreat group, the distance-based approch implemented in the function “anosim” of the vegan R package. The other R packages MiRKAT, and HMP also used for multivariate differential abundance testing using model-based methods. 
        b. Univariate differential abundance testing, which based on the hypothesis there are particular taxa are responsible of that global difference. However, the univariants differential abundance testing should be after Multivariate differential abundance testing had positive results.for small group of experiment design, edgeR and DEseq2 provides parametric statistical approches. As mentioned before, if no covariants has been considered in experiment design and data analysis process, false discovery should be considered when received results.
        d. Microbial signatures discovery: The microbiome analysis is aimed to the identification of microbial signatures, groups of microbial taxa that are predictive of a phenotype of interest. This is the most chellenge project. The identification of microbial signatures involves both modeling and variable selection: modeling the response variable and identifying the given taxa with the highest prediction or classification accuracy.  Poor.G et al Nature. 2020 Mar; 579:567-574 "Microbiome analyses of blood and tissues suggest cancer diagnostic approach". This paper provides very useful methodes for data integration and modeling fpr data analysis.
        e. Microbiome data is compositional because the information that abundance tables contain is relative. So there are several of R packages were used for Multivariate differential abundance testing, Multivariate differential abundance testing and Microbial signatures discovery. QIIME2 also supports methods for compositional data analysis.   
     
The data analysis need strong domain knowledge that is the main reason it is difficult to prepare one pipeline to process different data.


### Installation 
I used easybuild to install QIIME2 on HPC/AWS or my local ubuntu server. I attached easyconfig file

#### install qiime2
QIIME2 working environment can be loaded as "module load QIIME2/2022.02"


#### install R packages

The whole R base-version and required R packages have been installed with QIIME2 using the same easyconfig file, install on server together. 

### Download Reference database, and calssifier

Compared to greengene, silva could be much better for taxonomic annotation. 



### How to run 

```bash
module load qiime2-2022.02

# debug
snakemake -np -r --debug-dag result/03-denoise/table_final.qza --configfile config.yaml --snakefile Snakefile

# wrokflow figure
snakemake --dag --debug-dag result/03-denoise/table_final.qza | dot -Tsvg > workflow.svg

# run
snakemake result/03-denoise/table_final.qza --configfile config.yaml --snakefile Snakefile --cores 2
```


