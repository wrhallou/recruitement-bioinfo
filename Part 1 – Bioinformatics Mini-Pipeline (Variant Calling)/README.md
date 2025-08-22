# Intro
- I chose to make the mini-pipeline using workflow manager language nextflow + singularity for containers. So both need to be installed on the machine to run the pipeline.
- The main worflow contains 5 processes : fastQC, trimming, alignment, coverage, markdup
- Tools used in the mini-pipeline : FastQC, fastp, bowtie2, samtools, picard tools. They are standad well documented tools used in the bioinformatics community.
- If not installed on the machine download the singularity image and put it in the directory nextflow-pipeline/containers. Download with : 
 wget https://community-cr-prod.seqera.io/docker/registry/v2/blobs/sha256/55/5541b7ffb62eefc1234757e4ce7d5cb7348e020bbfe1d08bbe643cc997d3c335/data -O nextflow-pipeline/containers/container.sif
- hg38 reference needs to be downloaded and an index for bowtie2 created .

# Mini-Pipeline structure
- The pipeline perfomrs QC anlaysis and trimming; then alignment on hg38 human genome reference; then extracts information about alignment like coverage, percentage of alignement; and finally marking duplcates is performed.
- The pipeline should be compatible with most linux command lines shells.
- To run the pipeline (use full paths ) :
      nextflow run nextflow-pipeline/workflows/variant-analysis_workflow -c nextflow-pipeline/conf/base.conf -profile singularity --rawreads [path/to/rawreads/*_R{1,2}.fastq.gz] --outdir [path/to/put/dir] --hg38Ref [path/to/bowtie2/reference]

# Summarry of results
A summary of results is provided with multiQC tool. File in the main directory of part1.

#Description of continuig variant call analysis
For germline calling, I’d use GATK HaplotypeCaller. If these samples were tumor/normal pairs, I’d instead use GATK Mutect2 with panel-of-normals and additional somatic filtering steps.

- First i would do a Base Quality Score Recalibration (BQSR) with GATK using known sites (dbSNP, Mills indels).

- Then perfomr the variant calling per-sample with gatk HaplotypeCaller -ERC GVCF; and joint genotyping with gatk GenotypeGVCFs.

- Finally variant filtering with threshold filters since not enough samples are available to use VQSR.
  
