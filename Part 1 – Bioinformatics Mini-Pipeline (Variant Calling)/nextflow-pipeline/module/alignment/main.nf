process alignment {
  tag "$sample_id"
        publishDir "${params.outdir}/${sample_id}/alignment_to_hg38", mode: 'copy'

   
    input:
    tuple val(sample_id), path(reads)
	path (hostRef)
 
    output:
    tuple val(sample_id), path("*bowtie2_Stats.txt")
	tuple val(sample_id), path("*.bam"), emit : bamAlignment
 
	script:
	
	def (R1,R2) = reads
    """
	bowtie2 --local --time --threads 4 -x $params.hg38Ref -1 $R1 -2 $R2 2> ${sample_id}_bowtie2_Stats.txt | samtools sort -@ 4 -o ${sample_id}_reads2hg38_sorted.bam
    """
}