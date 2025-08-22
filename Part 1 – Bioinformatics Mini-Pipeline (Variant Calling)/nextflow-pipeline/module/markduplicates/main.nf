process markdup {
  tag "$sample_id"
  
      publishDir "${params.outdir}/${sample_id}/markduplicates", mode: 'copy'

   
    input:
    tuple val(sample_id), path(bamAlignment)
 
    output:
    tuple val(sample_id), path("*.{bam,txt}")
 
	script:
	
    """
	samtools addreplacerg -r "@RG\tID:RG1\tSM:SampleName\tPL:Illumina\tLB:Library.fa" -o replaceRG_${bamAlignment} $bamAlignment
	java -jar /opt/conda/share/picard-3.4.0-0/picard.jar MarkDuplicates I=replaceRG_${bamAlignment} O=${sample_id}_dedup.bam M=${sample_id}_metrics.txt
    """
}