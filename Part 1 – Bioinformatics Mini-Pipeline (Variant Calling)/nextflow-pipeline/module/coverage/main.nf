process coverage {
  tag "$sample_id"
  
      publishDir "${params.outdir}/${sample_id}/coverage", mode: 'copy'

   
    input:
    tuple val(sample_id), path(bamAlignment)
 
    output:
    tuple val(sample_id), path("*.tab")
 
	script:
	
    """
	samtools coverage ${bamAlignment} > ${sample_id}_reads2hg38_coverage.tab
	samtools depth ${bamAlignment} > ${sample_id}_reads2hg38_depth.tab
    """
}