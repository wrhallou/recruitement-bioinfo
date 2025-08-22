process fastQC {
  tag "$sample_id"
  
  publishDir "${params.outdir}/${sample_id}/QC", mode: 'copy', pattern: '*.{html,zip}'
   
    input:
    tuple val(sample_id), path(reads)

	
    output:
	tuple val(sample_id), path("*.{html,zip}")
 
	
	script:
	
	def (R1,R2) = reads
	"""
	fastqc -t 16 $R1 $R2
	"""
}
