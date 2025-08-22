process trimming {
  tag "$sample_id"
  
    publishDir "${params.outdir}/${sample_id}/trimming", mode: 'copy', pattern: '*.{html,zip}'
   
    input:
    tuple val(sample_id), path(reads)

	
    output:
	tuple val(sample_id), path("*.fastq"), emit : trimmedReads
	tuple val(sample_id), path ("*.{html,json}")
 
	
	script:
	
	def (R1,R2) = reads
	"""
	fastp -i ${R1} -I ${R2} -o ${sample_id}_trimmed_R1.fastq -O ${sample_id}_trimmed_R2.fastq -l 70 -q 25 -u 50 -w 4 -j ${sample_id}_fastp.json -h ${sample_id}_fastp.html
	"""
}