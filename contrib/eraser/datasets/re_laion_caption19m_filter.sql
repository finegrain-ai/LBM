COPY (
    SELECT 
        *
    FROM parquet_scan('./data/supermodelresearch/Re-LAION-Caption19M/laion_19m.parquet')
    WHERE (
        pwatermark < 0.2 AND
        aesthetic_score > 5.6
    )
) TO './data/Re-LAION-1300K-parquet/' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);