files with multiple occurnces of the same phrog_id corresponding to the same prot_id

these are based on:

```text
KR063268.csv
KR063268.gff
```

but I added the following to KR063268.gff:

```text
KR063268.1	Prodigal_v2.6.3	CDS	5050	6020	79.3	+	0	ID=KR063268|9;partial=00;start_type=ATG;rbs_motif=GGA/GAG/AGG;rbs_spacer=5-10bp;gc_cont=0.497;conf=100.00;score=80.55;cscore=71.60;sscore=8.96;rscore=2.00;uscore=0.89;tscore=4.77;
KR063268.1	Prodigal_v2.6.3	CDS	6021	7021	79.3	+	0	ID=KR063268|10;partial=00;start_type=ATG;rbs_motif=GGA/GAG/AGG;rbs_spacer=5-10bp;gc_cont=0.497;conf=100.00;score=80.55;cscore=71.60;sscore=8.96;rscore=2.00;uscore=0.89;tscore=4.77;
```

to create two consecutive jokers
