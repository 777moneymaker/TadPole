files with multiple consecutive unknown proteins 

these are based on:

```text
KR063268.csv
KR063268.gff
```

but I added the following to KR063268.gff:

```text

KR063268.1	Prodigal_v2.6.3	CDS	5050	6020	79.3	+	0	ID=KR063268|9;partial=00;start_type=ATG;rbs_motif=GGA/GAG/AGG;rbs_spacer=5-10bp;gc_cont=0.497;conf=100.00;score=80.55;cscore=71.60;sscore=8.96;rscore=2.00;uscore=0.89;tscore=4.77;
KR063268.1	Prodigal_v2.6.3	CDS	6021	7021	79.3	+	0	ID=KR063268|10;partial=00;start_type=ATG;rbs_motif=GGA/GAG/AGG;rbs_spacer=5-10bp;gc_cont=0.497;conf=100.00;score=80.55;cscore=71.60;sscore=8.96;rscore=2.00;uscore=0.89;tscore=4.77;
KR063268.1	Prodigal_v2.6.3	CDS	7023	8033	79.3	+	0	ID=KR063268|11;partial=00;start_type=ATG;rbs_motif=GGA/GAG/AGG;rbs_spacer=5-10bp;gc_cont=0.497;conf=100.00;score=80.55;cscore=71.60;sscore=8.96;rscore=2.00;uscore=0.89;tscore=4.77;
KR063268.1	Prodigal_v2.6.3	CDS	8044	9033	79.3	+	0	ID=KR063268|12;partial=00;start_type=ATG;rbs_motif=GGA/GAG/AGG;rbs_spacer=5-10bp;gc_cont=0.497;conf=100.00;score=80.55;cscore=71.60;sscore=8.96;rscore=2.00;uscore=0.89;tscore=4.77;
KR063268.1	Prodigal_v2.6.3	CDS	9053	9233	79.3	+	0	ID=KR063268|13;partial=00;start_type=ATG;rbs_motif=GGA/GAG/AGG;rbs_spacer=5-10bp;gc_cont=0.497;conf=100.00;score=80.55;cscore=71.60;sscore=8.96;rscore=2.00;uscore=0.89;tscore=4.77;
KR063268.1	Prodigal_v2.6.3	CDS	9523	9633	79.3	+	0	ID=KR063268|14;partial=00;start_type=ATG;rbs_motif=GGA/GAG/AGG;rbs_spacer=5-10bp;gc_cont=0.497;conf=100.00;score=80.55;cscore=71.60;sscore=8.96;rscore=2.00;uscore=0.89;tscore=4.77;
KR063268.1	Prodigal_v2.6.3	CDS	9783	9999	79.3	+	0	ID=KR063268|15;partial=00;start_type=ATG;rbs_motif=GGA/GAG/AGG;rbs_spacer=5-10bp;gc_cont=0.497;conf=100.00;score=80.55;cscore=71.60;sscore=8.96;rscore=2.00;uscore=0.89;tscore=4.77;

```

I also fabricated these phrogs

```text
KR063268|12,phrog_69,237,393,3.6e-104,349.6
KR063268|15,phrog_420,237,393,3.6e-104,349.6
```

to create two consecutive jokers
