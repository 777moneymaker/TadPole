# Scripts
Collection of scripts helpful during the project.

## make_subset.py

Creates a smaller subset of phrogs and gffs. Helpful for testing around.

### usage:
```bash
/m/t/p/p/data ❯ ./make_subset.py phrog gff 420
420/420
created: subset_of_420
```

### outcome:
```text
/m/t/p/p/data ❯ ls subset_of_420/phrog | wc                                                                                                                                              
    420     420    5577
/m/t/p/p/data ❯ ls subset_of_420/gff | wc
    420     420    5577
```
### seed
seed is saved; and can be provided for reproducible results

```text
~/m/T/scripts ❯ ./make_subset.py $PHROG_DIR $GFF_DIR 2 -s 2022 && mv subset_of_2 sub0seed2022
~/m/T/scripts ❯ ./make_subset.py $PHROG_DIR $GFF_DIR 2 -s 2022 && mv subset_of_2 sub1seed2022
~/m/T/scripts ❯ ./make_subset.py $PHROG_DIR $GFF_DIR 2 && mv subset_of_2 subseedrandom
~/m/T/scripts ❯ tree sub*
sub0seed2022
├── gff
│   ├── MT680615.gff
│   └── OM818327.gff
├── phrog
│   ├── MT680615.csv
│   └── OM818327.csv
└── seed.txt
sub1seed2022
├── gff
│   ├── MT680615.gff
│   └── OM818327.gff
├── phrog
│   ├── MT680615.csv
│   └── OM818327.csv
└── seed.txt
subseedrandom
├── gff
│   ├── MH937464.gff
│   └── MK937606.gff
├── phrog
│   ├── MH937464.csv
│   └── MK937606.csv
└── seed.txt
```
