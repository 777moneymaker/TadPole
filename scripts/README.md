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

