# TadPole 🐸 - Improving functional annotation of virus genes
## Authors
- ***ukmrs*** (https://github.com/ukmrs)
- ***777moneymaker***
- ***phenolophthaleinum*** (https://github.com/phenolophthaleinum)

## Two parsers versions:
- poteznyparser written in Rust by ***ukmrs***
- pajtonparser written in Python by ***777moneymaker*** using initial code from ***phenolophthaleinum*** and part of the design idea from ***ukmrs***

## Usage and example
```bash
>>> python parsers/pajtonparser/pajtonparser.py --help

usage: TadPole [-h] [-d DISTANCE] -o OUTPUT [--number] [--collapse] phrog_dir gff_dir

viruses go brrr lol

positional arguments:
  phrog_dir
  gff_dir

optional arguments:
  -h, --help            show this help message and exit
  -d DISTANCE, --distance DISTANCE
  -o OUTPUT, --output OUTPUT
                        prefix/prefix-path for output files
  --number              Add numbers to jokers?
  --collapse            Should we collapse unknown proteins into one string with prefix?

Yo
```
![TadPole](tadpole.gif)
