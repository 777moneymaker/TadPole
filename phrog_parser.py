import argparse
import pandas as pd
import gffpandas.gffpandas as gffpd
from io import StringIO
from pathlib import Path
import tempfile

parser = argparse.ArgumentParser(
                    prog = 'TadPole',
                    description = 'viruses go brrr lol',
                    epilog = 'Yo')

parser.add_argument('phrog_dir', type=str)
parser.add_argument('gff_dir', type=str)           # positional argument
parser.add_argument('-d', '--max-dist', dest="max_dist", type=int, default = 1, required = False)      # option that takes a value
parser.add_argument('-a', '--add-number', dest="add_number", type=bool, required = False, default = True)
parser.add_argument('-c', '--collapse', dest="collapse", type=bool, required = False, default = False)


def parse_phrog(phrog_dir: Path, gff_dir: Path, max_dist: int, add_number: bool = True, collapse: bool = False) -> list[list]:
    """
    :param gff_dir: Dir containing gff files
    :param phrog_dir: Dir containing phrog files
    :param max_dist: Describes how big is the distance between two words
    """
    if not isinstance(gff_dir, Path):
        raise TypeError("Gff dir is not a Path obj")
    if not isinstance(phrog_dir, Path):
        raise TypeError("Phrog dir is not a Path obj")
    if not isinstance(max_dist, (int, float, complex)):
        raise TypeError("max_dist is not an int")
    if not isinstance(collapse, bool):
        raise TypeError("collapse should be a bool obj")
    if not isinstance(add_number, bool):
        raise TypeError("add_number should be a bool obj")
    if not phrog_dir.is_dir():
        raise TypeError("Phrog dir is not dir")
    if not gff_dir.is_dir():
        raise TypeError("Gff dir is not dir")
    
    unknown_prot = "joker"
    gff_files = list(gff_dir.iterdir())
    phrog_files = list(phrog_dir.iterdir())

    gff_files.sort()
    phrog_files.sort()


    for gff_file, phrog_file in zip(gff_files, phrog_files):
        with open(gff_file, "r") as fh:
            text = "".join(line.rstrip(";") for line in fh)
        with tempfile.TemporaryDirectory() as td:
            f_name = Path(td) / 'test'
            with open(f_name, 'w') as fh:
                fh.write(text)
            gff_data = gffpd.read_gff3(fh).attributes_to_columns()
        
        phrogs = pd.read_csv(phrog_file)
        phrogs = phrogs.rename(columns={"prot_id": "ID"})
        
        df_phrogs = pd.merge(phrogs, gff_data, how="right", on="ID")[["ID", "hmm_id", "seq_id", "start", "end", "strand"]]
        df_phrogs.fillna(unknown_prot, inplace=True)

        g = df_phrogs.groupby((df_phrogs['strand'].shift() != df_phrogs["strand"]).cumsum())
        names = g.apply(lambda x: x.iloc[::-1] if x["strand"].iloc[0] == "-" else x)['hmm_id'].to_list()
        line = ' '.join(names)
        print(line)

def main():
    args = parser.parse_args()
    phrog_dir = Path(args.phrog_dir)
    gff_dir = Path(args.gff_dir)
    parse_phrog(phrog_dir=phrog_dir, gff_dir=gff_dir, max_dist=args.max_dist, add_number=args.add_number, collapse=args.collapse)


if __name__ == "__main__":
    main()