import argparse
import json
import sys
import tempfile
import pickle
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np
import gffpandas.gffpandas as gffpd

parser = argparse.ArgumentParser(
    prog="TadPole", description="viruses go brrr lol", epilog="Yo"
)

parser.add_argument("phrog_dir", type=str)
parser.add_argument("gff_dir", type=str)  # positional argument
parser.add_argument(
    "-d", "--distance", dest="distance", type=int, default=np.inf, required=False
)
parser.add_argument(
    "-o", "--output", dest="output", help="prefix/prefix-path for output files", type=str, required=True
)
parser.add_argument(
    "--number", dest="number", help="Add numbers to jokers?", action="store_true"
)
parser.add_argument(
    "--collapse", dest="collapse", help="Should we collapse unknown proteins into one string with prefix?", action="store_true"
)

@dataclass(frozen=True)
class PhrogLocation:
    """Object containing gff and phrog directories paths"""
    phrog_dir: Path
    gff_dir: Path

@dataclass(frozen=True)
class PhrogOptions:
    """Object containig metadata for phrog and gff parser"""
    distance: int
    number: bool
    collapse: bool

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def parse_phrog(location: PhrogLocation, options: PhrogOptions) -> list[list[str]]:
    """Parse phrog files with their respective gff options into list of phrogs.

    Args:
        location (PhrogOptions): Phrog location obj containing dir to read data from.
        options (PhrogOptions): Phrog options obj containing parsing options.
    Returns:
        list of sentences (lists) containing phrogs
    Raises:
        AssertionError: If parsed files are in wrong format/length.
    """
    # Check if objs are valid
    if not isinstance(location.gff_dir, Path):
        raise TypeError("Gff dir is not a Path obj")
    if not isinstance(location.phrog_dir, Path):
        raise TypeError("Phrog dir is not a Path obj")
    if not isinstance(options.max_dist, (int, float, complex)):
        raise TypeError("max_dist is not an int")
    if not isinstance(options.collapse, bool):
        raise TypeError("collapse should be a bool obj")
    if not isinstance(options.add_number, bool):
        raise TypeError("add_number should be a bool obj")
    if not location.phrog_dir.is_dir():
        raise TypeError("Phrog dir is not dir")
    if not location.gff_dir.is_dir():
        raise TypeError("Gff dir is not dir")

    unknown_prot = "joker" # Name for unknown
    gff_files = sorted(location.gff_dir.iterdir())
    phrog_files = sorted(location.phrog_dir.iterdir())
    file_counter = 1 # Count files

    # Outer list containing sentences
    paragraph: list = []
    for gff_file, phrog_file in zip(gff_files, phrog_files):
        # Remove ';' from right side of lines
        with open(gff_file, "r") as fh:
            text = "".join(line.rstrip(";") for line in fh)
        # Create tmp dir to save fixed file and parse fixed gff from it
        with tempfile.TemporaryDirectory() as td:
            f_name = Path(td) / "test"
            with open(f_name, "w") as fh:
                fh.write(text)
            gff_data = gffpd.read_gff3(f_name).attributes_to_columns()

        # Read phrogs
        phrogs = pd.read_csv(phrog_file)
        phrogs.rename(columns={"prot_id": "ID"}, inplace=True)
        
        # Sql Join
        df_phrogs = pd.merge(phrogs, gff_data, how="right", on="ID")[
            ["hmm_id", "start", "end", "strand"]
        ]
        # Replace NAs with jokers
        df_phrogs.fillna(unknown_prot, inplace=True)

        # Extract cols
        phrogs = df_phrogs["hmm_id"].values.tolist()
        strands = df_phrogs["strand"].values.tolist()
        starts = df_phrogs["start"].values.tolist()[1:]
        ends = df_phrogs["end"].values.tolist()[:-1]

        # Check trivial conditions just to make sure
        assert len(phrogs) == len(
            strands), "Len of phrogs and strands is wrong. Oof"

        dists = [float(s) - float(e) for s, e in zip(starts, ends)]
        # Add zero so each dist on index 'i' is a distance of 'i' phrog to previous phrog
        # So - dist of first phrog is 0 because it's first etc...
        dists.insert(0, 0) 

        # Again trivial assertion
        assert len(dists) == len(phrogs), "Len of dists is not valid"

        sentence: list = []
        i: int = 0
        prev_strand: str = strands[0]
        for i, _ in enumerate(phrogs):
            # If strand changed or dist is too big then
            # push the current sentence and start a new one (clear list)
            if strands[i] != prev_strand or dists[i] > options.max_dist:
                paragraph.append(sentence if prev_strand ==
                                 '+' else list(reversed(sentence)))
                sentence = []
            sentence.append(phrogs[i])
            prev_strand = strands[i]

        # Push last slice which for loop didnt pushed
        paragraph.append(sentence if prev_strand ==
                         '+' else list(reversed(sentence)))
        print(f"Done {file_counter}/{len(phrog_files)}", end="\r")

    if options.collapse:
        for i in range(len(paragraph)):
            prev = object()
            # Remove consecutive duplicated unknown jokers
            paragraph[i] = [prev := x for x in paragraph[i] if prev != x]
    if options.add_number:
        unknown_counter: int = 1
        for i in range(len(paragraph)):
            for j in range(len(paragraph[i])):
                if paragraph[i][j] == unknown_prot:
                    paragraph[i][j] = unknown_prot + str(unknown_counter)
                    unknown_counter += 1

    return paragraph


def main():
    args = parser.parse_args()
    phrog_dir = Path(args.phrog_dir)
    gff_dir = Path(args.gff_dir)
    print("Started parsing...")

    loc = PhrogLocation(phrog_dir, gff_dir)
    meta = PhrogOptions(args.distance, args.number, args.collapse)

    res = parse_phrog(loc, meta)

    pickle_path = f"{args.output}.pickle"
    text_path = f"{args.output}.txt"
    try:
        with open(pickle_path, "wb") as fh1, open(text_path, "w") as fh2:
            pickle.dump(res, fh1)
            fh2.write(json.dumps(res))
    except TypeError:
        eprint("Pickle or json serialization oofed.")
        raise

    print("Done processing all files.")


if __name__ == "__main__":
    main()
