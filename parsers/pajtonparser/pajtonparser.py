import argparse
import json
import pickle
import sys
from pathlib import Path

from numpy import inf
from tadpole import pond


argparser = argparse.ArgumentParser(
    prog="TadPole", description="viruses go brrr lol", epilog="Yo"
)

argparser.add_argument("phrog_dir", type=str)
argparser.add_argument("gff_dir", type=str)  # positional argument
argparser.add_argument(
    "-d", "--distance", dest="distance", type=int, default=inf, required=False
)
argparser.add_argument(
    "-o", "--output", dest="output", help="prefix/prefix-path for output files", type=str, required=True
)
argparser.add_argument(
    "--number", dest="number", help="Add numbers to jokers?", action="store_true"
)
argparser.add_argument(
    "--collapse", dest="collapse", help="Should we collapse unknown proteins into one string with prefix?", action="store_true"
)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def main():
    args = argparser.parse_args()

    loc = pond.PondLocation(Path(args.phrog_dir), Path(args.gff_dir))
    opt = pond.PondOptions(args.distance, args.number, args.collapse)

    parser = pond.PondParser(loc, opt)

    print("Started parsing...")

    res = parser.parse()

    pickle_path = f"{args.output}.pickle"
    text_path = f"{args.output}.txt"
    try:
        with open(pickle_path, "wb") as fh1, open(text_path, "w", encoding="utf-8") as fh2:
            pickle.dump(res, fh1)
            fh2.write(json.dumps(res))
    except TypeError:
        eprint("Pickle or json serialization oofed.")
        raise
    print()
    print("Done processing all files.")


if __name__ == "__main__":
    main()
