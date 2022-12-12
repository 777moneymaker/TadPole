import argparse
import json
import sys
from pathlib import Path
import pickle

from alive_progress import alive_bar
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


group = argparser.add_mutually_exclusive_group()
group.add_argument(
    "--number", dest="number", help="Add numbers to jokers?", action="store_true"
)
group.add_argument(
    "--collapse", dest="collapse", help="Should we collapse unknown proteins into one string with prefix?", action="store_true"
)

cfg_group = argparser.add_argument_group()
cfg_group.add_argument(
    "-p", "--props", dest = "props", help = "props location", type = str, required=False, default=None,
)
cfg_group.add_argument(
    "-c", "--config", dest = "config", help = "which protein props to add?", type = str, required=False, default=None
)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def main():
    args = argparser.parse_args()
    if (args.props and not args.config) or (not args.props and args.config):
        raise ValueError("Props and config must always occur together")

    loc = pond.PondLocation(Path(args.phrog_dir), Path(args.gff_dir))
    opt = pond.PondOptions(args.distance, args.number, args.collapse)
    props = args.props

    config = None
    if args.props:
        with open(args.props) as fh, open(args.config) as fhc, alive_bar(title = "Prot params"):
            props = json.load(fh)
            config = json.load(fhc)
        valid_cfg = {'molecular_weight', 'instability_index', 'isoelectric_point', 'gravy', 'aromaticity'}
        if any(key not in valid_cfg for key in config):
            raise ValueError("Config contains invalid entries")
        if any(not isinstance(value, bool) for value in config.values()):
            raise ValueError("Config values should be bool's only")
        config = pond.PondConfig(props, config) 

    parser = pond.PondParser(loc, opt)
    res = parser.parse(pond_config = config)

    pickle_path = f"{args.output}.pickle"
    text_path = f"{args.output}.txt"
    try:
        with open(pickle_path, "wb") as fh1, open(text_path, "w", encoding="utf-8") as fh2:
            pickle.dump(res, fh1)
            fh2.write(json.dumps(res))
    except TypeError:
        eprint("Dill or json serialization oofed.")
        raise
    print()
    print("Done processing all files.")

if __name__ == "__main__":
    # main()
    main()
