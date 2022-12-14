import argparse
import json
import pickle
import sys
from pathlib import Path

from alive_progress import alive_bar
from jsonschema import validate, ValidationError
from numpy import inf

from tadpole import pond, utils
from tadpole.logger import MAIN_LOGGER



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
        MAIN_LOGGER.error("Props and config not supplied together")
        raise ValueError("Props and config must always occur together")

    loc = pond.PondLocation(Path(args.phrog_dir), Path(args.gff_dir))
    opt = pond.PondOptions(args.distance, args.number, args.collapse)
    
    if args.config:
        with open(args.props) as fh, open(args.config) as fhc, alive_bar(title = "Props & cfg ") as bar:
                config = json.load(fhc)
                bar()
                try:
                    validate(instance = config, schema = utils.config_schema)
                except ValidationError as e:
                    eprint(f"Config was not validated succesfully. Details: {e.message}")
                    MAIN_LOGGER.error(f"Config didn't pass the jsonschema validation. Details: {e.message}")
                    exit(-1)
                bar()
                props = json.load(fh)
                bar()
                pond_config = pond.PondConfig(props, config)
    

    parser = pond.PondParser(loc, opt)
    res = parser.parse(pond_config = pond_config)

    pickle_path, text_path = f"{args.output}.pickle", f"{args.output}.txt"

    try:
        with open(pickle_path, "wb") as fh1, open(text_path, "w", encoding="utf-8") as fh2:
            pickle.dump(res, fh1)
            fh2.write(json.dumps(res))
    except TypeError as e:
        eprint(f"JSON serialization oofed with message {e.message}")
        MAIN_LOGGER.error(f"JSON serialization offed with message {e.message}")
        exit(-1)

    print("\nDone processing all files.")

if __name__ == "__main__":
    # main()
    main()
