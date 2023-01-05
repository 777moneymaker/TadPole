#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

from alive_progress import alive_bar
from jsonschema import validate, ValidationError

from tadpole import pond, utils
from tadpole.logger import MAIN_LOGGER



argparser = argparse.ArgumentParser(
    prog="TadPole", description="viruses go brrr lol", epilog="Yo"
)

argparser.add_argument("phrog_dir", type=str)
argparser.add_argument("gff_dir", type=str)  # positional argument
argparser.add_argument(
    "-o", "--output", dest="output", help="prefix/prefix-path for output files", type=str, required=True
)

cfg_group = argparser.add_argument_group()
cfg_group.add_argument(
    "-p", "--props", dest = "props", help = "props location", type = str, required=False, default=None,
)
cfg_group.add_argument(
    "-c", "--config", dest = "config", help = "which protein props to add?", type = str, required=False, default=None
)

def main():
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    MAIN_LOGGER.info(f"Started program at {now}:")
    args = argparser.parse_args()

    pond_location = pond.PondLocation(Path(args.phrog_dir), Path(args.gff_dir))
    
    if args.config:
        with open(args.config) as fh, alive_bar(title = "Config...") as bar:
            config = json.load(fh)
            try:
                validate(instance = config, schema = utils.CONFIG_SCHEMA)
                if not config.get("distance"):
                    config["distance"] = float("INF") # Default inf distance if not present 
            except ValidationError as e:
                MAIN_LOGGER.error(f"Config didn't pass the jsonschema validation. Details: {e.message}")
                raise
            bar()
        if config["sequential"] == "encode" and not args.props:
            MAIN_LOGGER.critical("Config's 'sequential' prop is set to 'encode' but protein props not supplied")
            raise ValueError("Config 'sequential' is set to 'encode' but protein props not supplied")

        if not args.props:
            assert config["sequential"] != "encode", "Protein props not supplied but 'sequential' set to encode"
            print("Config supplied without protein props")
            props = None
        else:
            with open(args.props) as fh, alive_bar(title = "Props...") as bar:
                props = json.load(fh)
                bar()
            
        pond_parameters = pond.PondParams(props, config)
    else:
        print("No config supplied, using default. Protein props will not be used.")
        MAIN_LOGGER.info("No config supplied, using default")
        pond_parameters = pond.PondParams(None, utils.DEFAULT_CONFIG)

    parser = pond.PondParser(pond_location, pond_parameters)
    res = parser.parse()

    pickle_path, text_path = f"{args.output}.pickle", f"{args.output}.txt"

    try:
        with open(pickle_path, "wb") as fh1, open(text_path, "w", encoding="utf-8") as fh2:
            pickle.dump(res, fh1)
            fh2.write(json.dumps(res))
    except TypeError as e:
        MAIN_LOGGER.error(f"JSON serialization offed with message {e.message}")
        raise

    print("\nDone processing all files.")

if __name__ == "__main__":
    # main()
    main()
