#!/usr/bin/env python3
import shutil
import random
import argparse
from pathlib import Path
import sys


def panic(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(
        prog="make_subset",
        description="makes dirs containing subsets of phrogs and gffs",
        epilog="pozdro świry",
    )
    argparser.add_argument("phrog_dir", type=str)
    argparser.add_argument("gff_dir", type=str)
    argparser.add_argument(
        "sample_size",
        type=int,
        help="amount of phrog gff pairs that a new subset should contain",
    )
    argparser.add_argument(
        "-s", "--seed", dest="seed", type=int, default=None, required=False
    )
    return argparser.parse_args()


def get_dir_contents(directory):
    try:
        return list(directory.iterdir())
    except NotADirectoryError:
        panic(f"you lied about {directory} being a directory!!! :C")


def main():
    args = parse_args()
    phrog_dir = Path(args.phrog_dir)
    gff_dir = Path(args.gff_dir)

    phrogs = get_dir_contents(phrog_dir)
    gffs = get_dir_contents(gff_dir)
    assert len(phrogs) == len(gffs), "there is different amount of phrogs ang gffs :C"
    assert args.sample_size <= len(
        phrogs
    ), f"can't request larger sample size than the actual phrog population = {len(phrogs)}!"

    subset_dir = Path(f"subset_of_{args.sample_size}")

    try:
        subset_dir.mkdir()
    except FileExistsError:
        panic(
            f"such subset already exist, if you wanna another one rename {subset_dir} manually xd "
        )

    newphrogs = subset_dir / "phrog"
    newgffs = subset_dir / "gff"

    newphrogs.mkdir()
    newgffs.mkdir()

    phrogs.sort()
    gffs.sort()

    seed = random.randrange(sys.maxsize) if args.seed is None else args.seed
    random.seed(seed)

    with open(subset_dir / "seed.txt", "w") as f:
        f.write(f"{seed}\n")

    samples = random.sample(range(0, len(phrogs)), args.sample_size)

    for idx, sample in enumerate(samples, start=1):
        gff_to_copy = gffs[sample]
        phrog_to_copy = phrogs[sample]

        shutil.copy(phrog_to_copy, newphrogs / phrog_to_copy.name)
        shutil.copy(gff_to_copy, newgffs / gff_to_copy.name)
        print(f"\r{idx}/{args.sample_size}", end="")
    print()
    print(f"created: {subset_dir}")


if __name__ == "__main__":
    main()
