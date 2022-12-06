#!/usr/bin/env python3
from tadpole import pond
from pathlib import Path
from typing import List

TESTDIR = Path(__file__).parent
TESTFILESDIR = TESTDIR / "helpfiles"


class ImposterDir:
    def __init__(self, files) -> None:
        self.files: List = files

    def iterdir(self):
        yield from self.files

    def isdir(files):
        return True  # yes of course I am dir c:


def phrogize_and_jokerize(nested_list: List[List[int]]):
    for sub in nested_list:
        for idx, val in enumerate(sub):
            if val == -1:
                sub[idx] = "joker"
            else:
                sub[idx] = f"phrog_{val}"


def create_pond_location(phrog_dir, gff_dir) -> pond.PondLocation:
    # bypasses miłosz's merciless checks by inputing real directories first
    locations = pond.PondLocation(TESTDIR, TESTDIR)
    locations.phrog_dir = ImposterDir(phrog_dir)
    locations.gff_dir = ImposterDir(gff_dir)
    return locations


def test_same_phrogs_to_one_prot_id():
    phrog_dir = [TESTFILESDIR / "multipleocc/KR063268.csv"]
    gff_dir = [TESTFILESDIR / "multipleocc/KR063268.gff"]
    # bypass miłosz's merciless checks
    locations = create_pond_location(phrog_dir, gff_dir)

    options = pond.PondOptions(float("INF"), False, False)
    expected = [[2503], [453, 929, -1, 1109, 13612, 306, 117271, 30486, 30486]]
    phrogize_and_jokerize(expected)

    parser = pond.PondParser(locations, options)
    result = parser.parse()

    assert expected == result
