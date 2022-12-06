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


class TestSamePhrogs:
    """
    test suite for ISSUE #9
    https://github.com/777moneymaker/TadPole/issues/9
    """
    phrog_dir = [TESTFILESDIR / "multipleocc/KR063268.csv"]
    gff_dir = [TESTFILESDIR / "multipleocc/KR063268.gff"]
    locations = create_pond_location(phrog_dir, gff_dir)

    def test_non_collapse(self):
        options = pond.PondOptions(float("INF"), number=False, collapse=False)
        expected = [
            [2503],
            [453, 929, -1, 1109, 13612, 306, 11271, 30486, 30486, -1, -1],
        ]
        phrogize_and_jokerize(expected)

        parser = pond.PondParser(self.locations, options)
        result = parser.parse()

        assert expected == result

    def test_collapse(self):
        options = pond.PondOptions(float("INF"), number=False, collapse=True)
        expected = [
            [2503],
            [453, 929, -1, 1109, 13612, 306, 11271, 30486, 30486, -1],
        ]
        phrogize_and_jokerize(expected)

        parser = pond.PondParser(self.locations, options)
        result = parser.parse()

        assert expected == result

