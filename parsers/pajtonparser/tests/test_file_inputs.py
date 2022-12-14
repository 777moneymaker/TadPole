#!/usr/bin/env python3
from tadpole import pond
from pathlib import Path
from typing import List
import sys
import pytest

TESTDIR = Path(__file__).parent
TESTFILESDIR = TESTDIR / "helpfiles"


def eprint(args, **kwargs):
    print(args, **kwargs, file=sys.stderr)


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
            if val > 0:
                sub[idx] = f"phrog_{val}"
            elif val == 0:
                sub[idx] = "joker"
            else:
                sub[idx] = f"joker{val * -1}"


def create_pond_location(phrog_dir, gff_dir) -> pond.PondLocation:
    # bypasses Miłosz's merciless checks by inputing real directories first
    locations = pond.PondLocation(TESTDIR, TESTDIR)
    locations.phrog_dir = ImposterDir(phrog_dir)
    locations.gff_dir = ImposterDir(gff_dir)
    return locations


@pytest.mark.skip(reason="The content of the test should be verified on meeting")
def test_distance():
    phrog_dir = [TESTFILESDIR / "test_files/4.only_plus_strand.csv"]
    gff_dir = [TESTFILESDIR / "test_files/4.only_plus_strand.gff"]
    locations = create_pond_location(phrog_dir, gff_dir)
    options = pond.PondOptions(distance=50, number=False, collapse=False)

    expected = [[8855, 1199, 1646, 8452, 1944, 1675, 1012, 702]]
    phrogize_and_jokerize(expected)

    parser = pond.PondParser(locations, options)
    result = parser.parse()

    assert expected == result


def test_spliting_words_by_symbol():
    expected = [[2199], [239], [21102, 25990]]
    phrogize_and_jokerize(expected)

    phrog_dir = [TESTFILESDIR / "test_files/2.one_plus_strand.csv"]
    gff_dir = [TESTFILESDIR / "test_files/2.one_plus_strand.gff"]
    locations = create_pond_location(phrog_dir, gff_dir)
    options = pond.PondOptions(distance=float("INF"), number=False, collapse=False)

    parser = pond.PondParser(locations, options)
    result = parser.parse()

    assert expected == result


def test_all_same_jokers_6th_times_colapse():
    expected = [["joker"]]

    phrog_dir = [TESTFILESDIR / "test_files/1.with_unknown_phrog_same_location.csv"]
    gff_dir = [TESTFILESDIR / "test_files/1.with_unknown_phrog_same_location.gff"]
    locations = create_pond_location(phrog_dir, gff_dir)
    options = pond.PondOptions(distance=float("INF"), number=False, collapse=True)

    parser = pond.PondParser(locations, options)
    result = parser.parse()

    assert expected == result


def test_only_minus_strand_to_check_reverse_func():
    expected = [[21102, 25990, 2199]]
    phrogize_and_jokerize(expected)

    phrog_dir = [TESTFILESDIR / "test_files/3.only_minus_strand.csv"]
    gff_dir = [TESTFILESDIR / "test_files/3.only_minus_strand.gff"]
    locations = create_pond_location(phrog_dir, gff_dir)
    options = pond.PondOptions(distance=float("INF"), number=False, collapse=False)

    parser = pond.PondParser(locations, options)
    result = parser.parse()

    assert expected == result


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
            [453, 929, 0, 1109, 13612, 306, 11271, 30486, 30486, 0, 0],
        ]
        phrogize_and_jokerize(expected)

        parser = pond.PondParser(self.locations, options)
        result = parser.parse()

        assert expected == result

    def test_collapse(self):
        options = pond.PondOptions(float("INF"), number=False, collapse=True)
        expected = [
            [2503],
            [453, 929, 0, 1109, 13612, 306, 11271, 30486, 30486, 0],
        ]
        phrogize_and_jokerize(expected)

        parser = pond.PondParser(self.locations, options)
        result = parser.parse()

        assert expected == result


class TestMultipleUnknown:
    """tests behavior with multiple unknown consecutive proteins
    the test files contain four unknown streaks:
    Second sentence:
    1 unknown
    3 consecutive unknowns
    2 consecutive unknowns
    Third Sentence:
    1 unknown only
    Fourth Sentence:
    2 unknowns only"""

    phrog_dir = [TESTFILESDIR / "multiple_unknown/KR063268.csv"]
    gff_dir = [TESTFILESDIR / "multiple_unknown/KR063268.gff"]
    locations = create_pond_location(phrog_dir, gff_dir)

    def test_vanilla(self):
        options = pond.PondOptions(
            float("INF"), number=False, collapse=False, consecutive=False
        )

        expected = [
            [2503],
            [453, 929, 0, 1109, 13612, 306, 11271, 30486, 30486, 0, 0, 0, 69, 0, 0, 420],
            [0],
            [0, 0],
        ]
        phrogize_and_jokerize(expected)

        parser = pond.PondParser(self.locations, options)
        result = parser.parse()

        assert expected == result

    def test_collapse(self):
        options = pond.PondOptions(float("INF"), number=False, collapse=True)
        expected = [
            [2503],
            [453, 929, 0, 1109, 13612, 306, 11271, 30486, 30486, 0, 69, 0, 420],
            [0],
            [0],
        ]
        phrogize_and_jokerize(expected)

        parser = pond.PondParser(self.locations, options)
        result = parser.parse()

        assert expected == result

    def test_number(self):
        options = pond.PondOptions(float("INF"), number=True, collapse=False)
        expected = [
            [2503],
            [
                453,
                929,
                -1,
                1109,
                13612,
                306,
                11271,
                30486,
                30486,
                -2,
                -3,
                -4,
                69,
                -5,
                -6,
                420,
            ],
            [-7],
            [-8, -9],
        ]
        phrogize_and_jokerize(expected)

        parser = pond.PondParser(self.locations, options)
        result = parser.parse()

        assert expected == result

    def test_consecutive(self):
        options = pond.PondOptions(
            float("INF"), number=False, collapse=False, consecutive=True
        )

        expected = [
            [2503],
            [453, 929, -1, 1109, 13612, 306, 11271, 30486, 30486, -3, 69, -2, 420],
            [-1],
            [-2],
        ]
        phrogize_and_jokerize(expected)

        parser = pond.PondParser(self.locations, options)
        result = parser.parse()

        eprint(expected)
        eprint('--')
        eprint(result)
        eprint('--')
        assert expected == result


class TestArgExclusivity:
    def test_number_and_consecutive(self):
        with pytest.raises(ValueError):
            pond.PondOptions(
                float("INF"), number=True, collapse=False, consecutive=True
            )

    def test_all(self):
        with pytest.raises(ValueError):
            pond.PondOptions(float("INF"), number=True, collapse=True, consecutive=True)
