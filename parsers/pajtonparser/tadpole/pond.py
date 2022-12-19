#!/usr/bin/env python3
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from enum import Enum
from alive_progress import alive_bar
from alive_progress.animations.spinners import bouncing_spinner_factory
from Bio import SeqIO, SeqUtils
from Bio.SeqUtils import ProtParam
from threading import Thread

from .logger import POND_LOGGER

PHROG_SPINNER = bouncing_spinner_factory(("🐸", "🐸"), 8, background = ".", hide = False, overlay =True)

class PondThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class Strand(Enum):
    NEG = 0
    POS = 1

    @classmethod
    def into(cls, val: str):
        if not isinstance(val, str):
            POND_LOGGER.error(f"Wrong strand type '{type(val)}' provided")
            raise TypeError(f"Into strand should have a 'str' type but {type(val)} provided.")
        
        if val not in ("+", "-"):
            POND_LOGGER.error(f"Wrong strand character '{val}' provided")
            raise ValueError(f"Valid values for Strand.into() are ['+', '-'] but '{val}' provided.")

        return cls.POS if val == "+" else cls.NEG


@dataclass
class PondLocation:
    """Object containing gff and phrog directories paths"""
    phrog_dir: Path
    gff_dir: Path

    def __init__(self, phrog_dir: Path, gff_dir: Path):
        if not isinstance(phrog_dir, Path):
            POND_LOGGER.error(f"phrog dir provided as type '{type(phrog_dir)}' but Path required")
            raise TypeError("Phrog dir is not a Path obj")
        if not isinstance(gff_dir, Path):
            POND_LOGGER.error(f"gff dir provided as type '{type(gff_dir)}' but Path required")
            raise TypeError("Gff dir is not a Path obj")

        if not phrog_dir.is_dir():
            POND_LOGGER.error(f"Phrog dir is not dir")
            raise TypeError("Phrog dir is not dir")
        if not gff_dir.is_dir():
            POND_LOGGER.error(f"GFF dir is not a dir")
            raise TypeError("Gff dir is not dir")

        self.phrog_dir = phrog_dir
        self.gff_dir = gff_dir

@dataclass
class PondOptions:
    """Object containig metadata for phrog and gff parser"""
    distance: int
    number: bool
    collapse: bool
    consecutive: bool

    def __init__(self, distance: int, number=False, collapse=False, consecutive=False):
        if number + collapse + consecutive > 1:
            raise ValueError("number/collapse/consecutive options are exclusive")

        if not isinstance(distance, (int, float, complex)):
            POND_LOGGER.error(f"distance provided as type '{type(distance)}' but int required")
            raise TypeError("distance is not an int")
        if distance < 0:
            POND_LOGGER.error(f"distance < 0 provided")
            raise ValueError("Distance < 0 is not valid")
        if not isinstance(collapse, bool):
            POND_LOGGER.error(f"collapse provided as type '{type(collapse)}' but bool required")
            raise TypeError("collapse should be a bool obj")
        if not isinstance(number, bool):
            POND_LOGGER.error(f"number provided as type '{type(number)}' but bool required")
            raise TypeError("number should be a bool obj")
        if not isinstance(consecutive, bool):
            POND_LOGGER.error(f"consecutive provided as type '{type(consecutive)}' but bool required")
            raise TypeError("consecutive should be a bool obj")
        
        
        self.distance = distance
        self.number = number
        self.collapse = collapse
        self.consecutive = consecutive

@dataclass
class PondConfig:
    params: dict[dict[str, float]] # 'Protein_ID': { 'chemical_property': 'value' }
    config: dict[bool] # 'chemical_property': True/False

@dataclass
class PondRecord:
    id: str
    phrogs: list[str]
    start: int
    end: int
    strand: Strand
    dist: int = 0 # Defaulted to be used as optional arg

class PondMap():
    def __init__(self):
        """TODO: Implement this?"""
        self.d = defaultdict(list)

    def __getitem__(self, key):
        return self.d[key]

    def __setitem__(self, key, value):
        if not isinstance(value, str):
            POND_LOGGER.error(f"In PondMap type '{type(value)}' wrongly supplied")
            raise TypeError("PondMap values should be str's only.")
        self.d[key].append(value)

    def clear(self):
        self.d.clear()

class PondParser:
    def __init__(self, location: PondLocation, options: PondOptions, unknown: str = "joker"):
        self.location = location
        self.options = options
        self.map = PondMap()
        self.records = []
        self.unknown = unknown
        self.content = None

    def _fill_map(self):
        files = list(self.location.phrog_dir.iterdir())
        with alive_bar(len(files), title = "PHROGs      ",  dual_line = True, spinner = PHROG_SPINNER) as bar:
            bar.text = "--> Looping on some PHROGs"
            for file in files:
                with open(file, encoding="utf-8") as fh:
                    next(fh)
                    for line in fh:
                        prot, phrog = line.split(",")[:2]
                        self.map[prot].append(phrog)
                bar()
            POND_LOGGER.info("Processed all available phrog files")


    @staticmethod
    def parse_proteins(faa_dir: Path) -> dict[dict]:
        if not isinstance(faa_dir, Path):
            POND_LOGGER.error("Faa dir supplied not as Path obj")
            raise TypeError(".faa dir is not a Path obj")
        if not faa_dir.is_dir():
            POND_LOGGER.error("Faa dir supplied not as dir")
            raise TypeError(".faa dir is not dir")
        
        files = list(faa_dir.iterdir())
        prot_props = dict()
        with alive_bar(len(files), title = "FAAs        ", dual_line = True, spinner = PHROG_SPINNER) as bar:
            bar.text = "--> Parsing FAAs"
            for file in files:
                with open(file, encoding = "utf-8") as fh:
                    for record in SeqIO.parse(fh, "fasta"):
                        id = str(record.id)
                        seq = str(record.seq).replace("*", "").replace("X", "")
                        prot_param = ProtParam.ProteinAnalysis(seq) 
                        # prot_iso = IsoelectricPoint.IsoelectricPoint(seq)
                        props = {
                            "molecular_weight": prot_param.molecular_weight(),
                            "aromaticity": prot_param.aromaticity(),
                            "instability_index": prot_param.instability_index(),
                            "gravy": prot_param.gravy(),
                            "isoelectric_point": prot_param.isoelectric_point()
                        }
                        prot_props[id] = props              
                bar()
            POND_LOGGER.info("Processed all available faa files.")
        return prot_props

    def collapse_consecutive(self, lst: list[str]) -> list[str]:
        """This method was generated using OpenAI API.

        Prompt used:
        '''
        Lets say we have list [a, b, c, d, joker, joker, joker, g, joker, joker, a, d, joker].
        Create a python code that will collapse all consecutive jokers and add a number indicating how many jokers were collapsed. Return a list.

        Like this: [a, b, c, d, joker3, g, joker2, a, d, joker1]
        '''
        """
        new_lst = []
        joker_count = 0
        for phrog in lst:
            if phrog == self.unknown:
                joker_count += 1
            else:
                if joker_count > 0:
                    new_lst.append(f'{self.unknown}{joker_count}')
                    joker_count = 0
                new_lst.append(phrog)
        if joker_count > 0:
            new_lst.append(f'{self.unknown}{joker_count}')
        return new_lst


    def parse(self, pond_config: PondConfig = None) -> list[list[str]]:
        self._fill_map()
        Sentence = list[str]
        
        files = list(self.location.gff_dir.iterdir())
        with alive_bar(len(files), title = "GFFs        ", dual_line = True, spinner = PHROG_SPINNER) as bar:
            bar.text = "--> Parsing GFFs"
            for file in files:
                with open(file, encoding="utf-8") as fh:
                    predicate = lambda x: not (x.startswith("#") or x.strip() == "")
                    lines = filter(predicate, fh)
                    for j, line in enumerate(lines):
                        *_, start, end, _, strand, _, prot = line.split("\t")
                        start, end = int(start), int(end)
                        prot = prot.split(";", maxsplit=1)[0].lstrip("ID=")
                        phrogs = self.map[prot]
                        strand = Strand.into(strand)
                        if not phrogs:
                            phrogs = ["joker"]
                            if pond_config is not None: 
                                params = pond_config.params.get(prot)
                                if params:
                                    phrogs = [
                                        "joker_" + "|".join(f"{key}:{val :.2f}" for key, val in params.items() if pond_config.config[key])
                                    ]
                        dist = 0 if j == 0 else start - self.records[j - 1].end
                        record = PondRecord(prot, phrogs, start, end, strand, dist)
                        self.records.append(record)
                bar()
            POND_LOGGER.info("Processed all availabe gff files")
        records: list[PondRecord] = iter(self.records)
        previous: PondRecord = next(records)
        sentence: Sentence = []
        paragraph: list[Sentence] = []

        # Add first phrogs
        sentence.extend(previous.phrogs)
        counter: int = 1
        with alive_bar(len(self.records), title = "Sentences   ", dual_line = True, spinner = PHROG_SPINNER) as bar:
            bar.text = "--> Ordering some sentences"
            for record in records:
                if record.strand != previous.strand or record.dist > self.options.distance:
                    paragraph.append(sentence if previous.strand == Strand.POS else list(reversed(sentence)))
                    sentence = []
                sentence.extend(record.phrogs)
                previous = record
                bar()
            else:
                paragraph.append(sentence if previous.strand == Strand.POS else list(reversed(sentence)))
                bar()

        if self.options.collapse:
            with alive_bar(len(paragraph), title = "Collapsing  ", spinner = PHROG_SPINNER) as bar:
                for i, _ in enumerate(paragraph):
                    previous = object()
                    paragraph[i] = [previous := x for x in paragraph[i] if not (previous == self.unknown == x)]
                    bar()
                POND_LOGGER.info(f"Collapsed successfully all consecutive duplicates from {len(paragraph)} words")
        
        if self.options.number:
            with alive_bar(len(paragraph), title = "Numering    ", spinner = PHROG_SPINNER) as bar:
                for sentence in paragraph:
                    for i, phrog in enumerate(sentence):
                        if phrog == self.unknown:
                            sentence[i] = f"{phrog}{counter}"
                            counter += 1
                    bar()
                POND_LOGGER.info(f"Added numbers successfully for jokers from {len(paragraph)} words")

        if self.options.consecutive:
            with alive_bar(len(paragraph), title="consecutive ", spinner=PHROG_SPINNER) as bar:
                for i, _ in enumerate(paragraph):
                    # previous = paragraph[i][0]
                    # unkown_counter = int(previous == self.unknown)

                    # if len(paragraph[i]) == 1 and unkown_counter:
                    #     paragraph[i] = ["joker1"]
                    #     bar()
                    #     continue

                    # new_sentence = []
                    # if not unkown_counter:
                    #     new_sentence.append(previous)

                    # for phrog in paragraph[i][1:]:
                    #     if phrog == self.unknown:
                    #         unkown_counter += 1
                    #         previous = self.unknown
                    #         continue

                    #     # breaks the streak of unknown proteins
                    #     if previous == self.unknown:
                    #         new_sentence.append(f"joker{unkown_counter}")
                    #         unkown_counter = 0

                    #     new_sentence.append(phrog)
                    #     previous = phrog

                    # # adds unknown proteins at the end of the sentence
                    # if unkown_counter != 0:
                    #     new_sentence.append(f"joker{unkown_counter}")

                    # paragraph[i] = new_sentence
                    # bar()
                
                    paragraph[i] = self.collapse_consecutive(paragraph[i])
                    bar()

                POND_LOGGER.info(f"Added consecutive numbering successfully for jokers from {len(paragraph)} words")


        self.content = paragraph
        return paragraph
