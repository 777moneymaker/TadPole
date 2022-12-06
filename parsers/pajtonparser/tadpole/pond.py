#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from enum import Enum
from alive_progress import alive_bar
from alive_progress.animations.spinners import bouncing_spinner_factory

PHROG_SPINNER = bouncing_spinner_factory(("🐸", "🐸"), 8, background = ".", hide = False, overlay =True)

class Strand(Enum):
    NEG = 0
    POS = 1

    @classmethod
    def into(cls, val: str):
        if not isinstance(val, str):
            raise TypeError(f"Into strand should have a 'str' type but {type(val)} provided.")
        
        if val not in ("+", "-"):
            raise ValueError(f"Valid values for Strand.into() are ['+', '-'] but '{val}' provided.")

        return cls.POS if val == "+" else cls.NEG


@dataclass
class PondLocation:
    """Object containing gff and phrog directories paths"""
    phrog_dir: Path
    gff_dir: Path

    def __init__(self, phrog_dir: Path, gff_dir: Path):
        if not isinstance(gff_dir, Path):
            raise TypeError("Gff dir is not a Path obj")
        if not isinstance(phrog_dir, Path):
            raise TypeError("Phrog dir is not a Path obj")
        if not phrog_dir.is_dir():
            raise TypeError("Phrog dir is not dir")
        if not gff_dir.is_dir():
            raise TypeError("Gff dir is not dir")
        
        self.phrog_dir = phrog_dir
        self.gff_dir = gff_dir

@dataclass
class PondOptions:
    """Object containig metadata for phrog and gff parser"""
    distance: int
    number: bool
    collapse: bool

    def __init__(self, distance: int, number: bool, collapse: bool):
        if not isinstance(distance, (int, float, complex)):
            raise TypeError("distance is not an int")
        if distance < 0:
            raise ValueError("Distance < 0 is not valid")
        if not isinstance(collapse, bool):
            raise TypeError("collapse should be a bool obj")
        if not isinstance(number, bool):
            raise TypeError("number should be a bool obj")
        
        self.distance = distance
        self.number = number
        self.collapse = collapse

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
            raise TypeError("PondMap values should be str's only.")
        self.d[key].append(value)

    def clear(self):
        self.d.clear()

    def remove_duplicates(self):
        for k, v in self.d.items():
            self.d[k] = list(dict.fromkeys(v)) # removes duplicates while preserving order
    

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
        with alive_bar(len(files), title = "PHROGs    ",  dual_line = True, spinner = PHROG_SPINNER) as bar:
            bar.text = "--> Looping on some PHROGs"
            for file in files:
                with open(file, encoding="utf-8") as fh:
                    next(fh)
                    for line in fh:
                        prot, phrog = line.split(",")[:2]
                        self.map[prot].append(phrog)
                bar()
        self.map.remove_duplicates() 
        
    def parse(self) -> list[list[str]]:
        self._fill_map()
        Sentence = list[str]
        
        files = list(self.location.gff_dir.iterdir())
        with alive_bar(len(files), title = "GFFs      ", dual_line = True, spinner = PHROG_SPINNER) as bar:
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

                        dist = 0 if j == 0 else start - self.records[j - 1].end
                        record = PondRecord(prot, phrogs, start, end, strand, dist)
                        self.records.append(record)
                bar()
        records: list[PondRecord] = iter(self.records)
        prev: PondRecord = next(records)
        sentence: Sentence = []
        paragraph: list[Sentence] = []

        # Add first phrogs
        sentence.extend(prev.phrogs)
        counter: int = 1
        with alive_bar(len(self.records), title = "Sentences ", dual_line = True, spinner = PHROG_SPINNER) as bar:
            bar.text = "--> Ordering some sentences"
            for record in records:
                if record.strand != prev.strand or record.dist > self.options.distance:
                    paragraph.append(sentence if prev.strand == Strand.POS else list(reversed(sentence)))
                    sentence = []
                sentence.extend(record.phrogs)
                prev = record
                bar()
            else:    
                paragraph.append(sentence if prev.strand == Strand.POS else list(reversed(sentence)))
                bar()

        if self.options.collapse:
            with alive_bar(len(paragraph), title = "Collapsing", spinner = PHROG_SPINNER) as bar:
                for i, _ in enumerate(paragraph):
                    prev = object()
                    paragraph[i] = [prev := x for x in paragraph[i] if prev != x]
                    bar()
        
        if self.options.number:
            with alive_bar(len(paragraph), title = "Numering  ", spinner = PHROG_SPINNER) as bar:
                for sentence in paragraph:
                    for i, phrog in enumerate(sentence):
                        if phrog == self.unknown:
                            sentence[i] = f"{phrog}{counter}"
                            counter += 1
                    bar()
            
        self.content = paragraph
        return paragraph
        
