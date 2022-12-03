#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from enum import Enum

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
    

class PondParser:
    def __init__(self, location: PondLocation, options: PondOptions, unknown: str = "joker"):
        self.location = location
        self.options = options
        self.map = PondMap()
        self.records = []
        self.unknown = unknown
        self.content = None

    def _fill_map(self):
        for file in self.location.phrog_dir.iterdir():
            with open(file) as fh:
                next(fh)
                for line in fh:
                    prot, phrog = line.split(",")[:2]
                    self.map[prot].append(phrog)    
        
    def parse(self) -> list[list[str]]:
        self._fill_map()
        Sentence = list[str]

        for i, file in enumerate(self.location.gff_dir.iterdir()):
            with open(file) as fh:
                for line in fh:
                    if line.startswith("#") or line.strip() == "":
                        continue
                    data = line.split("\t")
                    start, end, strand = int(data[3]), int(data[4]), Strand.into(data[6])
                    prot = data[8].split(";", maxsplit=1)[0].lstrip("ID=")
                    phrogs = self.map[prot]
                    if not phrogs:
                        phrogs = ["joker"]

                    record = PondRecord(prot, phrogs, start, end, strand)
                    self.records.append(record)
            print(f"\rParsed {i+1} files...")
        records: list[PondRecord] = iter(self.records)
        prev: PondRecord = next(records)
        sentence: Sentence = []
        paragraph: list[Sentence] = []
        
        prev.dist = 0
        # Add first phrogs
        sentence.extend(prev.phrogs)
        for record in records:
            record.dist = record.start - prev.end
            if record.strand != prev.strand or record.dist > self.options.distance:
                paragraph.append(sentence if prev.strand == Strand.POS else list(reversed(sentence)))
                sentence = []
            sentence.extend(record.phrogs)
            prev = record
        else:
            paragraph.append(sentence if prev.strand == Strand.POS else list(reversed(sentence)))

        if self.options.collapse:
            for i, _ in enumerate(paragraph):
                prev = object()
                # Remove consecutive duplicated unknown jokers
                paragraph[i] = [prev := x for x in paragraph[i] if prev != x]
        if self.options.number:
            unknown_counter: int = 1
            for i, _ in enumerate(paragraph):
                for j, _ in enumerate(paragraph[i]):
                    if paragraph[i][j] == self.unknown:
                        paragraph[i][j] = f"{self.unknown}{unknown_counter}"
                        unknown_counter += 1
            
        self.content = paragraph
        return paragraph
        
