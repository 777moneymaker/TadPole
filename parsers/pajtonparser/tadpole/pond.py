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
from statistics import quantiles

from .logger import POND_LOGGER

PHROG_SPINNER = bouncing_spinner_factory(("🐸", "🐸"), 8, background = ".", hide = False, overlay =True)

class PondThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        # Thread.__init__(self, group, target, name, args, kwargs)
        # self._return = None
        raise NotImplementedError

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class Strand(Enum):
    NEG = "-"
    POS = "+"

    @classmethod
    def into(cls, val: str):
        if not isinstance(val, str):
            POND_LOGGER.error(f"Wrong strand type '{type(val)}' provided")
            raise TypeError(f"Into strand should have a 'str' type but {type(val)} provided.")
        
        if val not in ("+", "-"):
            POND_LOGGER.error(f"Wrong strand character '{val}' provided")
            raise ValueError(f"Valid values for Strand.into() are ['+', '-'] but '{val}' provided.")

        return cls.POS if val == "+" else cls.NEG


class Sequential(Enum):
    NONE = "none"
    ENUMERATE = "enumerate",
    COLLAPSE = "collapse",
    CONSECUTIVE = "consecutive",
    ENCODE = "encode"


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
class PondParams:
    props: dict[dict[str, float]] # 'Protein_ID': { 'chemical_property': 'value' }
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
    def __init__(self, location: PondLocation, params: PondParams, unknown: str = "joker"):
        self.location = location
        self.params = params
        self.map = PondMap()
        self.records = []
        self.unknown = unknown
        self.content = None
        self.props_ranges = dict()
        self.resulting_ranges = dict()

    def _compute_props_ranges(self):
        if not self.params.props:
            return
        
        POND_LOGGER.debug("Started producing ranges for protein params")
        # Compute range of possible values for all protein props
        # Not all props are true (we don't need to compute all of them)

        # First get the lists of all values for each of the protein props
        props_values = {
            "molecular_weight": [], 
            "aromaticity": [], 
            "instability_index": [], 
            "gravy": [], 
            "isoelectric_point": []
        }
        with alive_bar(len(props_values) * 3, title = "Props...", dual_line = True, spinner = PHROG_SPINNER) as bar:
            for prop in props_values.keys():
                if not self.params.config[prop]: # If user set property to false
                    continue
                bar.text = f"--> Finding possible values for property: {prop}"
                for inner_dict in self.params.props.values(): # Produces inner dict for each protein in a form of {prop: value}
                    props_values[prop].append(inner_dict[prop]) # add the value
                bar()
            # Remove unused props
            props_values = {key: value for key, value in props_values.items() if value} # Leave only if value (if not [])
            # Now produce ranges
            bar.text = "--> Computing ranges"
            for prop, values in props_values.items():
                # Not the props_values is a dict in a form of
                # {"prop": (lowest_possible_value, biggest_possible_value)}
                
                props_values[prop] = PondParser.range_from_list(values)
                print(f"Computed range for property '{prop}': '{props_values[prop]}'")
                bar()
            self.props_ranges = props_values

            # A'la PHRED+33? This is range that we map all proteins props to
            # Produces 
            resulting_ranges = {}
            bar.text = f"--> Calculating resulting non-overlapping ranges'"
            for prop, new_range in zip(self.props_ranges.keys(), PondParser.non_overlapping_intervals(num_intervals=len(self.props_ranges))):
                print(f"Computed new range for property '{prop}': '{new_range}'")
                resulting_ranges[prop] = new_range
                bar()
            self.resulting_ranges = resulting_ranges
            
            POND_LOGGER.debug("Successfully produced all ranges")


    def _fill_map(self):
        if self.params.props:
            self._compute_props_ranges()

        files = list(self.location.phrog_dir.iterdir())
        with alive_bar(len(files), title = "PHROGs...",  dual_line = True, spinner = PHROG_SPINNER) as bar:
            bar.text = "--> Looping on some PHROGs"
            for file in files:
                with open(file, encoding="utf-8") as fh:
                    next(fh)
                    for line in fh:
                        prot, phrog = line.split(",")[:2]
                        if self.params.config["sequential"] == "encode" and self.params.props:
                            bar.text = "--> Encoding proteins"
                            encoded_chars = []                        
                            # Get props for this protein
                            props = self.params.props.get(prot) # Like {molecular_weight: 6043}... etc.
                            # if protein not found
                            # Skip this protein in our word, add to logs
                            if not props:
                                POND_LOGGER.error(f"Protein '{prot}' not found")
                            else : 
                                for prop, prop_value in props.items():
                                    if not self.params.config[prop]: # If property is false
                                        continue
                                    from_range = self.props_ranges[prop]
                                    encoded_char = PondParser.remap_value(from_range, self.resulting_ranges[prop], prop_value)
                                    encoded_chars.extend(encoded_char)
                                phrog = "".join(encoded_chars) + "{0:05d}".format(int(phrog.replace("phrog_", ""))) # Leave only phrog number
                        self.map[prot].append(phrog)
                bar()
            POND_LOGGER.info("Processed all available phrog files")

    @staticmethod
    def non_overlapping_intervals(num_intervals, interval_length=10):
        intervals = []
        start = 34
        for i in range(num_intervals):
            end = start + interval_length - 1
            intervals.append((start, end))
            start = end + 1 
        return intervals

    @staticmethod
    def remap_value(from_range, to_range, value):
        from_min, from_max = from_range
        to_min, to_max = to_range
        if value > to_max:
            return "!" # Intervals are created so they begin with 34 ascii, not 33, so we can use '!' for outliers
        # Calculate the proportional position of the value in the input range
        position = (value - from_min) / (from_max - from_min)
        # Calculate the value in the output range by linearly scaling the
        # position to the range of the output range
        mapped_value = position * (to_max - to_min) + to_min
        return chr(round(mapped_value))

    @staticmethod
    def range_from_list(numbers):
        """Computes the range of values from a list excluding extreme outliers"""
        q1, _, q3 = quantiles(numbers)
        iqr = q3 - q1
        upperbound = q3 + (iqr * 3)
        filtered_numbers = [
            number for number in numbers if number <= upperbound
        ]
        return min(filtered_numbers), max(filtered_numbers)

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
        with alive_bar(len(files), title = "FAAs...", dual_line = True, spinner = PHROG_SPINNER) as bar:
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


    def parse(self) -> list[list[str]]:
        self._fill_map()
        Sentence = list[str]

        files = list(self.location.gff_dir.iterdir())
        with alive_bar(len(files), title = "GFFs...", dual_line = True, spinner = PHROG_SPINNER) as bar:
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
                            phrogs = [self.unknown]
                            if self.params.props and self.params.config["sequential"] == "encode": # This is where the fun begins
                                bar.text = "--> Encoding jokers"
                                encoded_chars = []
                                # Get props for this protein
                                props = self.params.props.get(prot) # Like {molecular_weight: 6043}... etc.
                                # if protein not found
                                if not props: # Skip this protein in our word, add to logs
                                    POND_LOGGER.error(f"Protein '{prot}' not found")
                                    phrogs = ["!!!!!#####"]
                                else:   
                                    for prop, prop_value in props.items():
                                        if not self.params.config[prop]: # If property is false
                                            continue
                                        from_range = self.props_ranges[prop]
                                        encoded_char = PondParser.remap_value(from_range, self.resulting_ranges[prop], prop_value)
                                        encoded_chars.extend(encoded_char)
                                    phrogs = ["".join(encoded_chars) + "#####"] # 5 times '#' to equalize to phrogs

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
        with alive_bar(len(self.records), title = "Sentences...", dual_line = True, spinner = PHROG_SPINNER) as bar:
            bar.text = "--> Ordering some sentences"
            for record in records:
                if record.strand != previous.strand or record.dist > self.params.config["distance"]:
                    paragraph.append(sentence if previous.strand == Strand.POS else list(reversed(sentence)))
                    sentence = []
                sentence.extend(record.phrogs)
                previous = record
                bar()
            else:
                paragraph.append(sentence if previous.strand == Strand.POS else list(reversed(sentence)))
                bar()

        if self.params.config["sequential"] == "collapse":
            with alive_bar(len(paragraph), title = "Collapsing...", spinner = PHROG_SPINNER) as bar:
                for i, _ in enumerate(paragraph):
                    previous = object()
                    paragraph[i] = [previous := x for x in paragraph[i] if not (previous == self.unknown == x)]
                    bar()
                POND_LOGGER.info(f"Collapsed successfully all consecutive duplicates from {len(paragraph)} words")
        
        if self.params.config["sequential"] == "enumerate":
            with alive_bar(len(paragraph), title = "Enumerating...", spinner = PHROG_SPINNER) as bar:
                for sentence in paragraph:
                    for i, phrog in enumerate(sentence):
                        if phrog == self.unknown:
                            sentence[i] = f"{phrog}{counter}"
                            counter += 1
                    bar()
                POND_LOGGER.info(f"Added enumerated successfully for jokers from {len(paragraph)} words")

        if self.params.config["sequential"] == "consecutive":
            with alive_bar(len(paragraph), title="Consecutive...", spinner=PHROG_SPINNER) as bar:
                for i, _ in enumerate(paragraph):
                    paragraph[i] = self.collapse_consecutive(paragraph[i])
                    bar()

                POND_LOGGER.info(f"Added consecutive numbering successfully for jokers from {len(paragraph)} words")


        self.content = paragraph
        return paragraph
