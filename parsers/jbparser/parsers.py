from pathlib import Path
from typing import List, Dict, Generator


# Prodigal GFF3

class Cursor:
    """
    Representation of the current position in the genome
    Used to test continuity of gene groups
    """

    def __init__(self, max_distance):
        self.max_distance = max_distance
        self.genome = None
        self.strand = None
        self.position = None

    def is_continuation(self,
                        genome: str,
                        start: int,
                        end: int,
                        strand: str):
        if genome != self.genome:
            continuation = True if self.genome is None else False
        elif strand != self.strand or start - self.position > self.max_distance:
            continuation = False
        else:
            continuation = True
        rev = True if self.strand == '-' else False
        self.genome = genome
        self.strand = strand
        self.position = end
        return continuation, rev


def sentences_from_gff(gff_path: Path,
                       max_distance: int = 1000) -> Generator[List[str], None, None]:
    """
    Parse gff to produce RAW gene-order sentences
    based on Prodigal gene identifiers.
    These sentences have to be annotated with PHROGS
    and edited to get final representation
    :param gff_path:
    :return: generator yielding sentences
             [g1_cds1, g1_cds2],
             [g1_cds3, g1_cds4, g1_cds5],
             (...),
             [(...), gX_cdsX]
    """
    with gff_path.open() as gff:
        cursor = Cursor(max_distance)
        current_sentence = []
        for line in gff:
            if line.startswith('#'):  # skip comments in the file header
                pass
            else:
                genome, _, _, start, end, _, strand, _, astring = line.rstrip('\n').split('\t')
                start, end = int(start), int(end)
                attrbutes = {k: v for k, v in [a.split('=') for a in astring.split(';') if a]}
                gene_n = attrbutes['ID'].split('_')[-1]
                gene_id = f'{genome}_{gene_n}'
                cotinuation, rev = cursor.is_continuation(genome, start, end, strand)
                if cotinuation:
                    current_sentence.append(gene_id)
                else:
                    yielded_sentence = list(reversed(current_sentence)) if rev else current_sentence
                    current_sentence = [gene_id]
                    yield yielded_sentence
        yield current_sentence


# MMsequs2

# The first column is the matching PHROG while the second column is the ID of your sequence,
# then you have alnScore seqIdentity eVal qStart qEnd qLen tStart tEnd tLen,
# respectively (Note that residues numbering starts at 0).
# BASED ON https://phrogs.lmge.uca.fr/READMORE.php
PARSER_DICT = {'prot_id': 1,
               'hmm_id': 0,
               'prot_start': 8,
               'prot_end': 9,
               'evalue': 4,
               'score': 2}

class Domain:
    """
    Object representing a single
    MMseqs2 hit -
    a domain or protein family
    aligned to an amino acid sequence
    """

    def __init__(self, line: str):
        """
        Object initialisation (creation of the class instance)
        the functions assumes that has a line from hmmsearch
        and has to be told otherwise if hmmscan was used
        """
        line = line.rstrip('\n')
        split_line = line.split('\t')
        self.prot_id = split_line[PARSER_DICT['prot_id']]
        self.hmm_id = split_line[PARSER_DICT['hmm_id']]
        self.prot_start, self.prot_end = int(split_line[PARSER_DICT['prot_start']]), int(
            split_line[PARSER_DICT['prot_end']])
        self.score = float(split_line[PARSER_DICT['score']])
        self.evalue = float(split_line[PARSER_DICT['evalue']])
        self.line = line

    def __repr__(self):
        """
        How an instance should look like in print etc.
        """
        return f'{self.hmm_id} [{self.prot_start} - {self.prot_end}] ({self.score})'

    def gff(self) -> str:
        """
        Represent a domain as a GFF line
        :return: gff-formatted line (no newline (\n) at the end)
        """
        attributes = {'HMM': self.hmm_id}
        attribute_string = ';'.join(f'{k}={v}' for k, v in attributes.items())
        return '\t'.join([self.prot_id,
                          'MMseqs2',
                          'alignment',
                          str(self.prot_start),
                          str(self.prot_end),
                          str(self.score),
                          '.', '.',
                          attribute_string])

    def seq_length(self) -> int:
        """
        Calculate length of domain in AA
        :return: length of domain in AA
        """
        return self.prot_end - self.prot_start + 1

    def overlaps(self, other: 'Domain') -> bool:
        """
        Check if any of the two analysed domains overlap
        on more than 50% of its length with the other
        :param other:
        :return: are these domains overlapping?
        """
        if self.prot_end >= other.prot_start and self.prot_start <= other.prot_end:
            overlap_start = max(self.prot_start, other.prot_start)
            overlap_end = min(self.prot_end, other.prot_end)
            overlap = overlap_end - overlap_start + 1
            if overlap < 1:
                raise NotImplementedError()
            if any([overlap > e.seq_length() * 0.5 for e in (self, other)]):
                return True
        elif other.prot_end >= self.prot_start and other.prot_start <= self.prot_end:
            return other.overlaps(self)
        return False


def select_top(domain_list: List[Domain]) -> Domain:
    """
    Given a list of domains choose top-scoring one
    :param domain_list: list of two or more domains
    :return: Top scoring domain from the cluster
    """
    ranking = sorted(domain_list, key=lambda dom: dom.score, reverse=True)
    return ranking[0]


def parse_mmtsv(file_path: Path,
                filtering: str = 'cull',
                max_evalue: float = 1e-3) -> Dict[str, List[str]]:
    """
    Read MMseqs2 file ane return a dictionary of
    domains annotated on every protein
    :param file_path: path to a MMseqs2 tsv file
    :param filtering: what alignments should be reported:
                      None - all domains
                      'best' - only one top scoring domain for each protein
                      'cull' - all domains unless they overlap with higher scoring one [default]
    :param max_evalue: max mmseq
    :return: {prot1: []}
    """
    with file_path.open() as inpt:
        domains_in_proteins = {}
        discarded, kept = 0, 0
        for line in inpt:
            if line.strip() and not line.startswith('#'):
                domain = Domain(line)
                if domain.evalue <= max_evalue:
                    if domain.prot_id not in domains_in_proteins:
                        domains_in_proteins[domain.prot_id] = []
                    domains_in_proteins[domain.prot_id].append(domain)
                    kept += 1
                else:
                    discarded += 1

    if filtering is None:
        # if len(filtered_domains) > 1:
        #     print(culled_domains_in_proteins[protein_id])
        return {protein: [d.hmm_id for d in domains] for protein, domains in domains_in_proteins.items()}
    elif filtering == 'best':
        return {protein: [select_top(domains).hmm_id] for protein, domains in domains_in_proteins.items()}

    elif filtering == 'cull':
        culled_domains_in_proteins = {}
        while domains_in_proteins:
            protein_id, domains = domains_in_proteins.popitem()
            domains.sort(key=lambda dom: dom.prot_start)
            filtered_domains = []
            while domains:
                new_domain = domains.pop(0)
                overlapping_domains = []
                non_overlapping_domains = []
                while domains:
                    old_domain = domains.pop(0)
                    if new_domain.overlaps(old_domain):
                        overlapping_domains.append(old_domain)
                    else:
                        non_overlapping_domains.append(old_domain)
                if overlapping_domains:
                    main_domain = select_top(overlapping_domains + [new_domain])
                    domains = non_overlapping_domains + [main_domain]
                    domains.sort(key=lambda dom: dom.prot_start)
                else:
                    filtered_domains.append(new_domain)
                    domains = non_overlapping_domains
            culled_domains_in_proteins[protein_id] = filtered_domains
        return {protein: [d.hmm_id for d in domains] for protein, domains in culled_domains_in_proteins.items()}
    else:
        raise NotImplementedError(f'Unknown filtering method: {filtering}')
