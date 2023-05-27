from typing import Dict, List
from pathlib import Path
from itertools import count

from parsers import parse_mmtsv, sentences_from_gff
import pyfastx as pfx
import argparse
import json
import pickle
from utils import checkpoint, Parallel, BatchParallel


def annotate_phrogs(raw_sentence: List[str],
                    annotation_dict: Dict[str, List[str]]) -> List[str]:
    """
    Add PHROG annotations to sentence
    :param raw_sentence: [g1_cds1, g1_cds2]
    :param annotation_dict: {g1_cds1: [phrog1], g1_cds3: [phrog13, phrog337]}
    :return: [phrog91, gene_91_01, phrog34, gene_11_37 (...)]
    """
    annotated_sentebnce = []
    for gene_id in raw_sentence:
        phrase = annotation_dict[gene_id] if gene_id in annotation_dict else [gene_id]
        annotated_sentebnce.extend(phrase)
    return annotated_sentebnce


def annotate_sequeces(raw_sentence: List[str],
                      annotation_dict: Dict[str, List[str]],
                      phrog_sequences: Dict[str, str],
                      all_sequences: Dict[str, str]) -> List[str]:
    """
    Translate gene identifiers into sequence "words"
    - consensus sequences for annotated PRHROGs
    - predicted product sequences for unannotated proteins
    :param raw_sentence: [g1_cds1, g1_cds2]
    :param annotation_dict: {g1_cds1: [phrog1], g1_cds3: [phrog13, phrog337]}
    :param phrog_sequences: dict-like object with consensus sequence for every PHROG
    :param all_sequences: dict-like object with consensus sequence for every gene product
    :return: [GVQFDEN..., MVVCDED..., (...)]
    """
    annotated_sentebnce = []
    for gene_id in raw_sentence:
        if gene_id in annotation_dict:
            phrase = [phrog_sequences[phrog].seq for phrog in annotation_dict[gene_id]]
        else:
            phrase = [all_sequences[gene_id].seq]
        annotated_sentebnce.extend(phrase)
    return annotated_sentebnce


def handle_unannotated(annotated_sentence: List[str],
                       method: str = '1xs',
                       filter_nonphrog: bool = False):
    """
    Replace gene identifiers with no PHROG annotations with
    appropriate auxiliary tokens.
    :param annotated_sentence: gene sentence after PHROG annotation
    :param method: unannotated genes should be represented as:
                  '1xs' - merge strings of unannotated genes and represent them as a number [default]
                  'x1' - number unannotated genes within the string and represent each separately
                  'x' - transforms each unannotated gent to simple "x"
    :return: [phrog91, 1xs, phrog34, 4xs, (...)]
    """
    phrog_found = False
    transformed_sentence = []

    # method '1xs'
    if method == '1xs':
        u_count, u_word = count(1), []
        for word in annotated_sentence:
            if not word.startswith('phrog_'):
                u_word = [f'{next(u_count)}xs']
            else:
                transformed_sentence.extend(u_word + [word])
                u_count, u_word = count(1), []
                phrog_found = True
        if u_word:
            transformed_sentence.extend(u_word)

    # method 'x1'
    elif method == 'x1':
        u_count = count(1)
        for word in annotated_sentence:
            if not word.startswith('phrog_'):
                word = f'x{next(u_count)}'
            else:
                phrog_found = True
            transformed_sentence.append(word)

    # method 'x'
    elif method == 'x':
        for word in annotated_sentence:
            if not word.startswith('phrog_'):
                word = 'x'
            else:
                phrog_found = True
            transformed_sentence.append(word)

    # faulty method
    else:
        raise NotImplementedError(f'Unknown handling method: {method}')
    assert not any([isinstance(e, list) for e in transformed_sentence])
    if filter_nonphrog and not phrog_found:
        transformed_sentence = None

    return transformed_sentence


def word2vec_tokenizer(gff: Path,
                       mmtsv: Path,
                       filtering: str = 'cull',
                       max_evalue: float = 1e-3,
                       unannotated_as: str = '1xs',
                       filter_nonphrog: bool = True) -> List[List[str]]:
    """
    Generate the corpus for training of Word2Vec embeddings
    based on prodigal GFF3 file and MMseqs2 PHROG annotations
    :param gff: GFF3 file from prodigal annotation
    :param mmtsv: tabular output of the MMseqs2 PHROG annotation
    :param filtering: what MMseqs2 alignments should be reported:
                      None - all domains
                      'best' - only one top scoring domain for each protein
                      'cull' - all domains unless they overlap with higher scoring one [default]
    :param max_evalue: maximum MMseqs2 evalue of a domain to consider
    :param unannotated_as: unannotated genes should be represented as:
                          '1xs' - merge strings of unannotated genes and represent them as a number [default]
                          'x1' - number unannotated genes within the string and represent each separately
                          'x' - transforms each unannotated gene to simple "x"
                          'gid' - do not transform original gene identifiers
    :return: [[phrog91, 1xs, phrog34, 4xs, (...)], (...)]
    """
    # print(f'gff: {gff}; type={type(gff)}')
    # print(f'mmtsv: {mmtsv}')
    # print(f'filtering: {filtering}')
    # print(f'max_evalue: {max_evalue}')
    # print(f'unannotated_as: {unannotated_as}')
    # print(f'filter_nonphrog: {filter_nonphrog}')
    # TODO: progress bars
    raw_sentences = sentences_from_gff(gff)
    annotation = parse_mmtsv(mmtsv,
                             filtering=filtering,
                             max_evalue=max_evalue)
    # corpus = [annotate_phrogs(s, annotation) for s in raw_sentences]  # TODO [remove after test]
    corpus = [r for r in BatchParallel(annotate_phrogs,
                                       list(raw_sentences),
                                       kwargs={'annotation_dict': annotation}, bar=True).result]

    if not unannotated_as == 'gid':

        # corpus = [handle_unannotated(s, method=unannotated_as, filter_nonphrog=True) for s in corpus]  # TODO [remove after test]
        corpus = [r for r in BatchParallel(handle_unannotated,
                                           list(corpus),
                                           kwargs={'method': unannotated_as,
                                                   'filter_nonphrog': filter_nonphrog}, bar=True).result]

    return [s for s in corpus if s is not None]


def fasttext_seq_tokenizer(gff: Path,
                           mmtsv: Path,
                           phrogs_faa: Path,
                           prodigal_faa: Path,
                           filtering: str = 'cull',
                           max_evalue: float = 1e-3) -> List[List[str]]:
    """
    Generate the corpus for training of FastText embeddings
    based on prodigal GFF3 file and MMseqs2 PHROG annotations
    as well as consensus sequences of PHROG MSAs
    :param gff: GFF3 file from prodigal annotation
    :param mmtsv: tabular output of the MMseqs2 PHROG annotation
    :param phrogs_faa: fasta file with PHROG consensus sequences
    :param prodigal_faa: fasta file with other files
    :param filtering: what MMseqs2 alignments should be reported:
                      None - all domains
                      'best' - only one top scoring domain for each protein
                      'cull' - all domains unless they overlap with higher scoring one [default]
    :param max_evalue: maximum MMseqs2 evalue of a domain to consider
    :return: [[GVQFDEN..., MVVCDED..., (...)], (...)]
    """
    raw_sentences = sentences_from_gff(gff)
    annotation = parse_mmtsv(mmtsv,
                             filtering=filtering,
                             max_evalue=max_evalue)
    phrog_consensus = pfx.Fasta(phrogs_faa.as_posix())
    all_proteins = pfx.Fasta(prodigal_faa.as_posix())
    sequence_sentences = [annotate_sequeces(s, annotation,
                                            phrog_consensus,
                                            all_proteins)
                          for s in raw_sentences]
    translator_dict = {seq: seq_id for seq_id, seq in phrog_consensus}
    return sequence_sentences, translator_dict


if __name__ == '__main__':
    # g = Path('toy.prodigal.gff')
    # m = Path('toy.mmseqs2.tsv')
    # pf = Path('MSA_Phrogs_M50_CONSENSUS.fasta')
    # af = Path('toy.proteins.faa')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-g", "--gff", help="Path to gff file.", type=Path)
    parser.add_argument("-m", "--mmtsv", help="Path to mmseqs2 protein-phrog annotation file.", type=Path)
    parser.add_argument("-f", "--filtering", choices=['None', 'best', 'cull'], default='cull', type=str,
                        help="what MMseqs2 alignments should be reported:\nNone - all domains\n'best' - only one top scoring domain for each protein\n'cull' - all domains unless they overlap with higher scoring one [default]")
    parser.add_argument("-e", "--max_evalue", type=float, default=1e-3, help="Maximum MMseqs2 evalue of a domain to consider")
    parser.add_argument("-o", "--output", help="prefix/prefix-path for output files", type=str)

    subparsers = parser.add_subparsers(title='Available models', description='For which model the corpus will be created?', dest='subparser_model',
                                       help="Call this script with model name and -h to get help, e.g. tokenizers.py word2vec -h")

    w2v_parser = subparsers.add_parser('word2vec', help='Generate the corpus for word2vec', formatter_class=argparse.RawTextHelpFormatter)
    w2v_parser.add_argument("-j", "--unannotated_as", choices=['1xs', 'x1', 'x', 'gid'], default='1xs', type=str,
                        help="unannotated genes (jokers) should be represented as:\n'1xs' - merge strings of unannotated genes and represent them as a number [default]\n'x1' - number unannotated genes within the string and represent each separately\n'x' - transforms each unannotated gene to simple 'x'\n'gid' - do not transform original gene identifiers")
    w2v_parser.add_argument("-n", "--filter_nonphrog", help="idk", default=True, type=bool)
    
    ft_parser = subparsers.add_parser('fasttext', help='Generate the corpus for FastText', formatter_class=argparse.RawTextHelpFormatter)
    ft_parser.add_argument("-c", "--phrogs_faa", help="Fasta file with PHROG consensus sequences", type=Path)
    ft_parser.add_argument("-a", "--prodigal_faa", help="Fasta file with other proteins", type=Path)

    args = parser.parse_args()

    pickle_path, text_path = f"{args.output}.pickle", f"{args.output}.txt"
    
    match args.subparser_model:
        case 'word2vec':
            # corpus = word2vec_tokenizer(Path(args.gff_file), Path(args.mmseq_file), filtering=args.filtering, unannotated_as=args.jokers)
            corpus = word2vec_tokenizer(**{k: v for k, v in vars(args).items() if k in word2vec_tokenizer.__code__.co_varnames})

            try:
                with open(pickle_path, "wb") as fh1, open(text_path, "w", encoding="utf-8") as fh2:
                    pickle.dump(corpus, fh1)
                    fh2.write(json.dumps(corpus))
            except TypeError as e:
                print(f"JSON serialization offed with message {e.message}")
                raise
        case 'fasttext':
            corpus, translation = fasttext_seq_tokenizer(**{k: v for k, v in vars(args).items() if k in fasttext_seq_tokenizer.__code__.co_varnames})
            trans_path = f"{args.output}_translation.json"
            try:
                with open(pickle_path, "wb") as fh1, open(text_path, "w", encoding="utf-8") as fh2, open(trans_path, 'w') as fh3:
                    pickle.dump(corpus, fh1)
                    fh2.write(json.dumps(corpus))
                    json.dump(translation, fh3)
            except TypeError as e:
                print(f"JSON serialization offed with message {e.message}")
                raise
        case _:
            parser.print_help()
    # test_run = word2vec_tokenizer(g, m, filtering=None, unannotated_as='1xs')
    # print(test_run)
    # test_run = word2vec_tokenizer(g, m, filtering='cull', unannotated_as='x1')
    # print(test_run)
    # test_run = word2vec_tokenizer(g, m, filtering='best', unannotated_as='gid')
    # print(test_run)
    # test_run = fasttext_seq_tokenizer(g, m,
    #                                   phrogs_faa=pf,
    #                                   prodigal_faa=af)
    # print(test_run)
