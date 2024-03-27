""" 
This script is used for inference.

Variable conventions:
-- we can infer publications with whatever id they have as long as they have the required metadata
-- however the id will be called "doi" in the code

python3 inference.py 
--in_path="/storage2/sotkot/bibliometrics_toolkit/data/P1_2000_2010_split/oa_pubs_dir_5" 
--out_path="/storage2/sotkot/bibliometrics_toolkit/data/P1_2000_2010_split/oa_pubs_dir_5_output" 
--only_l4=True 
--file_type="parquet"
"""


import os
import json
import argparse
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import distutils.util

from pprint import pprint
from tqdm import tqdm
from collections import Counter
from venue_parser import VenueParser
from multigraph import MultiGraph
from utils import TextProcessor
from itertools import groupby


def parse_args():
    ##############################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default='/input_files', help="The directory where the chunks of publications along with metadata exist", required=False)
    parser.add_argument("--out_path", type=str, default='/output_files', help="The directory where the output files will be written", required=False)
    parser.add_argument("--log_path", type=str,default='fos_inference.log',  help="The path for the log file.", required=False)
    parser.add_argument("--emphasize", type=str,default='citations',  help="If you want to emphasize in published venue or the cit/refs", required=False)
    parser.add_argument("--only_l4", type=lambda x:bool(distutils.util.strtobool(x)), default=False,  help="If you want to only infer L4", required=False)
    parser.add_argument("--extra_metadata", type=lambda x:bool(distutils.util.strtobool(x)), default=False,  help="If you want to save the metadata from the input file", required=False)
    parser.add_argument("--file_type", type=str, default='parquet',  help="the file type we will load", required=True)
    parser.add_argument("--batch_size", type=int, default=10000,  help="The batch size", required=False)
    args = parser.parse_args()
    return args
    ##############################################


def load_excel(my_path):
    """
    Load the excel file with the Level 4 names.

    Parameters:
    my_path (str): The path to the excel file.

    Returns:
    list: A list of dictionaries representing the Level 4 names from the excel file.
    """
    level_4 = pd.read_excel(my_path).fillna('N/A').to_dict('records')
    return level_4


# ----------------------------------------------------------#
# initializations
venue_parser = VenueParser(abbreviation_dict='data/venues_maps.p')
multigraph = MultiGraph('data/scinobo_inference_graph.p')
text_processor = TextProcessor()

# load mappings
with open('data/L2_to_L1.json', 'r') as fin:
    L2_to_L1 = json.load(fin)
with open('data/L3_to_L2.json', 'r') as fin:
    L3_to_L2 = json.load(fin)
with open('data/L4_to_L3.json', 'r') as fin:
    L4_to_L3 = json.load(fin)

level_4_names = load_excel('data/level_4_names.xlsx') # this is always in the repository -- no need to pass a path
level_4_ids_2_names = {level_4['Level 4']: level_4['Level 4 Name'] for level_4 in level_4_names}
# ----------------------------------------------------------#


def infer_relationship(multigraph, top_L3, top_L4, overwrite, relationship):
    """
    Infer the relationship between entities in the multigraph.
    Args:
        multigraph (Multigraph): The multigraph containing the entities.
        top_L3 (int): The maximum number of links to infer for entities in L3 layer.
        top_L4 (int): The maximum number of links to infer for entities in L4 layer.
        overwrite (bool): Flag indicating whether to overwrite existing links.
        relationship (str): The relationship to infer.

    Returns:
        None
    """
    multigraph.infer_layer(entity_chain=["doi", "venue", "L4"], relationship_chain=[relationship, "in_L4"],
                            overwrite=overwrite, max_links=top_L4)
    multigraph.infer_layer(entity_chain=["doi", "venue", "L3"], relationship_chain=[relationship, "in_L3"],
                            overwrite=overwrite, max_links=top_L3)


def emit_candidate_ngrams(processed_text, topk):
    """
    Extracts candidate n-grams from the processed text and retrieves similar nodes from the inference graph.

    Args:
        processed_text (str): The processed text from which to extract n-grams.
        topk (int): The maximum number of similar nodes to retrieve.

    Returns:
        tuple: A tuple containing two elements:
            - A list of unigrams extracted from the processed text.
            - A set of similar nodes retrieved from the inference graph.
    """
    # extract ngrams and similar ngrams from the inference graph--once here
    trigrams = text_processor.get_ngrams(processed_text, k=3)
    bigrams = text_processor.get_ngrams(processed_text, k=2)
    unigrams = text_processor.get_ngrams(processed_text, k=1)
    if trigrams == [] and bigrams == [] and unigrams == []:
        return []
    # the bigrams and trigrams that are identical will also be in the hits
    # concat bigrams and trigrams
    my_ngrams = list(set(bigrams + trigrams))
    my_ngrams_hits = text_processor.retrieve_similar_nodes(my_ngrams, topk)
    if my_ngrams_hits:
        my_ngrams_hits = set([b for bi in my_ngrams_hits if bi for tup in bi for b in tup])
    else:
        my_ngrams_hits = set()
    return unigrams, my_ngrams_hits


def add_to_predictions(tups, title, abstract):
    """
    Adds inferred L5 and L6 categories to the respective tuples with L4 categories.

    Args:
        tups (list): A list of tuples representing the predictions with L4 categories.
        title (str): The title of the document.
        abstract (str): The abstract of the document.

    Returns:
        list: A list of tuples representing the predictions with L4, L5, and L6 categories.
    """
    # infer L5/L6 for the inferred L4s
    processed_title = text_processor.preprocess_text(title)
    processed_abstract = text_processor.preprocess_text(abstract)
    my_text = processed_title + processed_abstract
    res = emit_candidate_ngrams(my_text, 5)
    if not res:
        return []
    my_unigrams, bi_tri_grams = res
    all_hits = set(my_unigrams) | bi_tri_grams
    all_l5s = [node[0] for node in multigraph.nodes(data='L5') if node[1] and any([t[3][0] in node[0] for t in tups if len(t) > 3])]
    l5s = filter_level_5(
        all_l5s,
        multigraph,
        all_hits,
        only_text=False
    )
    ######################################################
    # add to the respective tups with L4s the L5s and L6s
    final_tups = []
    for tup in tups:
        if len(tup) > 3:
            if l5s:
                l5 = [l5_item for l5_item in l5s if tup[3][0] in l5_item[0]]
                if not l5:
                    tup = tup + (None, None)
                    final_tups.append(tup)
                    continue
                # add to tup
                for l5_item in l5:
                    final_tups.extend([(tup[0], tup[1], tup[2], tup[3], (l5_item[0], l5_item[1]), '/'.join(list(l5_item[2])))])
            else:
                tup = tup + (None, None)
                final_tups.append(tup)
        else:
            tup = tup + (None, None, None)
            final_tups.append(tup)
    ######################################################
    return final_tups


def infer_l5_l6(tups, title, abstract):
    """
    Infers the L5/L6 categories based on the given inputs.

    Args:
        tups (list): A list of tuples containing L4 categories and their corresponding scores.
        title (str): The title of the document.
        abstract (str): The abstract of the document.

    Returns:
        list: A list of tuples containing the inferred L5/L6 categories and their corresponding scores.
    """
    # get the inferred L4s and infer their L5/L6
    # check if the title and abstract are available
    if title != '' and abstract != '':
        preds = add_to_predictions(tups, title, abstract)
    elif title != '' and abstract == '':
        preds = add_to_predictions(tups, title, '')
    elif title == '' and abstract != '':
        preds = add_to_predictions(tups, '', abstract)
    else:
        # both are empty
        preds = add_to_predictions(tups, '', '')
    return preds


def infer(**kwargs):
    """
    Infers relationships between publications based on the provided input.

    Args:
        **kwargs: Keyword arguments that control the behavior of the inference process.
            - top_L1 (int): The number of top-level 1 relationships to consider. Default is 1.
            - top_L2 (int): The number of top-level 2 relationships to consider. Default is 2.
            - top_L3 (int): The number of top-level 3 relationships to consider. Default is 3.
            - top_L4 (int): The number of top-level 4 relationships to consider. Default is 4.
            - emphasize (str): The type of relationships to emphasize. Default is 'citations'.
            - only_l4 (bool): Whether to only infer level 4 relationships. Default is False.
            - payload (dict): A dictionary containing the input data for inference. It should have the following keys:
                - dois (list): A list of publication IDs.
                - published_venues (list): A list of published venues for the publications.
                - cit_ref_venues (list): A list of citation/reference venues for the publications.
                - titles (dict): A dictionary mapping publication IDs to their titles.
                - abstracts (dict): A dictionary mapping publication IDs to their abstracts.

    Returns:
        dict: A dictionary containing the inferred relationships for each publication. The keys are the publication IDs
        and the values are lists of dictionaries, where each dictionary represents a relationship and has the following keys:
            - L1 (str): The top-level 1 relationship.
            - L2 (str): The top-level 2 relationship.
            - L3 (str): The top-level 3 relationship.
            - L4 (str): The top-level 4 relationship.
            - L5 (str): The top-level 5 relationship.
            - L6 (str): The top-level 6 relationship.
            - score_for_L3 (float): The score for the top-level 3 relationship.
            - score_for_L4 (float): The score for the top-level 4 relationship.
            - score_for_L5 (float): The score for the top-level 5 relationship.
    """
    # defaults
    top_L1 = kwargs.get('top_L1', 1)
    top_L2 = kwargs.get('top_L2', 2)
    top_L3 = kwargs.get('top_L3', 3)
    top_L4 = kwargs.get('top_L4', 4)
    # other variables
    kwargs.get('emphasize', 'citations')
    only_l4 = kwargs.get('only_l4', False)
    # return_triplets = kwargs.get('return_triplets', True)
    # ids of publications
    ids = kwargs['payload']['dois']
    published_venues = kwargs['payload']['published_venues']
    cit_ref_venues = kwargs['payload']['cit_ref_venues']
    titles = kwargs['payload']['titles']
    abstracts = kwargs['payload']['abstracts']
    # add the publications to the graph that we are going to infer
    add(multigraph, published_venues, cit_ref_venues)
    # inferring relationships
    _ = [
        infer_relationship(ids, multigraph, top_L1, top_L2, top_L3, top_L4, overwrite=True, relationship='cites'),
        infer_relationship(ids, multigraph, top_L1, top_L2, top_L3, top_L4, overwrite=False, relationship='published')
    ]
    out = {}
    all_l3s = [(relationship[0], relationship[1], relationship[2]) for relationship in multigraph.edges(data='in_L3', nbunch=ids) if relationship[2]]
    all_l4s = [(relationship[0], relationship[1], relationship[2]) for relationship in multigraph.edges(data='in_L4', nbunch=ids) if relationship[2]]
    # aggregate to relationship[0] which is the id
    all_l3s = {k: [(i[1],i[2]) for i in list(v)] for k, v in groupby(all_l3s, key=lambda x: x[0])}
    all_l4s = {k: [(i[1],i[2]) for i in list(v)] for k, v in groupby(all_l4s, key=lambda x: x[0])}
    ########################################
    # clean the graph from the dois that where inferred
    multigraph.remove_nodes_from(ids)
    ########################################
    for doi in tqdm(ids, desc='Infer L5/L6'):
        if doi not in all_l3s and doi not in all_l4s:
            out[doi] = [
                (
                    {
                        'L1': None,
                        'L2': None,
                        'L3': None,
                        'L4': None,
                        'L5': None,
                        'L6': None,
                        'score_for_L3': 0.0, 
                        'score_for_L4': 0.0, 
                        'score_for_L5': 0.0
                    }
                )
            ]
        elif doi in all_l3s and doi not in all_l4s:
            # we only inferred L3s
            L3 = all_l3s[doi]
            l3_mapping_to_l2 = [(tup, list(L3_to_L2[tup[0]].keys())) for tup in L3]
            flatten_l3_to_l2 = [(tup[0], l2) for tup in l3_mapping_to_l2 for l2 in tup[1]]
            l2_mapping_to_l1 = [(tup[0], tup[1], L2_to_L1[tup[1]]) for tup in flatten_l3_to_l2]
            out[doi] = [
                (
                    {
                        'L1': triplet[2],
                        'L2': triplet[1],
                        'L3': triplet[0][0],
                        'L4': None,
                        'L5': None,
                        'L6': None,
                        'score_for_L3': triplet[0][1], 
                        'score_for_L4': 0.0, 
                        'score_for_L5': 0.0
                    }
                ) for triplet in l2_mapping_to_l1
            ]
        elif doi not in all_l3s and doi in all_l4s:
            # we only inferred L4s
            L4 = all_l4s[doi]
            my_triplets = [(L2_to_L1[list(L3_to_L2[L4_to_L3[tup[0]]].keys())[0]],
                            list(L3_to_L2[L4_to_L3[tup[0]]].keys())[0], L4_to_L3[tup[0]], tup) for tup in L4]
            ############################################
            # infer L5 and L6
            if not only_l4:
                my_triplets = infer_l5_l6(my_triplets, titles[doi], abstracts[doi])
            else:
                my_triplets = [(tup[0], tup[1], tup[2], tup[3], None, None) if len(tup) > 3 else (tup[0], tup[1], tup[2], None, None, None) for tup in my_triplets]
            out[doi] = [
                (
                    {
                        'L1': triplet[0],
                        'L2': triplet[1],
                        'L3': triplet[2],
                        'L4': triplet[3][0] if triplet[3] else None,
                        'L5': triplet[4][0] if triplet[4] else None,
                        'L6': triplet[5] if triplet[5] else None,
                        'score_for_L4': triplet[3][1] if triplet[3] else 0.0,
                        'score_for_L5': triplet[4][1] if triplet[4] else 0.0,
                    }
                ) for triplet in my_triplets
            ]
            ############################################
        else:
            # we inferred both L3s and L4s
            L3 = all_l3s[doi]
            L4 = all_l4s[doi]
            l3_mapping_to_l2 = [(tup, list(L3_to_L2[tup[0]].keys())) for tup in L3]
            flatten_l3_to_l2 = [(tup[0], l2) for tup in l3_mapping_to_l2 for l2 in tup[1]]
            l2_mapping_to_l1 = [(tup[0], tup[1], L2_to_L1[tup[1]]) for tup in flatten_l3_to_l2]
            filtered_l4 = [
                (l4, (L3[[l3[0] for l3 in L3].index(L4_to_L3[l4[0]])])) for l4 in L4 if
                        L4_to_L3[l4[0]] in [l3[0] for l3 in L3]
            ]
            my_tups = []
            for tup in l2_mapping_to_l1:
                if tup[0][0] in [i[1][0] for i in filtered_l4]:
                    my_tups.append((tup[2], tup[1], tup[0], filtered_l4[[i[1][0] for i in filtered_l4].index(tup[0][0])][0]))
                else:
                    my_tups.append((tup[2], tup[1], tup[0]))
            ############################################
            # infer the L5 and L6
            if not only_l4:
                my_tups = infer_l5_l6(
                    my_tups,
                    titles[doi],
                    abstracts[doi]
                )
            else:
                my_tups = [(tup[0], tup[1], tup[2], tup[3], None, None) if len(tup) > 3 else (tup[0], tup[1], tup[2], None, None, None) for tup in my_tups]
            ############################################
            out[doi] = [
                (
                    {
                        'L1': triplet[0],
                        'L2': triplet[1],
                        'L3': triplet[2][0],
                        'L4': triplet[3][0] if triplet[3] else None,
                        'L5': triplet[4][0] if triplet[4] else None,
                        'L6': triplet[5] if triplet[5] else None,
                        'score_for_L3': triplet[2][1] if triplet[2] else 0.0,
                        'score_for_L4': triplet[3][1] if triplet[3] else 0.0,
                        'score_for_L5': triplet[4][1] if triplet[4] else 0.0
                    }
                ) for triplet in my_tups
            ]
    return out


def add(multigraph, published_venues, cit_ref_venues):
    """
    Adds entities and relationships to the multigraph.

    Args:
        multigraph (Multigraph): The multigraph to add entities and relationships to.
        published_venues: A list of tuples representing the published venues.
            Each tuple contains a DOI and a venue name.
        cit_ref_venues: A list of tuples representing the cited/referenced venues.
            Each tuple contains a DOI and a venue name.
    """
    multigraph.add_entities(from_entities="doi", to_entities="venue", relationship_type="published",
                            relationships=published_venues)
    multigraph.add_entities(from_entities="doi", to_entities="venue", relationship_type="cites",
                            relationships=cit_ref_venues)


def one_ranking(l5s_to_keep, canditate_l5, my_occurences, my_graph):
    """
    Ranks the level 5 categories based on their scores and word occurrences.

    Args:
        l5s_to_keep (dict): A dictionary containing level 5 categories as keys and their associated keywords as values.
        canditate_l5 (list): A list of candidate level 5 categories to consider for ranking.
        my_occurences (dict): A dictionary containing the occurrences of each keyword.
        my_graph (dict): A dictionary representing the graph structure of the categories.

    Returns:
        list: A list of tuples containing the ranked level 4 categories, their highest ranked level 5 category, and the unique keywords associated with the level 5 category.

    """
    final_ranking = []
    l4_to_l5 = dict()
    for l5, kws in l5s_to_keep.items():
        # get l4
        l4 = '_'.join(l5.split('_')[:-1])
        if l4 not in l4_to_l5:
            l4_to_l5[l4] = []
            l4_to_l5[l4].append((l5, kws))
        else:
            l4_to_l5[l4].append((l5, kws))
    my_counter = dict()
    for l4, l5s in l4_to_l5.items():
        my_counter[l4] = len(l5s)
    # sorted counter by num of occurences
    sorted_counter = {key: l4_to_l5[key] for key in sorted(l4_to_l5, key=lambda x: len(l4_to_l5[x]), reverse=True)}
    # sort the values by the number of words in each l5
    for key, value in sorted_counter.items():
        sorted_counter[key] = sorted(value, key=lambda x: len(x[1]), reverse=True)
    # normalize the counter
    factor=1.0/sum(my_counter.values())
    my_counter = {key: value*factor for key, value in my_counter.items()}
    for l4, l5s in sorted_counter.items():
        for l5, kws in l5s:
            if l5 not in canditate_l5:
                continue
            total_score = sum([my_occurences[kw] * my_counter[l4] for kw in kws]) * len(kws)
            words_score = sum([my_occurences[kw] * my_graph[kw][l5][0]['in_L5'] for kw in kws])
            final_ranking.append((l5, total_score, words_score, kws))
    #########################################   
    # the above method produces a lot of level 5s under the same level 4 with the same score..re-rank them according 
    # to the score of each word in the l5
    sorted_final_ranking = sorted(final_ranking, key=lambda x: x[1], reverse=True)
    sorted_final_ranking_per_l4 = dict()
    for tup in sorted_final_ranking:
        l4 = '_'.join(tup[0].split('_')[:-1])
        if l4 not in sorted_final_ranking_per_l4:
            sorted_final_ranking_per_l4[l4] = []
            sorted_final_ranking_per_l4[l4].append(tup)
        else:
            sorted_final_ranking_per_l4[l4].append(tup)

    return [(l4, sorted(l5s, key= lambda x: x[2], reverse=True)[0], list(set([l1 for l_items in l5s for l1 in l_items[3]]))) for l4, l5s in sorted_final_ranking_per_l4.items()]


def filter_level_5(canditate_l5, my_graph, my_hits, only_text=False):
    """
    Filters level 5 candidates based on the given parameters.

    Args:
        canditate_l5 (list): List of level 5 candidates.
        my_graph (networkx.Graph): Graph containing the relationships between words.
        my_hits (list): List of hits.
        only_text (bool, optional): Flag indicating whether to consider only text. Defaults to False.

    Returns:
        list: List of filtered level 5 candidates.

    Raises:
        None
    """
    candidates = []
    candidates.extend(
        [(bi, [n for n in my_graph[bi] if 'L5' in my_graph.nodes[n]]) for bi in my_hits if bi in my_graph])
    words = []
    word_to_l5s = dict()
    for cand in candidates:
        words.append(cand[0])
        for c in cand[1]:
            try:
                word_to_l5s[c].add(cand[0])
            except KeyError:
                word_to_l5s[c] = {cand[0]}
    the_occurences = Counter(words)
    l5s_to_keep = {k: v for k, v in word_to_l5s.items() if len(v) > 1}
    if not l5s_to_keep:
        return []
    if only_text:
        # only for text
        #########################################
        results = one_ranking(l5s_to_keep, canditate_l5, the_occurences)
        # to the resulting L4 above the L5 add all the other words that match the L5s
        # get the words that we want to keep
        to_return_top_res = [(r[1][0], r[1][1], r[1][3], r[2]) for r in results[:2]]
        words_matched = set([w for r in to_return_top_res for w in r[3]])
        rest_of_words_to_check = set([kw for _, kws in l5s_to_keep.items() for kw in kws]).difference(words_matched)
        # remove from l5s_to_keep the l3s that we already have
        second_l5s_to_keep = {l5: kws.intersection(rest_of_words_to_check) for l5, kws in l5s_to_keep.items() if '_'.join(l5.split('_')[:-2]) not in ['_'.join(r[0].split('_')[:-2]) for r in to_return_top_res]}
        # do another cycle of ranking
        if second_l5s_to_keep:
            results2 = one_ranking(second_l5s_to_keep, canditate_l5, the_occurences)
            to_return_top_res.extend([(r[1][0], r[1][1], r[1][3], r[2]) for r in results2[:1]])
        return to_return_top_res
    else:
        final_ranking = []
        for l5, kws in l5s_to_keep.items():
            if l5 not in canditate_l5:
                continue
            total_score = 0
            for kw in kws:
                sc = my_graph[kw][l5][0]['in_L5']
                total_score += the_occurences[kw] * sc
            final_ranking.append((l5, total_score, kws))
        my_res = sorted(final_ranking, key=lambda x: x[1], reverse=True)[:3]
        return my_res


def test():
    # initializations
    my_venue_parser = VenueParser(abbreviation_dict='venues_maps.p')
    multigraph = MultiGraph('scinobo_inference_graph.p')
    text_processor = TextProcessor()
    my_title = """Embedding Biomedical Ontologies by Jointly Encoding Network Structure and Textual Node Descriptors"""
    my_abstract = """Network Embedding (NE) methods, which
    map network nodes to low-dimensional feature vectors, have wide applications in network analysis and bioinformatics. Many existing NE methods rely only on network structure, overlooking other information associated
    with the nodes, e.g., text describing the nodes.
    Recent attempts to combine the two sources of
    information only consider local network structure. We extend NODE2VEC, a well-known NE
    method that considers broader network structure, to also consider textual node descriptors
    using recurrent neural encoders. Our method
    is evaluated on link prediction in two networks derived from UMLS. Experimental results demonstrate the effectiveness of the proposed approach compared to previous work."""

    payload = {
        "doi": "10.18653/v1/w19-5032",
        "doi_cites_venues": {
            "10.18653/v1/w19-5032": {
                "acl": 2,
                "aimag": 1,
                "arxiv artificial intelligence": 1,
                "arxiv computation and language": 2,
                "arxiv machine learning": 1,
                "arxiv social and information networks": 1,
                "briefings in bioinformatics": 1,
                "comparative and functional genomics": 1,
                "conference of the european chapter of the association for computational linguistics": 1,
                "cvpr": 1,
                "emnlp": 3,
                "eswc": 1,
                "iclr": 2,
                "icml": 1,
                "ieee trans signal process": 1,
                "j mach learn res": 1,
                "kdd": 4,
                "naacl": 1,
                "nips": 1,
                "nucleic acids res": 1,
                "pacific symposium on biocomputing": 3,
                "physica a statistical mechanics and its applications": 1,
                "proceedings of the acm conference on bioinformatics computational biology and health informatics": 1,
                "sci china ser f": 1,
                "the web conference": 1
            }
        },
        "doi_publish_venue": {
            "10.18653/v1/w19-5032": {
                "proceedings of the bionlp workshop and shared task": 1
            }
        },
        "emphasize": "citations"
    }

    my_res = infer(payload, multigraph, my_venue_parser)
    my_l4 = my_res["10.18653/v1/w19-5032"][0]['L4'] if 'L4' in my_res["10.18653/v1/w19-5032"][0] else None
    if my_l4 is None:
        return
    title = text_processor.preprocess_text(my_title)
    abstract = text_processor.preprocess_text(my_abstract)
    my_text = title + ' ' + abstract
    my_l5s = [node[0] for node in multigraph.nodes(data='L5') if node[1] and my_l4 in node[0]]
    l5s = filter_level_5(
        my_text, my_l5s, False, 5, text_processor, multigraph, only_text=False
    )
    pprint(l5s)


def yielder_json(input_dir, files):
    """
    A generator function that yields the contents of JSON files in the given directory.

    Args:
        input_dir (str): The directory path where the JSON files are located.
        files (list): A list of file names to process.

    Yields:
        tuple: A tuple containing the loaded JSON data and the file name.

    """
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(input_dir, file), 'r') as fin:
                yield json.load(fin), file
    

def yielder_parquet(input_dir, files):
    """
    Generator function that yields each parquet file in the given directory.

    Args:
        input_dir (str): The directory path where the parquet files are located.
        files (list): A list of file names in the directory.

    Yields:
        tuple: A tuple containing the pandas DataFrame read from the parquet file and the file name.

    """
    for file in files:
        if file.endswith('.parquet'):
            yield pd.read_parquet(os.path.join(input_dir, file)).fillna('N/A'), file


def create_payload(dato):
    """
    Create a payload for inference based on the given data.

    Args:
        dato (list): A list of dictionaries containing publication data.

    Returns:
        dict: A dictionary representing the payload for inference. The dictionary has the following structure:
            {
                'dois': [],
                'cit_ref_venues': {},
                'published_venues': {},
                'titles': {},
                'abstracts': {}
            }
            - 'dois' (list): A list of publication IDs.
            - 'cit_ref_venues' (dict): A dictionary mapping publication IDs to weighted cited/referenced venues.
            - 'published_venues' (dict): A dictionary mapping publication IDs to published venues.
            - 'titles' (dict): A dictionary mapping publication IDs to publication titles.
            - 'abstracts' (dict): A dictionary mapping publication IDs to publication abstracts.
    """    
    payload = {
        'dois': [],
        'cit_ref_venues': {},
        'published_venues': {},
        'titles': {},
        'abstracts': {}
    }
    for d in tqdm(dato, desc='Creating payload for inference'):
        # input checks
        if 'id' not in d:
            continue
        my_id = d['id']
        if 'cit_venues' not in d and 'ref_venues' not in d:
            continue
        elif 'cit_venues' in d and 'ref_venues' not in d:
            cit_venues = d['cit_venues']
        elif 'cit_venues' not in d and 'ref_venues' in d:
            ref_venues = d['ref_venues']
        else:
            cit_venues = d['cit_venues']
            ref_venues = d['ref_venues']
        if 'pub_venue' not in d:
            pub_venue = ''
        else:
            pub_venue_res = venue_parser.preprocess_venue(d['pub_venue'])
            if pub_venue_res is None:
                pub_venue = ''
            else:
                pub_venue, _ = pub_venue_res[0], pub_venue_res[1]
        # preprocess the venues
        ##############################################
        counts_ref = []
        counts_cit = []
        for ven in cit_venues:
            res = venue_parser.preprocess_venue(ven)
            if res is None:
                continue
            else:
                pre_ven, _ = res[0], res[1]
            counts_cit.append(pre_ven)
        for ven in ref_venues:
            res = venue_parser.preprocess_venue(ven)
            if res is None:
                continue
            else:
                pre_ven, _ = res[0], res[1]
            counts_ref.append(pre_ven)
        ##############################################
        counts_ref = Counter([ven for ven in counts_ref if ven is not None]).most_common()
        counts_cit = Counter([ven for ven in counts_cit if ven is not None]).most_common()
        ##############################################
        weighted_referenced_venues = {}
        for c in counts_ref:
            weighted_referenced_venues[c[0]] = c[1]

        for c in counts_cit:
            try:
                weighted_referenced_venues[c[0]] += c[1]
            except KeyError:
                weighted_referenced_venues[c[0]] = c[1]
        ##############################################
        payload['cit_ref_venues'][my_id] = weighted_referenced_venues
        ##############################################
        if pub_venue != '':
            payload['published_venues'][my_id] = {pub_venue: 1}
        else:
            payload['published_venues'][my_id] = {}
        ##############################################
        payload['dois'].append(my_id)
        if 'title' not in d or d['title'] == 'NULL':
            payload['titles'][my_id] = ''
        else:
            payload['titles'][my_id] = d['title']
        if 'abstract' not in d or d['abstract'] == 'NULL':
            payload['abstracts'][my_id] = ''
        else:
            payload['abstracts'][my_id] = d['abstract']
    return payload


def process_pred(res, ftype, metadata=None, extra=False):
    """
    Process the predictions and generate a list of dictionaries containing the processed results.

    Args:
        res (dict): A dictionary containing the predictions.
        ftype (str): The file type of the input data.
        metadata (list, optional): A list of dictionaries containing metadata information. Defaults to None.
        extra (bool, optional): A flag indicating whether to include extra information in the processed results. Defaults to False.

    Returns:
        list: A list of dictionaries containing the processed results.

    Raises:
        None

    """
    # filter L4 and assign names while processing the predictions
    if ftype == 'jsonl':
        res_to_dump = []
        if extra and metadata is not None:
            chunk_dict = {d['id']: d for d in metadata}
        for k, v in res.items():
            if extra and metadata is not None:
                dato = chunk_dict[k]
                pub_year = dato['pub_year'] if 'pub_year' in dato and str(dato['pub_year']) != 'N/A' else None
                citations_per_year = list(dato['citations_per_year']) if 'citations_per_year' in dato and str(dato['citations_per_year']) != "N/A" else None
                doi = dato['doi'] if 'doi' in dato and str(dato['doi']) != 'N/A' else None
                res_to_dump.extend([{
                    'id': k, 
                    'fos_predictions': [
                        {
                            'L1': pr['L1'], 
                            'L2': pr['L2'], 
                            'L3': pr['L3'], 
                            'L4': None if pr['L3'] == 'developmental biology' or pr['L4'] not in level_4_ids_2_names or level_4_ids_2_names[pr['L4']] == 'N/A' else level_4_ids_2_names[pr['L4']], 
                            'L5': pr['L5'], 
                            'L6': pr['L6']    
                        } for pr in v[:2]
                    ],
                    'fos_scores': [
                        {
                            'score_for_level_3': pr['score_for_L3'],
                            'score_for_level_4': pr['score_for_L4'],
                            'score_for_level_5': pr['score_for_L5']
                        } for pr in v[:2]
                    ],
                    'pub_year': pub_year,
                    'citations_per_year': citations_per_year,
                    'doi': doi
                }])
            else:                        
                res_to_dump = [
                    {
                        'id': k, 
                        'fos_predictions': [
                            {
                                'L1': pr['L1'], 
                                'L2': pr['L2'], 
                                'L3': pr['L3'], 
                                'L4': None if pr['L3'] == 'developmental biology' or pr['L4'] not in level_4_ids_2_names or level_4_ids_2_names[pr['L4']] == 'N/A' else level_4_ids_2_names[pr['L4']], 
                                'L5': pr['L5'], 
                                'L6': pr['L6']    
                            } for pr in v[:2]
                        ],
                        'fos_scores': [
                            {
                                'score_for_level_3': pr['score_for_L3'],
                                'score_for_level_4': pr['score_for_L4'],
                                'score_for_level_5': pr['score_for_L5']
                            } for pr in v[:2]
                        ]
                    } for k, v in res.items()
                ]
    else:
        res_to_dump = []
        if extra and metadata is not None:
            chunk_dict = {d['id']: d for d in metadata}
        for k, v in res.items():
            if extra and metadata is not None:
                dato = chunk_dict[k]
                pub_year = dato['pub_year'] if 'pub_year' in dato and str(dato['pub_year']) != 'N/A' else None
                citations_per_year = list(dato['citations_per_year']) if 'citations_per_year' in dato and str(dato['citations_per_year']) != "N/A" else None
                doi = dato['doi'] if 'doi' in dato and str(dato['doi']) != 'N/A' else None
                res_to_dump.extend([{
                    'id': k,
                    'L1': pr['L1'],
                    'L2': pr['L2'],
                    'L3': pr['L3'],
                    'L4': None if pr['L3'] == 'developmental biology' or pr['L4'] not in level_4_ids_2_names or level_4_ids_2_names[pr['L4']] == 'N/A' else level_4_ids_2_names[pr['L4']],
                    'L5': pr['L5'], 
                    'L6': pr['L6'],
                    'score_for_L3': pr['score_for_L3'],
                    'score_for_L4': pr['score_for_L4'],
                    'pub_year': pub_year,
                    'citations_per_year': citations_per_year,
                    'doi': doi
                } for pr in v[:2]])
            else:
                res_to_dump.extend([{
                    'id': k,
                    'L1': pr['L1'],
                    'L2': pr['L2'],
                    'L3': pr['L3'],
                    'L4': None if pr['L3'] == 'developmental biology' or pr['L4'] not in level_4_ids_2_names or level_4_ids_2_names[pr['L4']] == 'N/A' else level_4_ids_2_names[pr['L4']],
                    'L5': pr['L5'], 
                    'L6': pr['L6'],
                    'score_for_L3': pr['score_for_L3'],
                    'score_for_L4': pr['score_for_L4']
                } for pr in v[:2]])
    return res_to_dump


def save_pred(res, ftype, opath):
    """
    Save the prediction results to a file.

    Args:
        res (list): The prediction results to be saved.
        ftype (str): The file type to save the results in. Valid options are 'jsonl' and any other value for parquet.
        opath (str): The output file path to save the results.

    Returns:
        None
    """
    if ftype == 'jsonl':
        with open(opath, 'w') as fout:
            json.dump(res, fout)
    else:
        # create the dataframe
        df = pd.DataFrame(res)
        # create the parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, opath)


def main():
    # parse args
    arguments = parse_args()
    # init the logger
    logging.basicConfig(
        filename=arguments.log_path,
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    logging.info("Running FoS inference")
    logger = logging.getLogger('inference')
    # create the output directory
    logger.info('Creating the output directory: {}'.format(arguments.out_path))
    os.makedirs(arguments.out_path, exist_ok=True)
    logger.info('Output directory created: {}'.format(arguments.out_path))
    # check if the input directory exists
    logger.info('Checking if the input directory exists: {}'.format(arguments.in_path))
    if not os.path.exists(arguments.in_path):
        logger.error('The input directory does not exist: {}'.format(arguments.in_path))
        raise Exception('The input directory does not exist: {}'.format(arguments.in_path))
    # make sure that the publications have the necessary metadata
    # get files and init yielder
    if arguments.file_type == 'jsonl':
        total_files = [f for f in os.listdir(arguments.in_path) if f.endswith('.json')]
        my_yielder = yielder_json
    else:
        total_files = [f for f in os.listdir(arguments.in_path) if f.endswith('.parquet')]
        my_yielder = yielder_parquet
    batch_size = arguments.batch_size
    for idx, tup in enumerate(tqdm(my_yielder(arguments.in_path, total_files), desc='Parsing input files for inference', total=len(total_files))):
        dato, file_name = tup[0], tup[1]
        # each dato has lines of publications
        # split the lines into chunks
        chunks = [dato[i:i + batch_size] for i in range(0, len(dato), batch_size)]
        chunk_predictions = []
        # parse the chunks -- for each chunk create the payload for inference
        logger.info(f'Inferring chunks of file number:{idx} and file name: {file_name}')
        for chunk in tqdm(chunks, desc=f'Inferring chunks of file number:{idx} and file name: {file_name}'):
            if arguments.file_type == 'parquet':
                chunk = chunk.fillna("NULL").to_dict('records')
            logger.info('Creating payload for chunk')
            payload_to_infer = create_payload(chunk)
            logger.info('Payload for chunk')
            # infer to Level 1 - Level 4
            logger.info('Inferring')
            infer_res = infer(
                payload = payload_to_infer,
                only_l4=arguments.only_l4
            )
            logger.info('Inference done for chunk')
            if arguments.extra_metadata:
                res_to_dump = process_pred(infer_res, arguments.file_type, metadata=chunk, extra=True)
            else:
                res_to_dump = process_pred(infer_res, arguments.file_type)
            chunk_predictions.extend(res_to_dump)
        # dump the predictions
        logger.info(f'Dumping the predictions for the file with index: {idx} and file name: {file_name}')
        output_file_name = os.path.join(arguments.out_path, file_name)
        save_pred(chunk_predictions, arguments.file_type, output_file_name)
        
        
if __name__ == '__main__':
    main()