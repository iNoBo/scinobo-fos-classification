""" 

This script is used for quantizing the graph embeddings. 
Saves the quantized embeddings in the same directory as the original embeddings.

args = [
    '--input', 'path/to/embeddings',
    '--output', 'path/to/output/embeddings',
    '--quantization', 'binary'
]
"""

import os
import pickle
import json
import argparse
import torch
import faiss
import numpy as np

from usearch.index import Index
from sentence_transformers.quantization import quantize_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description='Quantize graph embeddings')
    parser.add_argument('--input', type=str, required=True, help='Path to the input embeddings')
    parser.add_argument('--output', type=str, required=True, help='Path to the output embeddings')
    parser.add_argument('--quantization', type=str, default="binary", help='Number of bits to quantize the embeddings to')
    return parser.parse_args()


def main():
    ################
    # read arguments
    args = parse_args()
    input_path = args.input
    output_path = args.output
    quantization = args.quantization
    # check if output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    ################
    # load the embeddings
    with open(input_path, 'rb') as fin:
        embeddings = pickle.load(fin)
    # create word2idx and idx2word
    word2idx = {word: idx for idx, word in enumerate(embeddings.keys())}
    idx2word = {idx: word for word, idx in word2idx.items()}
    # they are loaded as a dict --> convert to tensor
    embeddings = torch.tensor(list(embeddings.values()))
    ################
    quantized_embeddings = quantize_embeddings(embeddings, quantization) # ubinary quantization
    binary_index = faiss.IndexBinaryFlat(embeddings.size(1))
    binary_index.add(quantized_embeddings)
    faiss.write_index_binary(binary_index, os.path.join(output_dir, 'graph_embeddings_faiss_ubinary.index'))
    # Convert the embeddings to "int8" for efficiently loading int8 indices with usearch
    int8_embeddings = quantize_embeddings(embeddings, "int8")
    index = Index(ndim=embeddings.size(1), metric="ip", dtype="i8")
    index.add(np.arange(len(int8_embeddings)), int8_embeddings)
    index.save(os.path.join(output_dir, "graph_embeddings_usearch_int8.index"))
    ################
    # save the quantized embeddings -- not a pickle
    with open(output_path, 'wb') as fout:
        torch.save(quantized_embeddings, fout)
    # save the word2idx and idx2word
    with open(os.path.join(output_dir, 'word2idx.json'), 'w') as fout:
        json.dump(word2idx, fout)
    with open(os.path.join(output_dir, 'idx2word.json'), 'w') as fout:
        json.dump(idx2word, fout)
    

if __name__ == '__main__':
    main()