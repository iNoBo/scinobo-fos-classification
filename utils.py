"""

This module contains functions for preprocessing data.

"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import re
import pycountry
import nltk
import torch

# this is for blocking tensorflow -- it reserves all the gpu memory for some reason
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 

import spacy

from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from gensim.parsing.preprocessing import preprocess_string, strip_multiple_whitespaces, remove_stopwords, strip_punctuation

# download stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering


os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 


class TextProcessor():
    """Class for text processing."""
    def __init__(self, filters=None, my_lemmatizer='spacy'):
        """Initialize the class."""
        if filters is not None:
            self.filters = filters
        else:
            self.filters = [
                lambda x: x.lower(),
                self.preprocess,
                lambda x: re.sub(r'([^\w\s\\.,]|_)', ' ', x).strip(),
                strip_punctuation,
                strip_multiple_whitespaces
            ]
        
        if torch.cuda.is_available():
            self.device = f'cuda:{0}'
        else:
            self.device = 'cpu'
        print(f'Using device: {self.device}')
        self.cwd = os.getcwd()
        self.embedder = SentenceTransformer(f'all-mpnet-base-v2', device=self.device, cache_folder=self.cwd)
        self.spacy_model = spacy.load("en_core_web_sm")
        self.lemmatizer_to_use = my_lemmatizer
        if self.lemmatizer_to_use == 'spacy':
            self.lemmatizer = self.spacy_model
        else:
            self.lemmatizer = WordNetLemmatizer()
        
        self.bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                                     t.replace('"', '').replace('/', ' ').replace('\\', '').replace("'",
                                                                                                    '').strip().lower())
                                                                                                  
        self.input_embeddings = 'graph_embeddings_with_L6_21_12_2022.p'
        self.embeddings = self.load_embeddings()
        self.node2idx = {key: idx for idx, key in enumerate(self.embeddings.keys())}
        self.idx2node = {v: k for k, v in self.node2idx.items()}

        # convert self.input_embeddings to a tensor
        self.embeddings = torch.tensor(list(self.embeddings.values()), device=self.device)

        my_langs = [
            'de', 'it', 'cs', 'da', 'lv', 'es', 'fr', 'bg', 'pl', 'nl', 'el', 'fi', 'sv', 'ro', 'ga', 'hu',
            'sk', 'hr', 'pt', 'no', 'sl', 'lt', 'lb', 'et', 'mt', 'so', 'he', 'tr', 'ru', 'th',
            'fa', 'ar', 'hi', 'af', 'sq', 'sw', 'ca', 'zh', 'mk', 'ko', 'ur', 'ml', 'vi', 'uk', 'id', 'bn', 'tl', 'ja'
        ]

        # add self.my_stopwords to the list of stopwords
        self.my_stopwords = set(stopwords.words('english'))
        self.my_stopwords.add('et al')
        self.my_stopwords.add('al')
        self.my_stopwords.add('et')
        self.my_stopwords.add('the')
        self.my_stopwords.add('update')
        self.my_stopwords.add('recent')
        self.my_stopwords.add('method')
        self.my_stopwords.add('different')
        self.my_stopwords.add('conclusion')
        self.my_stopwords.add('review')
        self.my_stopwords.add('case')
        self.my_stopwords.add('case study')
        self.my_stopwords.add('study')
        self.my_stopwords.add('an')
        self.my_stopwords.add('overview')
        self.my_stopwords.add('approach')
        self.my_stopwords.add('view')
        self.my_stopwords.add('key')
        self.my_stopwords.add('analysis')
        self.my_stopwords.add('trend')
        self.my_stopwords.add('general')
        self.my_stopwords.add('classic')
        self.my_stopwords.add('model')
        self.my_stopwords.add('step')
        self.my_stopwords.add('each')
        self.my_stopwords.add('amount')
        self.my_stopwords.add('>')
        self.my_stopwords.add('<')
        self.my_stopwords.add('interest')
        self.my_stopwords.add("publisher's")
        self.my_stopwords.add("ethic")
        self.my_stopwords.add("approval")
        self.my_stopwords.add("additional")
        self.my_stopwords.add("file")
        self.my_stopwords.add("supplementary")
        self.my_stopwords.add("supplementary material")
        self.my_stopwords.add("author")
        self.my_stopwords.add("publisher")
        self.my_stopwords.add("future research")
        self.my_stopwords.add("future work")
        self.my_stopwords.add("work")
        self.my_stopwords.add("future")
        self.my_stopwords.add("impact")
        self.my_stopwords.add("literature")
        self.my_stopwords.add("goal")
        self.my_stopwords.add("scope")
        self.my_stopwords.add("definition")
        self.my_stopwords.add("cost")
        self.my_stopwords.add("challenge")
        self.my_stopwords.add("objective")
        self.my_stopwords.add("application")
        self.my_stopwords.add("scope")
        self.my_stopwords.add("present")
        self.my_stopwords.add("status")
        self.my_stopwords.add("co")
        self.my_stopwords.add("-")
        self.my_stopwords.add(":")
        self.my_stopwords.add("outlook")
        self.my_stopwords.add("potential")
        self.my_stopwords.add("united states")
        self.my_stopwords.add("france")
        self.my_stopwords.add("greece")
        self.my_stopwords.add("india")
        self.my_stopwords.add("germany")
        self.my_stopwords.add("project")
        self.my_stopwords.add("product")
        self.my_stopwords.add("china")
        self.my_stopwords.add("japan")
        self.my_stopwords.add("south korea")
        self.my_stopwords.add("part ii")
        self.my_stopwords.add("brazil")
        self.my_stopwords.add("new")
        self.my_stopwords.add("background")
        self.my_stopwords.add("datum")
        self.my_stopwords.add("acknowledgement")
        self.my_stopwords.add("consent")
        self.my_stopwords.add("funding")
        self.my_stopwords.add("creation")
        self.my_stopwords.add("job")
        self.my_stopwords.add("part")
        self.my_stopwords.add('comprehensive')
        self.my_stopwords.add('survey')
        self.my_stopwords.add('research')
        self.my_stopwords.add('introduction')
        self.my_stopwords.add('discussion')
        self.my_stopwords.add('description')
        self.my_stopwords.update(my_langs)
        self.my_stopwords.update([cntr.alpha_2.lower() for cntr in pycountry.countries])
        self.my_stopwords.update([cntr.name.lower() for cntr in pycountry.countries])
        self.my_stopwords.update([cntr.alpha_3.lower() for cntr in pycountry.countries])

    def load_embeddings(self):
        with open(self.input_embeddings, 'rb') as fin:
            embeddings = pickle.load(fin)
        return embeddings

    def preprocess(self, x):
        """Preprocess text."""
        return re.sub(r'\s+', ' ', re.sub(r'&.*?;(?:\w*;|#|/?(?:span|p|strong))*', ' ', re.sub(r'<.*?>', ' ', x))).strip()
    
    def wordnet_lemmatize(self, x):
        return self.lemmatizer.lemmatize(x)
    
    def spacy_lemmatizer(self, x):
        spacy_doc = self.lemmatizer(x)
        return ' '.join([token.lemma_ for token in spacy_doc])
    
    def get_spacy_doc(self, x):
        return self.spacy_model(x)

    def preprocess_text(self, x):
        x = ' '.join([tok for tok in preprocess_string(x, filters=self.filters) if tok not in self.my_stopwords])
        # lemmatize
        if self.lemmatizer_to_use == 'spacy':
            x = self.spacy_lemmatizer(x)
        else:
            x = ' '.join(self.wordnet_lemmatize(tok) for tok in x.split())
        return x        

    def get_ngrams(self, x, k):
        return [' '.join(ng) for ng in ngrams(sequence=nltk.word_tokenize(x), n=k)]
    
    def retrieve_similar_nodes(self, query, k):
        try:
            query_embedding = self.embedder.encode(query, convert_to_tensor=True, device=self.device, show_progress_bar=False)
        except RuntimeError:
            print('Error in encoding query')
            return []
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=k, query_chunk_size=1000)
        # unpack hits and convert to nodes
        return [[(q, self.idx2node[h['corpus_id']]) for h in hit if h['score'] >= 0.8] for q, hit in zip(query, hits)]

    def cluster_kws(self, corpus_words, thresh):
        # embedder = SentenceTransformer(self.sentence_transformer_name, device=f'cuda:{self.device_id}')
        print('Embedding corpus words...')
        corpus_embeddings = self.embedder.encode(corpus_words)
        # Normalize the embeddings to unit length
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=thresh,
            linkage='average',
            affinity='cosine'
        )
        print('Clustering...')
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        my_clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in my_clustered_sentences:
                my_clustered_sentences[cluster_id] = []
            my_clustered_sentences[cluster_id].append(corpus_words[sentence_id])
        return my_clustered_sentences
