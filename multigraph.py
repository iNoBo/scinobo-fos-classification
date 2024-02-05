# /usr/bin/python3.6
# -*- coding: utf-8 -*-

import networkx as nx
import os
import logging
import pickle

from tqdm import tqdm

def mydate():
    from datetime import date
    return date.today().strftime("%m_%d_%y")


class MultiGraph(nx.MultiDiGraph):

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
        assert (os.path.isfile(path))

    def add_entities(self, from_entities, to_entities, relationship_type, relationships, cutoff=0):

        sources = [k for k in relationships.keys() if k]
        targets = [k for d in relationships.values() for k, v in d.items() if k]
        self.add_nodes_from(sources)
        self.add_nodes_from(targets)
        for node in sources:
            self.nodes[node][from_entities] = True
        for node in targets:
            self.nodes[node][to_entities] = True
        for (k, v) in relationships.items():
            # edw pou exei to if (k,t) not in self.edges(): an ena doi einai kai published alla kanei kai cite, tote 8a mpei mono mia akmi
            # opoia baleis prwti, eite to publish eite to cite
            # self.add_edges_from([(k, t, {relationship_type:v[t]/sum(list(v.values()))}) for t in v if v[t] >= cutoff if (k,t) not in self.edges() if k if t] ) 
            self.add_edges_from(
                [(k, t, {relationship_type: v[t] / sum(list(v.values()))}) for t in v if v[t] >= cutoff if k if t])

    def modified_add_entities(self, from_entities, to_entities, relationship_type, relationships, cutoff=0,
                              ignore=set([])):

        self.add_nodes_from([k for k in relationships.keys() if k not in ignore], entity=from_entities)
        self.add_nodes_from([k for d in relationships.values() for k, v in d.items() if k not in ignore],
                            entity=to_entities)
        for (k, v) in tqdm(relationships.items(), disable=True):
            if k in ignore:
                continue

    def load(self, path):
        super().__init__(pickle.load(open(path, 'rb')))

    def plot_degree_dist(self, nbins):
        import matplotlib.pyplot as plt
        degrees = [self.degree(n) for n in self.nodes()]
        plt.hist(degrees, nbins)
        plt.savefig(self.path.split(".")[0] + "_degree_distribution.png", bbox_inches="tight")

    def export_gephi(self, filename="multigraph.gexf"):
        nx.write_gexf(self, filename)

    def annotation_coverage(self, relationship, entity):
        coverage = 0
        entities = [n[0] for n in self.nodes(data=entity) if n[1]]
        for entity in entities:
            existing_assignments = [field[1:] for field in self.edges(data=relationship, nbunch=entity) if field[2]]
            if existing_assignments:
                coverage += 1
        coverage = coverage / (len(entities))
        return coverage

    def infer_layer(self, entity_chain, relationship_chain, overwrite=False, max_links=2, filters=[0, 0],
                    new_relationship="default"):
        assert (len(relationship_chain) == 2)
        assert (len(entity_chain) == 3)
        assert (len(filters) <= 2)

        """ Predefines for working along with the API"""
        starting_entities = [n[0] for n in list(self.nodes(data=entity_chain[0])) if n[1] if n[0]]

        new_edges = {}
        for entity in tqdm(starting_entities, disable=False, desc='Inferring relationships'):

            """ Predefines for working along with the API"""
            edgelist_existing = list(self.edges(data=relationship_chain[1], nbunch=entity))
            edgelist_neighbors = list(self.edges(data=relationship_chain[0], nbunch=entity))

            if not overwrite:
                existing_assignments = [field[1:] for field in edgelist_existing if
                                        field[2]]
                if existing_assignments:
                    continue


            # edw an exeis balei idi mia fora sto doi_publish_venue ena venue kai to kanei kai cite, tote to xaneis sto cite
            neighbors = [neigh[1:] for neigh in edgelist_neighbors if neigh[2] if
                         self.nodes[neigh[1]][entity_chain[1]] if neigh[2] >= filters[0] if neigh[1]]
            aggregate_dict = {}
            for neigh in neighbors:
                prev_weight = neigh[1]

                """ Predefines for working along with the API"""
                edgelist_fos = list(self.edges(data=relationship_chain[1], nbunch=neigh[0]))

                fos = [field[1:] for field in edgelist_fos if field[2] if
                       self.nodes[field[1]][entity_chain[2]] if field[2] >= filters[1] if field[1]]
                total_weights = sum([field[1] for field in fos])
                for field in fos:
                    fieldname = field[0]
                    curr_weight = field[1] / total_weights
                    try:
                        aggregate_dict[fieldname] += prev_weight * curr_weight
                    except KeyError:
                        aggregate_dict[fieldname] = prev_weight * curr_weight
                    '''
                    try:
                        assert(aggregate_dict[fieldname]<=1)
                    except AssertionError:
                        print(aggregate_dict[fieldname],'invalid weight')
                        print('entity',entity)
                        print('neighbor fos',fos)
                        assert(False)
                    '''
            candidates = sorted(aggregate_dict.items(), key=lambda kv: kv[1], reverse=True)
            new_edges[entity] = candidates[:max_links]

        if new_relationship == "default":
            new_relationship = relationship_chain[1]
        for entity, edges in new_edges.items():
            for edge in edges:
                self.add_edge(entity, edge[0])
                self.edges[entity, edge[0], 0][new_relationship] = edge[1]

    def infer_layer_one(self, entity, entity_chain, relationship_chain, overwrite=False, max_links=2, filters=[0, 0],
                    new_relationship="default"):
        assert (len(relationship_chain) == 2)
        assert (len(entity_chain) == 3)
        assert (len(filters) <= 2)

        new_edges = {}

        """ Predefines for working along with the API"""
        edgelist_existing = list(self.edges(data=relationship_chain[1], nbunch=entity))
        edgelist_neighbors = list(self.edges(data=relationship_chain[0], nbunch=entity))

        if not overwrite:
            existing_assignments = [field[1:] for field in edgelist_existing if
                                    field[2]]
            if existing_assignments:
                return

        # edw an exeis balei idi mia fora sto doi_publish_venue ena venue kai to kanei kai cite, tote to xaneis sto cite
        neighbors = [neigh[1:] for neigh in edgelist_neighbors if neigh[2] if
                     self.nodes[neigh[1]][entity_chain[1]] if neigh[2] >= filters[0] if neigh[1]]
        aggregate_dict = {}
        for neigh in neighbors:
            prev_weight = neigh[1]

            """ Predefines for working along with the API"""
            edgelist_fos = list(self.edges(data=relationship_chain[1], nbunch=neigh[0]))

            fos = [field[1:] for field in edgelist_fos if field[2] if
                   self.nodes[field[1]][entity_chain[2]] if field[2] >= filters[1] if field[1]]
            total_weights = sum([field[1] for field in fos])
            for field in fos:
                fieldname = field[0]
                curr_weight = field[1] / total_weights
                try:
                    aggregate_dict[fieldname] += prev_weight * curr_weight
                except KeyError:
                    aggregate_dict[fieldname] = prev_weight * curr_weight
                '''
                try:
                    assert(aggregate_dict[fieldname]<=1)
                except AssertionError:
                    print(aggregate_dict[fieldname],'invalid weight')
                    print('entity',entity)
                    print('neighbor fos',fos)
                    assert(False)
                '''
        candidates = sorted(aggregate_dict.items(), key=lambda kv: kv[1], reverse=True)
        new_edges[entity] = candidates[:max_links]

        if new_relationship == "default":
            new_relationship = relationship_chain[1]
        for entity, edges in new_edges.items():
            for edge in edges:
                self.add_edge(entity, edge[0])
                self.edges[entity, edge[0], 0][new_relationship] = edge[1]

    def __init__(self, path=None):
        super().__init__()
        if path and os.path.isfile(path):
            self.load(path)
