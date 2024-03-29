""" 
This script is used to test the my_graph class in the fos package.
We will test the following:
    - test the add_entities function --> adds all the entities, 
    adds the relationships, the scores are normalized and correct
    - test the infer_layer function
"""

import unittest
import os
# ------------------------------------------------------------ #
import sys
sys.path.append("./src") # since it is not installed yet, we need to add the path to the module 
# -- this is for when cloning the repo
# ------------------------------------------------------------ #

from fos.pipeline.multigraph import MultiGraph


DATA_PATH = "./src/fos/data"


class TestMultiGraph(unittest.TestCase):
    def setUp(self):
        self.graph = MultiGraph(os.path.join(DATA_PATH, 'scinobo_inference_graph.json')) # create an empty graph

    def test_add_entities(self):
        curr_nodes = self.graph.number_of_nodes()
        curr_edges = self.graph.number_of_edges()
        # Test adding entities and relationships
        from_entities = 'example_pub'
        to_entities = 'example_venue'
        relationship_type = 'example_relation'
        relationships = {'example_A': {'example_D': 1}, 'example_B': {'example_E': 2}, 'example_C': {'example_F': 3}}
        self.graph.add_entities(from_entities, to_entities, relationship_type, relationships)
        
        # Assert that the entities and relationships are added correctly
        self.assertEqual(self.graph.number_of_nodes(), curr_nodes + 6)
        self.assertEqual(self.graph.number_of_edges(), curr_edges + 3)
        self.assertEqual(self.graph.edges['example_A', 'example_D', 0]['example_relation'], 1)
        self.assertEqual(self.graph.edges['example_B', 'example_E', 0]['example_relation'], 1)
        self.assertEqual(self.graph.edges['example_C', 'example_F', 0]['example_relation'], 1)

    def test_infer_layer(self):
        # Test inferring relationships between entities
        entity_chain = ['doi', 'venue', 'L4']
        relationship_chain = ['cites', 'in_L4']
        self.graph.add_entities(
            'doi', 'venue', 'cites', 
            {'example_A': {'emnlp': 1}, 'example_B': {'emnlp': 2}, 'example_C': {'emnlp': 3}}
        )
        self.graph.add_entities(
            'doi', 'venue', 'cites', 
            {'example_D': {'kdd': 4}, 'example_E': {'kdd': 5}, 'example_F': {'kdd': 6}}
        )
        self.graph.infer_layer(entity_chain, relationship_chain)
        # check if edges exist with L4_artificial intelligence & image_9 for example_A & example_B, example_C
        # check if edges exist with L4_information systems_3 for example_D & example_E, example_F
        self.assertIsNotNone(self.graph['example_D']['L4_information systems_3'])
        self.assertIsNotNone(self.graph['example_E']['L4_information systems_3'])
        self.assertIsNotNone(self.graph['example_F']['L4_information systems_3'])
        self.assertIsNotNone(self.graph['example_A']['L4_artificial intelligence & image_9'])
        self.assertIsNotNone(self.graph['example_B']['L4_artificial intelligence & image_9'])
        self.assertIsNotNone(self.graph['example_C']['L4_artificial intelligence & image_9'])

if __name__ == '__main__':
    unittest.main()