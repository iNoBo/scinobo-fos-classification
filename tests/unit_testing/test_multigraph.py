""" 
This script is used to test the my_graph class in the fos package.
We will test the following:
    - test the add_entities function --> adds all the entities, 
    adds the relationships, the scores are normalized and correct
    - test the infer_layer function
"""

import unittest

from fos.graph_utils import MyMultiGraph


class TestMultiGraph(unittest.TestCase):
    def setUp(self):
        self.graph = MyMultiGraph() # create an empty graph

    def test_add_entities(self):
        # Test adding entities and relationships
        from_entities = ['A', 'B', 'C']
        to_entities = ['D', 'E', 'F']
        relationship_type = 'relation'
        relationships = {'A': {'D': 1}, 'B': {'E': 2}, 'C': {'F': 3}}
        self.graph.add_entities(from_entities, to_entities, relationship_type, relationships)

        # Assert that the entities and relationships are added correctly
        self.assertEqual(self.graph.number_of_nodes(), 6)
        self.assertEqual(self.graph.number_of_edges(), 3)
        self.assertEqual(self.graph.edges['A', 'D', 0]['relation'], 1)
        self.assertEqual(self.graph.edges['B', 'E', 0]['relation'], 2)
        self.assertEqual(self.graph.edges['C', 'F', 0]['relation'], 3)

    def test_infer_layer(self):
        # Test inferring relationships between entities
        entity_chain = ['A', 'B', 'C']
        relationship_chain = ['relation1', 'relation2']
        self.graph.add_entities(['A', 'B', 'C'], ['D', 'E', 'F'], 'relation1', {'A': {'D': 1}, 'B': {'E': 2}, 'C': {'F': 3}})
        self.graph.add_entities(['D', 'E', 'F'], ['G', 'H', 'I'], 'relation2', {'D': {'G': 4}, 'E': {'H': 5}, 'F': {'I': 6}})
        self.graph.infer_layer(entity_chain, relationship_chain)

        # Assert that the relationships are inferred correctly
        self.assertEqual(self.graph.number_of_nodes(), 9)
        self.assertEqual(self.graph.number_of_edges(), 6)
        self.assertEqual(self.graph.edges['A', 'D', 1]['relation2'], 4)
        self.assertEqual(self.graph.edges['B', 'E', 1]['relation2'], 5)
        self.assertEqual(self.graph.edges['C', 'F', 1]['relation2'], 6)

if __name__ == '__main__':
    unittest.main()