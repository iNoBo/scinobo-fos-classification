""" 

This script is used to test the inference of Field of Science (FoS) from publication data.
We will test the following:
    - create_payload function
        - check if it can handle wrong input --> it must be a list of dictionaries
        - check if it can handle missing fields
        - check if it can create the correct output, based on format
    - add function
        - check if it adds correctly the entities and relationships to the graph
    - infer_relationship function
        - check if it can correctly infer the relationships between entities in the graph
    - infer_fos function
        - 

"""

import unittest
from fos.inference import infer, infer_relationship, infer_l5_l6, create_payload
from fos.multigraph import MultiGraph


class InferenceTestCase(unittest.TestCase):
    def test_create_payload(self):
        # Test create_payload function with sample data
        data = [
            {
                "id": "1",
                "title": "Publication 1",
                "abstract": "Abstract 1",
                "pub_venue": "Venue 1",
                "cit_venues": ["Venue 2", "Venue 3"],
                "ref_venues": ["Venue 4", "Venue 5"]
            },
            {
                "id": "2",
                "title": "Publication 2",
                "abstract": "Abstract 2",
                "pub_venue": "Venue 6",
                "cit_venues": ["Venue 7", "Venue 8"],
                "ref_venues": ["Venue 9", "Venue 10"]
            }
        ]
        payload = create_payload(data)
        expected_payload = {
            "dois": ["1", "2"],
            "published_venues": ["Venue 1", "Venue 6"],
            "cit_ref_venues": ["Venue 2", "Venue 3", "Venue 4", "Venue 5", "Venue 7", "Venue 8", "Venue 9", "Venue 10"]
        }
        self.assertEqual(payload, expected_payload)

    def test_infer_relationship(self):
        # Test infer_relationship function with sample data
        multigraph = MultiGraph()
        multigraph.add_node("doi1", "venue1", "L4_1")
        multigraph.add_node("doi2", "venue2", "L4_2")
        infer_relationship(multigraph, top_L3=2, top_L4=3, overwrite=True, relationship="cites")
        self.assertEqual(multigraph.get_relationships("doi1", "venue1", "L4_1"), [("cites", "in_L4", "doi2", "venue2", "L4_2")])
        self.assertEqual(multigraph.get_relationships("doi2", "venue2", "L4_2"), [])

    def test_infer_l5_l6(self):
        # Test infer_l5_l6 function with sample data
        tups = [("L4_1", 0.8), ("L4_2", 0.6)]
        title = "Sample Title"
        abstract = "Sample Abstract"
        preds = infer_l5_l6(tups, title, abstract)
        expected_preds = [("L4_1", 0.8, "L5_1", "L6_1"), ("L4_2", 0.6, None, None)]
        self.assertEqual(preds, expected_preds)

    def test_infer(self):
        # Test infer function with sample data
        payload = {
            "dois": ["doi1", "doi2"],
            "published_venues": ["venue1", "venue2"],
            "cit_ref_venues": ["venue3", "venue4"]
        }
        result = infer(payload=payload)
        expected_result = [
            ("doi1", "venue1", "L4_1", "L5_1", "L6_1"),
            ("doi2", "venue2", "L4_2", None, None)
        ]
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()