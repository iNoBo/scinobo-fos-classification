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
# ------------------------------------------------------------ #
import sys
sys.path.append("./src") # since it is not installed yet, we need to add the path to the module 
# -- this is for when cloning the repo
# ------------------------------------------------------------ #
from fos.pipeline.inference import infer, infer_l5_l6, create_payload


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
            "cit_ref_venues": {'1': {'venue': 4}, '2': {'venue': 4}},
            "published_venues": {'1': {'venue': 1}, '2': {'venue': 1}},
            "titles": {'1': 'Publication 1', '2': 'Publication 2'},
            "abstracts": {'1': 'Abstract 1', '2': 'Abstract 2'}
        }
        self.assertEqual(payload, expected_payload)

    def test_infer_l5_l6(self):
        # Test infer_l5_l6 function with sample data
        tups = [("L4_1", 0.8), ("L4_2", 0.6)]
        title = "Sample Title"
        abstract = "Sample Abstract"
        preds = infer_l5_l6(tups, title, abstract)
        expected_preds = [("L4_1", 0.8, None, None, None), ("L4_2", 0.6, None, None, None)]
        self.assertEqual(preds, expected_preds)

    def test_infer(self):
        # Test infer function with sample data
        payload = {
            "dois": ["1", "2"],
            "cit_ref_venues": {'1': {'venue': 4}, '2': {'venue': 4}},
            "published_venues": {'1': {'venue': 1}, '2': {'venue': 1}},
            "titles": {'1': 'Publication 1', '2': 'Publication 2'},
            "abstracts": {'1': 'Abstract 1', '2': 'Abstract 2'}
        }
        result = infer(payload=payload)
        expected_result = {
            '1': [{'L1': None, 'L2': None, 'L3': None, 'L4': None, 'L5': None, 'L6': None, 'score_for_L3': 0.0, 'score_for_L4': 0.0, 'score_for_L5': 0.0}], 
            '2': [{'L1': None, 'L2': None, 'L3': None, 'L4': None, 'L5': None, 'L6': None, 'score_for_L3': 0.0, 'score_for_L4': 0.0, 'score_for_L5': 0.0}]
        }
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()