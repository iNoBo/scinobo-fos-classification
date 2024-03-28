""" 
This script is used to test the utils script in the fos package.
We will test the following:
    - the text processing function works as expected
    - the get_ngrams function works as expected
    - the retrieve_similar_nodes function works as expected -- handles when empty input
    - the cluster_kws function is not used on this repo -- do not test
"""

import unittest

from fos.utils import TextProcessor


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.text_processor = TextProcessor()

    def test_preprocess(self):
        # Test if the preprocess function removes stopwords, punctuation, and multiple whitespaces
        text = "This is a test sentence with stopwords, punctuation, and multiple    whitespaces."
        expected_output = "test sentence stopwords punctuation multiple whitespaces"
        self.assertEqual(self.text_processor.preprocess_text(text), expected_output)

    def test_get_ngrams(self):
        # Test if the get_ngrams function generates n-grams correctly
        text = "This is a test sentence."
        k = 2
        expected_output = ["This is", "is a", "a test", "test sentence"]
        self.assertEqual(self.text_processor.get_ngrams(text, k), expected_output)

    def test_retrieve_similar_nodes(self):
        # Test if the retrieve_similar_nodes function retrieves similar nodes correctly
        query = ["machine learning"]
        k = 5
        expected_output = ["artificial intelligence", "data science", "deep learning", "neural networks", "computer vision"]
        self.assertEqual(self.text_processor.retrieve_similar_nodes(query, k), expected_output)
        # Test if the retrieve_similar_nodes function handles empty input
        query = [""]
        k = 5
        expected_output = []
        self.assertEqual(self.text_processor.retrieve_similar_nodes(query, k), expected_output)


if __name__ == '__main__':
    unittest.main()
