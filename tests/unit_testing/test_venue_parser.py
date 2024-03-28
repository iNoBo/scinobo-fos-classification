""" 
This script is used to test the venue_parser class in the fos package.
We will test the following:
    - test the get_abbreviations function
    - test the preprocess_venue function
    - test the preprocess function
"""

import unittest
from fos.venue_parser import VenueParser


class TestVenueParser(unittest.TestCase):
    def setUp(self):
        abbreviation_dict = "src/fos/data/venues_maps.p"
        self.parser = VenueParser(abbreviation_dict)

    def test_get_abbreviations(self):
        string = "Proceedings of the International Conference on Machine Learning"
        cleaned_string = "proceedings international conference machine learning"
        abbreviations = self.parser.get_abbreviations(string, cleaned_string)
        self.assertEqual(abbreviations, "ICML")

    def test_preprocess(self):
        string = "Proceedings of the International Conference on Machine Learning"
        preprocessed_string = self.parser.preprocess(string)
        self.assertEqual(preprocessed_string, "proceedings international conference machine learning")

    def test_preprocess_venue(self):
        venue = "Proceedings of the International Conference on Machine Learning"
        preprocessed_venue = self.parser.preprocess_venue(venue)
        self.assertEqual(preprocessed_venue, "proceedings international conference machine learning")

    def test_preprocess_venue_remove_latin_numbers(self):
        venue = "Proceedings of the International Conference on Machine Learning IV"
        preprocessed_venue = self.parser.preprocess_venue(venue)
        self.assertEqual(preprocessed_venue, "proceedings international conference machine learning")

    def test_preprocess_venue_blacklist(self):
        venue = "n/a"
        preprocessed_venue = self.parser.preprocess_venue(venue)
        self.assertEqual(preprocessed_venue, None)

    def test_preprocess_venue_invalid_input(self):
        venue = 12345
        preprocessed_venue = self.parser.preprocess_venue(venue)
        self.assertEqual(preprocessed_venue, None)

if __name__ == '__main__':
    unittest.main()