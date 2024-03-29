""" 
This script is used to test the venue_parser class in the fos package.
We will test the following:
    - test the get_abbreviations function
    - test the preprocess_venue function
    - test the preprocess function
"""

import unittest
# ------------------------------------------------------------ #
import sys
sys.path.append("./src") # since it is not installed yet, we need to add the path to the module 
# -- this is for when cloning the repo
# ------------------------------------------------------------ #

from fos.pipeline.venue_parser import VenueParser


class TestVenueParser(unittest.TestCase):
    def setUp(self):
        abbreviation_dict = "src/fos/data/venues_maps.p"
        self.parser = VenueParser(abbreviation_dict)

    def test_get_abbreviations(self):
        string = "Empirical Methods in Natural Language Processing"
        abbreviations, _ = self.parser.preprocess_venue(string)
        self.assertEqual(abbreviations, "emnlp")

    def test_preprocess_venue_blacklist(self):
        venue = "n/a"
        preprocessed_venue, _ = self.parser.preprocess_venue(venue)
        self.assertEqual(preprocessed_venue, None)

    def test_preprocess_venue_invalid_input(self):
        venue = 12345
        preprocessed_venue, _ = self.parser.preprocess_venue(venue)
        self.assertEqual(preprocessed_venue, None)

if __name__ == '__main__':
    unittest.main()