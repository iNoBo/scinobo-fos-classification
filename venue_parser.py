"""

This script contains the preprocessing of the venues, and the abbreviation extractor

"""

import re
import logging
import os
import pickle

from collections import Counter
from ast import literal_eval
from pprint import pprint
from tqdm import tqdm


class VenueParser():

    def __init__(self, abbreviation_dict):

        # my regexes
        short_date = r"(?:\b(?<!\d\.)(?:(?:(?:[0123]?[0-9](?:[\.\-\/\~][0123]?[0-9])?(?:[\.\-\/(\s{1,2})]))(?:([0123]?[0-9])[\.\-\/][12][0-9]{3}|(\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\b)([\.\-\/(\s{1,2})][12][0-9]{3})?))|(?:[0123]?[0-9][\.\-\/][0123]?[0-9][\.\-\/][12]?[0-9]{2,3}))(?!\.\d)\b)"
        self.date_fallback = r"(?:(?:\b(?!\d\.)(?:(?:([0123]?[0-9])(?:(?:st|nd|rd|n?th)?\s?(\b(?:[Jj]an[.]?(?:uary)?|[Ff]eb[.]?(?:ruary)?|[Mm]ar[.]?(?:ch)?|[Aa]pr[.]?(?:il)?|May|[Jj]un[.]?(?:e)?|[Jj]ul[.]?(?:y)?|[Aa]ug[.]?(?:ust)?|[Ss]ept[.]??(?:ember)?|[Oo]ct[.]?(?:ober)?|[Nn]ov[.]?(?:ember)?|[Dd]ec[.]?(?:ember)?))?\s?[\.\-\/\~]\s?)([0123]?[0-9])(?:st|nd|rd|n?th)?(?:[\.\-\/(\s{1,2})])\s?(?:(\b(?:[Jj]an[.]?(?:uary)?|[Ff]eb[.]?(?:ruary)?|[Mm]ar[.]?(?:ch)?|[Aa]pr[.]?(?:il)?|May|[Jj]un[.]?(?:e)?|[Jj]ul[.]?(?:y)?|[Aa]ug[.]?(?:ust)?|[Ss]ept[.]??(?:ember)?|[Oo]ct[.]?(?:ober)?|[Nn]ov[.]?(?:ember)?|[Dd]ec[.]?(?:ember)?))\s?([1-3][0-9]{3})?\b)(?!\.\d)\b)))|(?:(\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\b))(?:\s)([0123]?[0-9])(?:\s?[\.\-\/\~]\s?)([0123]?[0-9]))"

        full_date_parts = [
            # prefix
            r"(?:(?<!:)\b\'?\d{1,4},? ?)",

            # month names
            r"\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\b",

            # suffix
            r"(?:(?:,? ?\'?)?\d{1,4}(?:st|nd|rd|n?th)?\b(?:[,\/]? ?\'?\d{2,4}[a-zA-Z]*)?(?: ?- ?\d{2,4}[a-zA-Z]*)?(?!:\d{1,4})\b)",
        ]

        __fd1 = "(?:{})".format("".join(
            [full_date_parts[0] + "?", full_date_parts[1], full_date_parts[2]]))
        __fd2 = "(?:{})".format("".join(
            [full_date_parts[0], full_date_parts[1], full_date_parts[2] + "?"]))

        self.date = "(?:" + "(?:" + __fd1 + "|" + __fd2 + ")" + "|" + short_date + ")"
        self.months_regex = r'\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\b'
        self.blacklist_words = r'(?i)(day|invited talks|oral session|speech given|posters|volume|issue)'
        self.word_ordinals = r'(?i)(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)'
        self.space_between_chars = r'([^\w\s\\.,]|_)'
        self.days_abbrev = ['mon', 'tue', 'thu', 'wed', 'fri', 'sat', 'sun']

        with open(abbreviation_dict, 'rb') as fin:
            self.abbreviation_dict = pickle.load(fin)

        self.abbreviation_dict["meeting of the association for computational linguistics"] = "acl"

    def get_abbreviations(self, string, cleaned_string):
        if not '(' in string or not ')' in string:
            if re.search(r'(?:[\-]\s+)([A-Z]+)\b', string) is None:
                return None

        # for some reason got an attribute error must investigate
        # the error occurs from bad entris at elastic: e.g. jem ) chem (
        try:
            abbrev = re.search(r'\((.*?)\)', string).group(1)  # get content of parenthesis

            # remove abbrev from the cleaned_string
            # -------------------------------------
            abbrev_to_remove = '\\b' + re.escape(abbrev).lower() + '\\b'
            cleaned_string = re.sub(abbrev_to_remove, '', cleaned_string)
            cleaned_string = re.sub(' +', ' ', cleaned_string).strip()

            if not cleaned_string or len(cleaned_string) < 3:
                return None
            # -------------------------------------
            # -------------------------------------
        except AttributeError:
            try:
                abbrev = re.search(r'(?:[\-]\s+)([A-Z]+)\b', string).group(1)
            except AttributeError:
                return None

        if not abbrev or len(abbrev) < 3:
            return None

        abbrev = abbrev.lower()
        abbrev = re.sub(r'[^a-z ]+', '', abbrev).strip()
        abbrev = re.sub(' +', ' ', abbrev)
        if not abbrev or len(abbrev) < 3:
            return None
        try:
            firstletters = [s[0] for s in cleaned_string.split(' ')]
        except:
            firstletters = [cleaned_string[0]]
        prev = "X"
        mismatch = 0

        for pos, letter in enumerate(abbrev):
            try:
                index = firstletters.index(letter)
                del firstletters[index]
            except ValueError:
                if prev + letter not in cleaned_string:
                    if mismatch == 1:
                        return None
                    else:
                        mismatch += 1
                else:
                    mismatch -= 1
            finally:
                prev = letter
        return abbrev

    def preprocess(self, string, get_abbrv=True):

        cleaned_string = re.sub(r'\([^)]*\)', '', string)
        cleaned_string = re.sub(self.space_between_chars, ' ', cleaned_string).lower().strip()

        # remove dates
        cleaned_string = re.sub(self.date, '', cleaned_string)
        cleaned_string = re.sub(self.date_fallback, '', cleaned_string)

        cleaned_string = re.sub(r'\b\d{1,4}(?:st|nd|rd|n?th)\b', '', cleaned_string)
        cleaned_string = re.sub(r'[^a-z ]+', '', cleaned_string).strip()

        # remove days, months
        cleaned_string = re.sub(self.months_regex, '', cleaned_string)

        # remove word ordinals and blacklist words
        cleaned_string = re.sub(self.word_ordinals, '', cleaned_string)
        cleaned_string = re.sub(self.blacklist_words, '', cleaned_string)

        # remove extra spaces
        cleaned_string = re.sub(' +', ' ', cleaned_string).strip()

        if not cleaned_string or len(cleaned_string) < 3 or cleaned_string in self.days_abbrev:
            return None, False

        # used for when you just want to preprocess a string
        if not get_abbrv:
            return cleaned_string, False

        # way faster than checking with if
        try:
            return self.abbreviation_dict[cleaned_string], True
        except KeyError:
            abbrev = self.get_abbreviations(string, cleaned_string)
            if abbrev:
                self.abbreviation_dict[cleaned_string] = abbrev
                return abbrev, True
            else:
                return cleaned_string, False

    def is_journal_or_conference(self, x):
        if 'journal' in x or 'conference' in x:
            return x
        sus_tokens = ['workshop', 'proceedings', 'thesis', 'bachelor', 'symposium']
        for sus in sus_tokens:
            if sus in x:
                return None
        return x

    def get_venue_crossref(self, citation):

        try:
            return self.preprocess_venue(citation['journal-title'])[0]
        except KeyError:
            return None

    def citation_year_crossref(self, citation):
        try:
            return int(citation['year'])
        except:
            return -1

    def preprocess_venue(self, venue, get_abbrv=True):
        # remove latin numbers
        if venue != 'IV':
            try:
                venue = re.sub(r'\b(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b', '', venue)
            except TypeError:
                return None, False

        blacklist = {'n/a', 'na', 'none', '', 'null', 'otherdata', 'nodata', 'unknown', '', None, 'author', 'crossref',
                     'arxiv', 'Crossref', 'Arxiv'}
        if venue in blacklist:
            return None, False
        else:
            venue = self.preprocess(venue, get_abbrv)
        return venue

    def preprocess_venues(self, venues):

        venues = [v.lower().strip() for v in venues if v]
        blacklist = {'n/a', 'na', 'none', '', 'null', 'otherdata', 'nodata', 'unknown', '', None, 'author', 'crossref',
                     'arxiv', 'Crossref', 'Arxiv'}
        venues = [v for v in venues if v not in blacklist]  # and self.is_journal_or_conference(v)]

        venues = list(map(self.preprocess, venues))
        return venues

    def postprocess(self, D):
        old = D.copy()
        merged = 0
        for k, v in old.items():
            try:
                mapped = self.abbreviation_dict[k]
                merged += 1
                if mapped not in D.keys():
                    D[mapped] = {}
            except KeyError:
                continue
            for neighbor, weight in v.items():
                try:
                    D[mapped][neighbor] += weight
                except KeyError:
                    D[mapped][neighbor] = weight
            del D[k]
        logging.info('postprocessing merged %d keys', merged)
        return D

    def extract_venue(self, doi_data):
        short_flag = True
        if doi_data['_source']['type'] == 'journal-article' or doi_data['_source']['type'] == 'proceedings-article':

            try:
                exists = doi_data['_source']['short-container-title']
                if not exists:
                    short_flag = False
            except KeyError:
                short_flag = False

            my_venues = doi_data['_source']['container-title']

            if len(my_venues) > 1 and short_flag:

                ven_abbrev = self.preprocess_venue(doi_data['_source']['short-container-title'][0], get_abbrv=False)[0]

                if ven_abbrev is None:
                    ven = self.preprocess_venue(my_venues[0])[0]

                    if ven is None:
                        return None, False
                    else:
                        return ven, False
                else:
                    for ven in my_venues:

                        # preprocess venue name but do not get abbreviation
                        # get_abbrv = True by default
                        ven = self.preprocess_venue(ven, get_abbrv=False)[0]

                        if ven is None:
                            continue
                        else:
                            self.abbreviation_dict[ven] = ven_abbrev

                    return ven_abbrev, False

            elif len(my_venues) == 1 and short_flag:

                ven_abbrev = self.preprocess_venue(doi_data['_source']['short-container-title'][0], get_abbrv=False)[0]

                if ven_abbrev is None:

                    prep_venue = self.preprocess_venue(my_venues[0])[0]
                    if prep_venue is None:
                        return None, False

                    return prep_venue, False

                else:
                    prep_venue = self.preprocess_venue(my_venues[0], get_abbrv=False)[0]
                    if prep_venue is not None:
                        self.abbreviation_dict[prep_venue] = ven_abbrev

                    return ven_abbrev, False

            else:

                if not my_venues:
                    return None, False
                venue = self.preprocess_venue(my_venues[0])[0]
                if venue is None:
                    return None, False
                else:
                    return venue, False
        else:
            return None, False

    def extract_year(self, doi_data, my_range=[]):
        try:
            year = doi_data['_source']['published-online'].split('/')[0]
        except KeyError:
            try:
                year = doi_data['_source']['published-print'].split('/')[0]
            except KeyError:
                try:
                    year = doi_data['_source']['issued']
                    if year is None:
                        return None
                    else:
                        year = doi_data['_source']['issued'].split('/')[0]
                except KeyError:
                    return None

        if my_range:
            if int(year) in my_range:
                return year
            else:
                return None
        else:
            return year
