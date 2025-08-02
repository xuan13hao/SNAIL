import os
import io
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from scipy.sparse import csr_matrix
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import spacy
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.matcher import Matcher
import requests
import re
import nltk

nltk.data.path.append("./punkt_data")
import nltk
nltk.download('punkt_tab')
# Define good, weak, and blacklisted words (unchanged)
GOOD_WORDS = {
    "databases", "archive", "atlas", "catalog", "catalogue", "classification", "collection", "database",
    "gateway", "hierarchy", "index", "knowledgebase", "portal", "project", "registry", "repository",
    "resource", "software", "application", "browser", "frame-work", "library", "package", "pipe-line",
    "program", "programme", "server", "service", "simulator", "suite", "system", "tool", "tool-kit",
    "viewer", "web-services", "workbench"
}

WEAK_WORDS = {
    "descriptors", "alignment", "annotation", "entry", "home-page", "record", "term", "web-site",
    "achieve", "build", "calculate", "compile", "compute", "create", "develop", "execute",
    "implement", "process", "produce", "provide", "reference", "retrieve", "run", "simulate",
    "update", "use", "view", "interface", "platform", "revision", "version"
}

BLACKLISTED_WORDS = {
    "algorithm", "approach", "consortium", "estimator", "file format", "file", "format", "measure", 
    "method", "operating system", "programing language", "programming language", "statistic"
}

# Define spaCy patterns (unchanged)
patterns = [
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'such'}, {'LOWER': 'as'}, {'POS': 'NOUN', 'OP': '+'}],  # X, such as Y
    [{'LOWER': 'such'}, {'POS': 'NOUN'}, {'LOWER': 'as'}, {'POS': 'NOUN', 'OP': '+'}],  # such X as Y
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'other'}, {'POS': 'NOUN'}],  # X, other Y
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'include'}, {'POS': 'NOUN', 'OP': '+'}],  # X, include Y
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'especially'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'any'}, {'POS': 'NOUN'}],  # X, and/or any other Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'some'}, {'LOWER': 'other'}, {'POS': 'NOUN'}],  # X, and/or some other Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'be'}, {'LOWER': 'a'}, {'POS': 'NOUN'}],  # X, and/or be a Y
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'like'}, {'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}],  # X, like Y
    [{'LOWER': 'such'}, {'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'as'}, {'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}],  # such X, as Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'like'}, {'LOWER': 'other'}, {'POS': 'NOUN'}],  # X, and/or like other Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'one'}, {'LOWER': 'of'}, {'LOWER': 'the'}, {'POS': 'NOUN'}],  # X, and/or one of the Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'one'}, {'LOWER': 'of'}, {'LOWER': 'these'}, {'POS': 'NOUN'}],  # X, and/or one of these Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'one'}, {'LOWER': 'of'}, {'LOWER': 'those'}, {'POS': 'NOUN'}],  # X, and/or one of those Y
    [{'LOWER': 'example'}, {'LOWER': 'of'}, {'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'be'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}],  # example of X, be Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'be'}, {'LOWER': 'example'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],  # X, and/or be example of Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'for'}, {'LOWER': 'example'}, {'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}],  # X, for example, Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'be'}, {'LOWER': 'call'}, {'POS': 'NOUN'}],  # X, and/or which be call Y# X, especially Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'be'}, {'LOWER': 'name'}, {'POS': 'NOUN'}],  # X, and/or which be name Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'mainly'}, {'POS': 'NOUN', 'OP': '+'}],  # X, mainly Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'mostly'}, {'POS': 'NOUN', 'OP': '+'}],  # X, mostly Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'notably'}, {'POS': 'NOUN', 'OP': '+'}],  # X, notably Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'particularly'}, {'POS': 'NOUN', 'OP': '+'}],  # X, particularly Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'principally'}, {'POS': 'NOUN', 'OP': '+'}],  # X, principally Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'in'}, {'LOWER': 'particular'}, {'POS': 'NOUN', 'OP': '+'}],  # X, in particular Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'except'}, {'POS': 'NOUN', 'OP': '+'}],  # X, except Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'other'}, {'LOWER': 'than'}, {'POS': 'NOUN', 'OP': '+'}],  # X, other than Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'e.g.'}, {'LOWER': ',', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}],  # X, e.g., Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': '(', 'OP': '?'}, {'LOWER': 'e.g.'}, {'LOWER': ',', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': '.', 'OP': '?'}, {'LOWER': ')'}],  # X (e.g. Y.)
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'i.e.'}, {'LOWER': ',', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}],  # X, i.e., Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'a'}, {'LOWER': 'kind'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],  # X, and/or a kind of Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'kind'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],  # X, and/or kind of Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'form'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],  # X, and/or form of Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'look'}, {'LOWER': 'like'}, {'POS': 'NOUN'}],  # X, and/or which look like Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'sound'}, {'LOWER': 'like'}, {'POS': 'NOUN'}],  # X, and/or which sound like Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'be'}, {'LOWER': 'similar'}, {'LOWER': 'to'}, {'POS': 'NOUN'}],  # X, which be similar to Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'example'}, {'LOWER': 'of'}, {'LOWER': 'this'}, {'LOWER': 'be'}, {'POS': 'NOUN'}],  # X, example of this be Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'type'}, {'POS': 'NOUN', 'OP': '+'}],  # X, type Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}, {'LOWER': 'type'}],  # X, and/or Y type
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'whether'}, {'POS': 'NOUN', 'OP': '+'}],  # X, whether Y
    [{'LOWER': 'compare'}, {'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'with'}, {'POS': 'NOUN'}],  # compare X, and/or with Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'compare'}, {'LOWER': 'to'}, {'POS': 'NOUN', 'OP': '+'}],  # X, compare to Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'among'}, {'LOWER': '-PRON-'}, {'POS': 'NOUN', 'OP': '+'}],  # X, among -PRON- Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'for'}, {'LOWER': 'instance'}],  # X, Y for instance
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'sort'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],  # X, and/or sort of Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'may'}, {'LOWER': 'include'}, {'POS': 'NOUN', 'OP': '+'}] # X, which may include Y
]
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
# make sure this matches one of the paths in nltk.data.path
DOWNLOAD_DIR = "./nltk_data"

# download the full punkt package (this will include collocations.tab)
nltk.download('punkt', download_dir=DOWNLOAD_DIR)
nltk.download('averaged_perceptron_tagger', download_dir=DOWNLOAD_DIR)
nltk.download('wordnet', download_dir=DOWNLOAD_DIR)
nltk.download('omw-1.4', download_dir=DOWNLOAD_DIR)

# then add that directory so word_tokenize can see it
nltk.data.path.append(DOWNLOAD_DIR)
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000000
nlp.disable_pipes("ner", "parser")
lemmatizer = WordNetLemmatizer()


matcher = Matcher(nlp.vocab)
for idx, pattern in enumerate(patterns):
    matcher.add(f"PATTERN_{idx}", [pattern])
    
print(f"Total matcher rules: {len(matcher)}")

# Load BMC regex dictionary
bmc_dict_file = "dbsoftDB.tsv"

def parse_regex_pattern(raw_pattern):
    """Parse regex pattern, handling OR conditions and parentheses."""
    raw_pattern = raw_pattern.strip()
    # Remove outer () if they wrap the entire pattern
    if raw_pattern.startswith('(') and raw_pattern.endswith(')'):
        try:
            re.compile(raw_pattern)
            raw_pattern = raw_pattern[1:-1]
        except re.error:
            pass  # Keep as-is if unbalanced

    # Split on '|' at top level
    sub_patterns = []
    current = ""
    paren_count = 0
    for char in raw_pattern:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        elif char == '|' and paren_count == 0:
            if current:
                sub_patterns.append(current)
                current = ""
            continue
        current += char
    if current:
        sub_patterns.append(current)

    valid_patterns = []
    for sub in sub_patterns:
        sub = sub.strip()
        if sub:
            # Ensure sub-pattern is wrapped in () if it was originally
            if sub[0] != '(' and sub[-1] != ')':
                test_sub = f"({sub})"
            else:
                test_sub = sub
            try:
                re.compile(test_sub, re.IGNORECASE)
                valid_patterns.append(test_sub)
            except re.error as e:
                print(f"Warning: Invalid sub-pattern '{sub}': {e}")
    return valid_patterns

# Load BMC patterns
bmc_patterns = []
try:
    with open(bmc_dict_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    _, pattern = line.strip().split("\t")
                    sub_patterns = parse_regex_pattern(pattern)
                    for sub_pattern in sub_patterns:
                        try:
                            compiled_pattern = re.compile(sub_pattern, re.IGNORECASE)
                            bmc_patterns.append(compiled_pattern)
                        except re.error as e:
                            print(f"Warning: Failed to compile sub-pattern '{sub_pattern}': {e}")
                except ValueError as e:
                    print(f"Warning: Skipping malformed line in {bmc_dict_file}: {line.strip()} ({e})")
    print(f"Loaded {len(bmc_patterns)} BMC regex patterns")
except FileNotFoundError:
    print(f"Warning: {bmc_dict_file} not found, is_bmc_db will be 0")
except UnicodeDecodeError as e:
    print(f"Error: Failed to decode {bmc_dict_file}: {e}, is_bmc_db will be 0")

# Fallback patterns
if not bmc_patterns:
    fallback_patterns = [
        r"AAindex", r"AmiGO", r"ArkDB", r"ADAPT", r"ASAP", r"ALIGN",
        r"Amino\s+Acid\s+Index\s+[Dd]ata[Bb]ase",
        r"Annotation\s+[Dd]ata[Bb]ase\s+for\s+Affymetrix\s+Probesets\s+and\s+Transcripts",
        r"The\s+Ark"
    ]
    for pattern in fallback_patterns:
        try:
            bmc_patterns.append(re.compile(pattern, re.IGNORECASE))
        except re.error as e:
            print(f"Warning: Failed to compile fallback pattern '{pattern}': {e}")
    print(f"Using {len(bmc_patterns)} fallback BMC patterns")

def is_bmc_db(phrase):
    if not bmc_patterns:
        return 0
    for pattern in bmc_patterns:
        if pattern.fullmatch(phrase):
            return 1
    return 0
dbsoft_keywords = {
    "tool", "software", "application", "pipeline", "suite", "platform",
    "resource", "repository", "system", "framework", "package", "interface",
    "service", "webserver", "library", "database", "catalog", "collection"
}
def has_dbsoft_keyword(desc):
    return int(any(kw in desc.lower() for kw in dbsoft_keywords))
# Load SCOWL English words
scowl_dir = "scowl"
english_words = set()
for filename in os.listdir(scowl_dir):
    if filename.startswith("english-words"):
        with open(os.path.join(scowl_dir, filename), "r", encoding="latin1") as f:
            english_words.update(line.strip().lower() for line in f if line.strip())

def is_english_word(word):
    return 1 if word.lower() in english_words else 0

# Load English acronyms
acronyms = set()
for filename in os.listdir(scowl_dir):
    if filename.startswith("english-abbreviations"):
        with open(os.path.join(scowl_dir, filename), "r", encoding="latin1") as f:
            acronyms.update(line.strip().upper() for line in f if line.strip())
url = "https://www.allacronyms.com/A/abbreviations"
try:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    all_acronyms = [item.text.strip() for item in soup.select("a[href*='/abbreviation/']") if item.text.isupper()]
    acronyms.update(all_acronyms)
except requests.RequestException as e:
    print(f"Warning: Failed to fetch acronyms from {url}: {e}")

def is_english_acronym(word):
    return 1 if word.upper() in acronyms else 0

# Load Bioconductor package names
bioconductor_file = "bioconductor_packages.txt"
try:
    with open(bioconductor_file, "r", encoding="utf-8") as f:
        bioconductor_names = set(line.strip().lower() for line in f if line.strip())
except FileNotFoundError:
    print(f"Warning: {bioconductor_file} not found, using default set")
    bioconductor_names = {'limma', 'deseq2', 'edger', 'biomart'}

def is_bioconductor_name(word):
    return 1 if word.lower() in bioconductor_names else 0
def has_dbsoft_keyword(desc):
    return int(any(kw in desc.lower() for kw in dbsoft_keywords))

# Load bio acronyms
acronyms_file = "acronyms.dic"
def load_bio_acronyms(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if lines and lines[0].isdigit():
                acronym_lines = lines[1:]
            else:
                acronym_lines = lines
            acronyms = set(line.upper() for line in acronym_lines if line)
            if not acronyms:
                raise ValueError(f"No valid acronyms found in {file_path}")
            return acronyms
    except (FileNotFoundError, ValueError, UnicodeDecodeError) as e:
        print(f"Warning: Error loading {file_path}: {e}, using default set")
        return {'BLAST', 'PDB', 'NCBI', 'DBSNP'}

bio_acronyms = load_bio_acronyms(acronyms_file)

def is_bio_acronym(word):
    return 1 if word.upper() in bio_acronyms else 0

def normalize_word(word):
    variants = [
        word.lower(),
        word.lower().replace("-", ""),
        word.lower().replace("-", " "),
        lemmatizer.lemmatize(word.lower()),
        lemmatizer.lemmatize(word.lower().replace("-", "")),
        lemmatizer.lemmatize(word.lower().replace("-", " "))
    ]
    return set(variants)

def find_exact_tag_positions(soup, full_text, words):
    positions = []
    current_idx = 0
    for tag in soup.find_all(['database', 'software']):
        tag_text = tag.get_text(strip=True)
        search_start = full_text.find(tag_text, current_idx)
        if search_start != -1:
            before = full_text[:search_start]
            word_start = len(before.split())
            word_len = len(tag_text.split())
            if word_start + word_len <= len(words):
                positions.append((tag_text, word_start, word_len))
            current_idx = search_start + len(tag_text)
    return positions

def get_window_words(words, start_idx, length, window_size=8):
    end_idx = start_idx + length
    before = words[max(0, start_idx - window_size):start_idx]
    after = words[end_idx:min(len(words), end_idx + window_size)]
    return before + after

def extract_features(words, pos_dict, matched_phrases, phrase, idx, length):
    pos_tag_value = pos_dict.get(idx, 'misc')
    
    window_words = get_window_words(words, idx, length)
    window_lower = [w.lower() for w in window_words]
    
    window_variants = set()
    for w in window_lower:
        window_variants.update(normalize_word(w))
    
    good_variants = set()
    for w in GOOD_WORDS:
        good_variants.update(normalize_word(w))
    
    weak_variants = set()
    for w in WEAK_WORDS:
        weak_variants.update(normalize_word(w))
    
    blacklist_variants = set()
    for w in BLACKLISTED_WORDS:
        blacklist_variants.update(normalize_word(w))
    
    goodheadword = "yes" if window_variants & good_variants else "no"
    weakword = "yes" if window_variants & weak_variants else "no"
    blacklisted_word = "yes" if window_variants & blacklist_variants else "no"
    pattern_match = "yes" if any(w in matched_phrases for w in window_lower) else "no"

    is_eng_word = is_english_word(phrase)
    is_eng_acronym = is_english_acronym(phrase)
    is_bio_cond = is_bioconductor_name(phrase)
    is_bio_acro = is_bio_acronym(phrase)
    is_uppercase = 1 if phrase.isupper() else 0
    is_lowercase = 1 if phrase.islower() else 0
    is_mixedcase = 1 if (any(c.isupper() for c in phrase) and any(c.islower() for c in phrase)) else 0
    is_bmc_db_match = is_bmc_db(phrase)

    return {
        'word': phrase,
        'pos_tag': pos_tag_value,
        'goodheadword': goodheadword,
        'weakword': weakword,
        'blacklistedword': blacklisted_word,
        'pattern_match': pattern_match,
        'is_english_word': is_eng_word,
        'is_english_acronym': is_eng_acronym,
        'is_bioconductor_name': is_bio_cond,
        'is_bio_acronym': is_bio_acro,
        'is_uppercase': is_uppercase,
        'is_lowercase': is_lowercase,
        'is_mixedcase': is_mixedcase,
        'is_bmc_db': is_bmc_db_match
    }
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r"\w+")
def extract_features_for_all_words(html_file):
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    clean_text = soup.get_text(separator=' ')
    # print(clean_text)
    all_words = clean_text.split()
    all_words = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in all_words]
    all_words = [word for word in all_words if word]
    # tokenized_words = tokenizer.tokenize(clean_text)
    tokenized_words = word_tokenize(clean_text)
    pos_tags = pos_tag(tokenized_words)
    pos_dict = {i: tag[1] for i, tag in enumerate(pos_tags)}


    
    doc = nlp(clean_text)
    matched_phrases = set(span.text.lower() for _, start, end in matcher(doc) for span in [doc[start:end]])
    # doc = nlp(clean_text)
    # matches = list(matcher(doc))
    # matched_token_indices = set()
    # for _, start, end in matches:
    #     matched_token_indices.update(range(start, end))

    # for match_id, start, end in matches:
    # span = doc[start:end]
    # print(f"Matched span: '{span.text}' at tokens {start}-{end}")


    
    dbsoft_phrases = find_exact_tag_positions(soup, clean_text, all_words)
    
    processed_indices = set()
    features = []
    labels = []
    
    for phrase, start_idx, length in dbsoft_phrases:
        if start_idx not in processed_indices:
            feature = extract_features(all_words, pos_dict, matched_phrases, phrase, start_idx, length)
            #feature = extract_features(all_words, pos_dict, matched_token_indices, phrase, start_idx, length)
            features.append(feature)
            labels.append('dbsoft')
            for i in range(start_idx, start_idx + length):
                processed_indices.add(i)
    
    for i, word in enumerate(all_words):
        if i not in processed_indices:
            feature = extract_features(all_words, pos_dict, matched_phrases, word, i, 1)
            #feature = extract_features(all_words, pos_dict, matched_token_indices, word, i, 1)
            features.append(feature)
            labels.append('other')
            processed_indices.add(i)
    
    # print(f"Total features extracted: {len(features)}")
    # print(f"Number of 'dbsoft' labels: {sum(1 for l in labels if l == 'dbsoft')}")
    # print(f"Number of 'other' labels: {sum(1 for l in labels if l == 'other')}")
    return features, labels
import os
import io
from bs4 import BeautifulSoup
import numpy as np


from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.matcher import Matcher
import requests
import re
import nltk

nltk.data.path.append("./punkt_data")

# Define good, weak, and blacklisted words (unchanged)
GOOD_WORDS = {
    "databases", "archive", "atlas", "catalog", "catalogue", "classification", "collection", "database",
    "gateway", "hierarchy", "index", "knowledgebase", "portal", "project", "registry", "repository",
    "resource", "software", "application", "browser", "frame-work", "library", "package", "pipe-line",
    "program", "programme", "server", "service", "simulator", "suite", "system", "tool", "tool-kit",
    "viewer", "web-services", "workbench"
}

WEAK_WORDS = {
    "descriptors", "alignment", "annotation", "entry", "home-page", "record", "term", "web-site",
    "achieve", "build", "calculate", "compile", "compute", "create", "develop", "execute",
    "implement", "process", "produce", "provide", "reference", "retrieve", "run", "simulate",
    "update", "use", "view", "interface", "platform", "revision", "version"
}

BLACKLISTED_WORDS = {
    "algorithm", "approach", "consortium", "estimator", "file format", "file", "format", "measure", 
    "method", "operating system", "programing language", "programming language", "statistic"
}

# Define spaCy patterns (unchanged)
patterns = [
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'such'}, {'LOWER': 'as'}, {'POS': 'NOUN', 'OP': '+'}],  # X, such as Y
    [{'LOWER': 'such'}, {'POS': 'NOUN'}, {'LOWER': 'as'}, {'POS': 'NOUN', 'OP': '+'}],  # such X as Y
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'other'}, {'POS': 'NOUN'}],  # X, other Y
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'include'}, {'POS': 'NOUN', 'OP': '+'}],  # X, include Y
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'especially'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'any'}, {'POS': 'NOUN'}],  # X, and/or any other Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'some'}, {'LOWER': 'other'}, {'POS': 'NOUN'}],  # X, and/or some other Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'be'}, {'LOWER': 'a'}, {'POS': 'NOUN'}],  # X, and/or be a Y
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'like'}, {'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}],  # X, like Y
    [{'LOWER': 'such'}, {'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'as'}, {'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}],  # such X, as Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'like'}, {'LOWER': 'other'}, {'POS': 'NOUN'}],  # X, and/or like other Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'one'}, {'LOWER': 'of'}, {'LOWER': 'the'}, {'POS': 'NOUN'}],  # X, and/or one of the Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'one'}, {'LOWER': 'of'}, {'LOWER': 'these'}, {'POS': 'NOUN'}],  # X, and/or one of these Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'one'}, {'LOWER': 'of'}, {'LOWER': 'those'}, {'POS': 'NOUN'}],  # X, and/or one of those Y
    [{'LOWER': 'example'}, {'LOWER': 'of'}, {'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'be'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}],  # example of X, be Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'be'}, {'LOWER': 'example'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],  # X, and/or be example of Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'for'}, {'LOWER': 'example'}, {'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}],  # X, for example, Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'be'}, {'LOWER': 'call'}, {'POS': 'NOUN'}],  # X, and/or which be call Y# X, especially Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'be'}, {'LOWER': 'name'}, {'POS': 'NOUN'}],  # X, and/or which be name Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'mainly'}, {'POS': 'NOUN', 'OP': '+'}],  # X, mainly Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'mostly'}, {'POS': 'NOUN', 'OP': '+'}],  # X, mostly Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'notably'}, {'POS': 'NOUN', 'OP': '+'}],  # X, notably Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'particularly'}, {'POS': 'NOUN', 'OP': '+'}],  # X, particularly Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'principally'}, {'POS': 'NOUN', 'OP': '+'}],  # X, principally Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'in'}, {'LOWER': 'particular'}, {'POS': 'NOUN', 'OP': '+'}],  # X, in particular Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'except'}, {'POS': 'NOUN', 'OP': '+'}],  # X, except Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'other'}, {'LOWER': 'than'}, {'POS': 'NOUN', 'OP': '+'}],  # X, other than Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'e.g.'}, {'LOWER': ',', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}],  # X, e.g., Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': '(', 'OP': '?'}, {'LOWER': 'e.g.'}, {'LOWER': ',', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': '.', 'OP': '?'}, {'LOWER': ')'}],  # X (e.g. Y.)
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'i.e.'}, {'LOWER': ',', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}],  # X, i.e., Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'a'}, {'LOWER': 'kind'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],  # X, and/or a kind of Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'kind'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],  # X, and/or kind of Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'form'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],  # X, and/or form of Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'look'}, {'LOWER': 'like'}, {'POS': 'NOUN'}],  # X, and/or which look like Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'sound'}, {'LOWER': 'like'}, {'POS': 'NOUN'}],  # X, and/or which sound like Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'be'}, {'LOWER': 'similar'}, {'LOWER': 'to'}, {'POS': 'NOUN'}],  # X, which be similar to Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'example'}, {'LOWER': 'of'}, {'LOWER': 'this'}, {'LOWER': 'be'}, {'POS': 'NOUN'}],  # X, example of this be Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'type'}, {'POS': 'NOUN', 'OP': '+'}],  # X, type Y
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}, {'LOWER': 'type'}],  # X, and/or Y type
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'whether'}, {'POS': 'NOUN', 'OP': '+'}],  # X, whether Y
    [{'LOWER': 'compare'}, {'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'with'}, {'POS': 'NOUN'}],  # compare X, and/or with Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'compare'}, {'LOWER': 'to'}, {'POS': 'NOUN', 'OP': '+'}],  # X, compare to Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'among'}, {'LOWER': '-PRON-'}, {'POS': 'NOUN', 'OP': '+'}],  # X, among -PRON- Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'for'}, {'LOWER': 'instance'}],  # X, Y for instance
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'sort'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],  # X, and/or sort of Y
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'may'}, {'LOWER': 'include'}, {'POS': 'NOUN', 'OP': '+'}] # X, which may include Y
]
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
# make sure this matches one of the paths in nltk.data.path
DOWNLOAD_DIR = "./nltk_data"

# download the full punkt package (this will include collocations.tab)
nltk.download('punkt', download_dir=DOWNLOAD_DIR)
nltk.download('averaged_perceptron_tagger', download_dir=DOWNLOAD_DIR)
nltk.download('wordnet', download_dir=DOWNLOAD_DIR)
nltk.download('omw-1.4', download_dir=DOWNLOAD_DIR)

# then add that directory so word_tokenize can see it
nltk.data.path.append(DOWNLOAD_DIR)
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000000
nlp.disable_pipes("ner", "parser")
lemmatizer = WordNetLemmatizer()


matcher = Matcher(nlp.vocab)
for idx, pattern in enumerate(patterns):
    matcher.add(f"PATTERN_{idx}", [pattern])
    
print(f"Total matcher rules: {len(matcher)}")

# Load BMC regex dictionary
bmc_dict_file = "linnaeus.regex.dic.BMCPaper.tsv"

def parse_regex_pattern(raw_pattern):
    """Parse regex pattern, handling OR conditions and parentheses."""
    raw_pattern = raw_pattern.strip()
    # Remove outer () if they wrap the entire pattern
    if raw_pattern.startswith('(') and raw_pattern.endswith(')'):
        try:
            re.compile(raw_pattern)
            raw_pattern = raw_pattern[1:-1]
        except re.error:
            pass  # Keep as-is if unbalanced

    # Split on '|' at top level
    sub_patterns = []
    current = ""
    paren_count = 0
    for char in raw_pattern:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        elif char == '|' and paren_count == 0:
            if current:
                sub_patterns.append(current)
                current = ""
            continue
        current += char
    if current:
        sub_patterns.append(current)

    valid_patterns = []
    for sub in sub_patterns:
        sub = sub.strip()
        if sub:
            # Ensure sub-pattern is wrapped in () if it was originally
            if sub[0] != '(' and sub[-1] != ')':
                test_sub = f"({sub})"
            else:
                test_sub = sub
            try:
                re.compile(test_sub, re.IGNORECASE)
                valid_patterns.append(test_sub)
            except re.error as e:
                print(f"Warning: Invalid sub-pattern '{sub}': {e}")
    return valid_patterns

# Load BMC patterns
bmc_patterns = []
try:
    with open(bmc_dict_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    _, pattern = line.strip().split("\t")
                    sub_patterns = parse_regex_pattern(pattern)
                    for sub_pattern in sub_patterns:
                        try:
                            compiled_pattern = re.compile(sub_pattern, re.IGNORECASE)
                            bmc_patterns.append(compiled_pattern)
                        except re.error as e:
                            print(f"Warning: Failed to compile sub-pattern '{sub_pattern}': {e}")
                except ValueError as e:
                    print(f"Warning: Skipping malformed line in {bmc_dict_file}: {line.strip()} ({e})")
    print(f"Loaded {len(bmc_patterns)} BMC regex patterns")
except FileNotFoundError:
    print(f"Warning: {bmc_dict_file} not found, is_bmc_db will be 0")
except UnicodeDecodeError as e:
    print(f"Error: Failed to decode {bmc_dict_file}: {e}, is_bmc_db will be 0")

# Fallback patterns
if not bmc_patterns:
    fallback_patterns = [
        r"AAindex", r"AmiGO", r"ArkDB", r"ADAPT", r"ASAP", r"ALIGN",
        r"Amino\s+Acid\s+Index\s+[Dd]ata[Bb]ase",
        r"Annotation\s+[Dd]ata[Bb]ase\s+for\s+Affymetrix\s+Probesets\s+and\s+Transcripts",
        r"The\s+Ark"
    ]
    for pattern in fallback_patterns:
        try:
            bmc_patterns.append(re.compile(pattern, re.IGNORECASE))
        except re.error as e:
            print(f"Warning: Failed to compile fallback pattern '{pattern}': {e}")
    print(f"Using {len(bmc_patterns)} fallback BMC patterns")

def is_bmc_db(phrase):
    if not bmc_patterns:
        return 0
    for pattern in bmc_patterns:
        if pattern.fullmatch(phrase):
            return 1
    return 0
dbsoft_keywords = {
    "tool", "software", "application", "pipeline", "suite", "platform",
    "resource", "repository", "system", "framework", "package", "interface",
    "service", "webserver", "library", "database", "catalog", "collection"
}
def has_dbsoft_keyword(desc):
    return int(any(kw in desc.lower() for kw in dbsoft_keywords))
import enchant
d = enchant.Dict("en_US")

def is_english_word(word):
    if not word or word.strip() == "":
        return 0
    return d.check(word.lower())
scowl_dir = "scowl"
# english_words = load_scowl_words(scowl_dir)
# scowl_dir = "/home/h392x566/NER/SNAIL_Xgboost/scowl"
english_words = set()
for filename in os.listdir(scowl_dir):
    if filename.startswith("english-words"):
        with open(os.path.join(scowl_dir, filename), "r", encoding="latin1") as f:
            english_words.update(line.strip().lower() for line in f if line.strip())
# print(english_words)
# def is_english_word(word):
#     return 1 if word.lower() in english_words else 0

# Load English acronyms
acronyms = set()
for filename in os.listdir(scowl_dir):
    if filename.startswith("english-abbreviations"):
        with open(os.path.join(scowl_dir, filename), "r", encoding="latin1") as f:
            acronyms.update(line.strip().upper() for line in f if line.strip())
url = "https://www.allacronyms.com/A/abbreviations"
try:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    all_acronyms = [item.text.strip() for item in soup.select("a[href*='/abbreviation/']") if item.text.isupper()]
    acronyms.update(all_acronyms)
except requests.RequestException as e:
    print(f"Warning: Failed to fetch acronyms from {url}: {e}")
# print("acronyms", acronyms)
def is_english_acronym(word):
    return 1 if word.upper() in acronyms else 0

# Load Bioconductor package names
bioconductor_file = "bioconductor_packages.txt"
try:
    with open(bioconductor_file, "r", encoding="utf-8") as f:
        bioconductor_names = set(line.strip().lower() for line in f if line.strip())
except FileNotFoundError:
    print(f"Warning: {bioconductor_file} not found, using default set")
    bioconductor_names = {'limma', 'deseq2', 'edger', 'biomart'}

def is_bioconductor_name(word):
    return 1 if word.lower() in bioconductor_names else 0
def has_dbsoft_keyword(desc):
    return int(any(kw in desc.lower() for kw in dbsoft_keywords))

# Load bio acronyms
acronyms_file = "acronyms.dic"
def load_bio_acronyms(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if lines and lines[0].isdigit():
                acronym_lines = lines[1:]
            else:
                acronym_lines = lines
            acronyms = set(line.upper() for line in acronym_lines if line)
            if not acronyms:
                raise ValueError(f"No valid acronyms found in {file_path}")
            return acronyms
    except (FileNotFoundError, ValueError, UnicodeDecodeError) as e:
        print(f"Warning: Error loading {file_path}: {e}, using default set")
        return {'BLAST', 'PDB', 'NCBI', 'DBSNP'}

bio_acronyms = load_bio_acronyms(acronyms_file)

def is_bio_acronym(word):
    return 1 if word.upper() in bio_acronyms else 0

def normalize_word(word):
    variants = [
        word.lower(),
        word.lower().replace("-", ""),
        word.lower().replace("-", " "),
        lemmatizer.lemmatize(word.lower()),
        lemmatizer.lemmatize(word.lower().replace("-", "")),
        lemmatizer.lemmatize(word.lower().replace("-", " "))
    ]
    return set(variants)

def find_exact_tag_positions(soup, full_text, words):
    positions = []
    current_idx = 0
    for tag in soup.find_all(['database', 'software']):
        tag_text = tag.get_text(strip=True)
        search_start = full_text.find(tag_text, current_idx)
        if search_start != -1:
            before = full_text[:search_start]
            word_start = len(before.split())
            word_len = len(tag_text.split())
            if word_start + word_len <= len(words):
                positions.append((tag_text, word_start, word_len))
            current_idx = search_start + len(tag_text)
    return positions

def get_window_words(words, start_idx, length, window_size=8):
    end_idx = start_idx + length
    before = words[max(0, start_idx - window_size):start_idx]
    after = words[end_idx:min(len(words), end_idx + window_size)]
    return before + after

def extract_features(words, pos_dict, matched_phrases, phrase, idx, length):
    pos_tag_value = pos_dict.get(idx, 'misc')
    
    window_words = get_window_words(words, idx, length)
    window_lower = [w.lower() for w in window_words]
    
    window_variants = set()
    for w in window_lower:
        window_variants.update(normalize_word(w))
    
    good_variants = set()
    for w in GOOD_WORDS:
        good_variants.update(normalize_word(w))
    
    weak_variants = set()
    for w in WEAK_WORDS:
        weak_variants.update(normalize_word(w))
    
    blacklist_variants = set()
    for w in BLACKLISTED_WORDS:
        blacklist_variants.update(normalize_word(w))
    
    goodheadword = "yes" if window_variants & good_variants else "no"
    weakword = "yes" if window_variants & weak_variants else "no"
    blacklisted_word = "yes" if window_variants & blacklist_variants else "no"
    pattern_match = "yes" if any(w in matched_phrases for w in window_lower) else "no"

    is_eng_word = is_english_word(phrase)
    is_eng_acronym = is_english_acronym(phrase)
    is_bio_cond = is_bioconductor_name(phrase)
    is_bio_acro = is_bio_acronym(phrase)
    is_uppercase = 1 if phrase.isupper() else 0
    is_lowercase = 1 if phrase.islower() else 0
    is_mixedcase = 1 if (any(c.isupper() for c in phrase) and any(c.islower() for c in phrase)) else 0
    is_bmc_db_match = is_bmc_db(phrase)

    return {
        'word': phrase,
        'pos_tag': pos_tag_value,
        'goodheadword': goodheadword,
        'weakword': weakword,
        'blacklistedword': blacklisted_word,
        'pattern_match': pattern_match,
        'is_english_word': is_eng_word,
        'is_english_acronym': is_eng_acronym,
        'is_bioconductor_name': is_bio_cond,
        'is_bio_acronym': is_bio_acro,
        'is_uppercase': is_uppercase,
        'is_lowercase': is_lowercase,
        'is_mixedcase': is_mixedcase,
        'is_bmc_db': is_bmc_db_match
    }
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r"\w+")
def extract_features_for_all_words(html_file):
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    clean_text = soup.get_text(separator=' ')
    # print(clean_text)
    all_words = clean_text.split()
    all_words = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in all_words]
    all_words = [word for word in all_words if word]
    # tokenized_words = tokenizer.tokenize(clean_text)
    tokenized_words = word_tokenize(clean_text)
    pos_tags = pos_tag(tokenized_words)
    pos_dict = {i: tag[1] for i, tag in enumerate(pos_tags)}


    
    doc = nlp(clean_text)
    matched_phrases = set(span.text.lower() for _, start, end in matcher(doc) for span in [doc[start:end]])
    
    dbsoft_phrases = find_exact_tag_positions(soup, clean_text, all_words)
    
    processed_indices = set()
    features = []
    labels = []
    
    for phrase, start_idx, length in dbsoft_phrases:
        if start_idx not in processed_indices:
            feature = extract_features(all_words, pos_dict, matched_phrases, phrase, start_idx, length)
            #feature = extract_features(all_words, pos_dict, matched_token_indices, phrase, start_idx, length)
            features.append(feature)
            labels.append('dbsoft')
            for i in range(start_idx, start_idx + length):
                processed_indices.add(i)
    
    for i, word in enumerate(all_words):
        if i not in processed_indices:
            feature = extract_features(all_words, pos_dict, matched_phrases, word, i, 1)
            #feature = extract_features(all_words, pos_dict, matched_token_indices, word, i, 1)
            features.append(feature)
            labels.append('other')
            processed_indices.add(i)
    
    return features, labels
loaded_model = joblib.load("best_xgb_model.joblib")

import csv
def predict_and_evaluate(model, X_test, y_test, test_features, test_labels, label_encoder,fp_out):
    y_pred = model.predict(X_test)

    # Get the encoded label for 'dbsoft'
    dbsoft_label = label_encoder.transform(['dbsoft'])[0]

    precision = precision_score(y_test, y_pred, pos_label=dbsoft_label)
    recall = recall_score(y_test, y_pred, pos_label=dbsoft_label)
    f1 = f1_score(y_test, y_pred, pos_label=dbsoft_label)
    cm = confusion_matrix(y_test, y_pred)

    # print("\nLabel encoding:")
    # for i, cls in enumerate(label_encoder.classes_):
    #     print(f"  {cls}: {i}")

    # print("\nTest Results:")
    # print(f"Precision (for 'dbsoft'): {precision:.4f}")
    # print(f"Recall (for 'dbsoft'): {recall:.4f}")
    # print(f"F1 Score (for 'dbsoft'): {f1:.4f}")
    # print("Confusion Matrix:")
    # print(cm)

    # Decode numerical predictions to class names
    pred_labels = label_encoder.inverse_transform(y_pred)
    true_labels = label_encoder.inverse_transform(y_test)
    dbsoft_words = [f['word'] for f, pred in zip(test_features, pred_labels) if pred == 'dbsoft']
    other_words = [f['word'] for f, pred in zip(test_features, pred_labels) if pred == 'other']

    false_positives = [
        {
            'word': f['word']
        }
        for f, pred, true in zip(test_features, pred_labels, true_labels)
        if pred == 'dbsoft' and true == 'other'
    ]

    def save_csv(data, filename):
        if not data:
            print(f"No data to write to {filename}")
            return
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['word'])
            writer.writeheader()
            writer.writerows(data)
        print(f"✅ Saved {len(data)} words to {filename}")

    save_csv(false_positives, fp_out)
    # save_csv(true_positives, "tp_positives.csv")

    return f1

def evaluate_pmc_text(label_encoder, txt_file):
    def safe_bool(val):
        return int(bool(val)) if not isinstance(val, bool) else int(val)

    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按文章分割
    articles = content.split("=" * 100)

    all_test_features = []
    all_test_labels = []
    all_test_records = []

    for article in articles:
        # 提取 Full Text 字段
        match = re.search(r"Full Text:\s*(.+)", article, re.DOTALL)
        if not match:
            continue
        full_text = match.group(1).strip()

        # 清理 HTML 等
        soup = BeautifulSoup(full_text, 'html.parser')
        clean_text = soup.get_text(separator=' ')
        if len(clean_text) < 50:
            continue  # 跳过太短的内容

        all_words = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in clean_text.split() if word]
        tokenized_words = word_tokenize(clean_text)
        pos_tags = pos_tag(tokenized_words)
        pos_dict = {i: tag[1] for i, tag in enumerate(pos_tags)}

        try:
            with nlp.disable_pipes("parser", "ner"):
                doc = nlp(clean_text)
        except Exception as e:
            print(f"❗ spaCy failed on article (length: {len(clean_text)}): {e}")
            continue

        matched_phrases = set(
            span.text.lower() for _, start, end in matcher(doc) for span in [doc[start:end]]
        )

        processed_indices = set()
        for i, word in enumerate(all_words):
            if i not in processed_indices:
                feature = extract_features(all_words, pos_dict, matched_phrases, word, i, 1)
                all_test_features.append(feature)
                all_test_labels.append("other")  # assume unknown
                processed_indices.add(i)

    # 整合为 DataFrame
    test_data_with_labels = [
        {**f, 'label': lbl} for f, lbl in zip(all_test_features, all_test_labels)
    ]
    test_df = pd.DataFrame(test_data_with_labels)

    # # 过滤掉太短或太长的词，去重
    test_df = test_df[test_df['word'].apply(lambda x: 1 <= len(x) <= 10)]
    # test_df = test_df.drop_duplicates(subset='word', keep='first')

    print(f"📄 Total tokens retained after filtering: {len(test_df)}")

    # 特征构建
    binary_features = [[
        1 if f['goodheadword'] == 'yes' else 0,
        1 if f['weakword'] == 'yes' else 0,
        1 if f['blacklistedword'] == 'yes' else 0,
        f['is_english_word'],
        f['is_english_acronym'],
        f['is_bioconductor_name'],
        f['is_bio_acronym'],
        1 if len(f['word'].split()) > 1 else 0,
        f['is_uppercase'],
        f['is_lowercase'],
        f['is_mixedcase'],
        safe_bool(f['is_bmc_db'])
    ] for _, f in test_df.iterrows()]

    X_test = csr_matrix(binary_features)
    label_encoder.fit(["dbsoft", "other"])
    y_test = label_encoder.transform(test_df['label'].tolist())

    return X_test, y_test, test_df, all_test_labels
# label_encoder = LabelEncoder()
# X_test2, y_test2, test_df, test_labels2 = evaluate_pmc_text( label_encoder,"PCB_articles.txt")
# predict_and_evaluate(loaded_model, X_test2, y_test2, test_df.to_dict(orient='records'), test_df['label'].tolist(), label_encoder,"test.csv")
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_csv>")
        print("Example: python script.py PCB_articles.txt test.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Load trained model
    loaded_model = joblib.load("best_xgb_model.joblib")

    # Encode label
    label_encoder = LabelEncoder()

    # Run inference pipeline on new PMC articles
    X_test2, y_test2, test_df, test_labels2 = evaluate_pmc_text(label_encoder, input_file)

    # Predict and evaluate
    predict_and_evaluate(
        loaded_model,
        X_test2,
        y_test2,
        test_df.to_dict(orient='records'),
        test_df['label'].tolist(),
        label_encoder,
        output_file
    )

if __name__ == "__main__":
    main()
