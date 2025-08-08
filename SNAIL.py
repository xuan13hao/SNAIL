import os
import sys
import csv
import re
import joblib
import requests
import nltk

from bs4 import BeautifulSoup
from scipy.sparse import csr_matrix
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import spacy
from spacy.matcher import Matcher
import enchant

# ----------------------------
# NLTK setup (single block)
# ----------------------------
DOWNLOAD_DIR = "./nltk_data"
nltk.data.path.append(DOWNLOAD_DIR)
nltk.download('punkt', download_dir=DOWNLOAD_DIR)
nltk.download('averaged_perceptron_tagger', download_dir=DOWNLOAD_DIR)
nltk.download('wordnet', download_dir=DOWNLOAD_DIR)
nltk.download('omw-1.4', download_dir=DOWNLOAD_DIR)

# ----------------------------
# Constants / dictionaries
# ----------------------------
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

# ----------------------------
# spaCy setup + patterns
# ----------------------------
patterns = [
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'such'}, {'LOWER': 'as'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'LOWER': 'such'}, {'POS': 'NOUN'}, {'LOWER': 'as'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'other'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'include'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'especially'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'any'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'some'}, {'LOWER': 'other'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'be'}, {'LOWER': 'a'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'like'}, {'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}],
    [{'LOWER': 'such'}, {'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'as'}, {'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'like'}, {'LOWER': 'other'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'one'}, {'LOWER': 'of'}, {'LOWER': 'the'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'one'}, {'LOWER': 'of'}, {'LOWER': 'these'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'one'}, {'LOWER': 'of'}, {'LOWER': 'those'}, {'POS': 'NOUN'}],
    [{'LOWER': 'example'}, {'LOWER': 'of'}, {'POS': 'NOUN'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'be'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'be'}, {'LOWER': 'example'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'for'}, {'LOWER': 'example'}, {'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'be'}, {'LOWER': 'call'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'be'}, {'LOWER': 'name'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'mainly'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'mostly'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'notably'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'particularly'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'principally'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'in'}, {'LOWER': 'particular'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'except'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'other'}, {'LOWER': 'than'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'e.g.'}, {'LOWER': ',', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': '(', 'OP': '?'}, {'LOWER': 'e.g.'}, {'LOWER': ',', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': '.', 'OP': '?'}, {'LOWER': ')'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'i.e.'}, {'LOWER': ',', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'a'}, {'LOWER': 'kind'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'kind'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'form'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'look'}, {'LOWER': 'like'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'sound'}, {'LOWER': 'like'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'be'}, {'LOWER': 'similar'}, {'LOWER': 'to'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'example'}, {'LOWER': 'of'}, {'LOWER': 'this'}, {'LOWER': 'be'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'type'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'POS': 'NOUN'}, {'LOWER': 'type'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'whether'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'LOWER': 'compare'}, {'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'with'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'compare'}, {'LOWER': 'to'}, {'POS': 'NOUN', 'OP': '+'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'for'}, {'LOWER': 'instance'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}, {'LOWER': 'and', 'OP': '?'}, {'LOWER': 'or', 'OP': '?'}, {'LOWER': 'sort'}, {'LOWER': 'of'}, {'POS': 'NOUN'}],
    [{'POS': 'NOUN', 'OP': '?'}, {'LOWER': ',', 'OP': '?'}, {'LOWER': 'which'}, {'LOWER': 'may'}, {'LOWER': 'include'}, {'POS': 'NOUN', 'OP': '+'}]
]

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000_000
nlp.disable_pipes("ner", "parser")

matcher = Matcher(nlp.vocab)
for idx, pattern in enumerate(patterns):
    matcher.add(f"PATTERN_{idx}", [pattern])

print(f"Total matcher rules: {len(matcher)}")

# ----------------------------
# BMC regex dictionary
# ----------------------------
bmc_dict_file = "dbsoftDB.tsv"

def parse_regex_pattern(raw_pattern: str):
    raw_pattern = raw_pattern.strip()
    # Remove outer () if they wrap the entire pattern and compile OK
    if raw_pattern.startswith('(') and raw_pattern.endswith(')'):
        try:
            re.compile(raw_pattern)
            raw_pattern = raw_pattern[1:-1]
        except re.error:
            pass

    sub_patterns, current, paren_count = [], "", 0
    for ch in raw_pattern:
        if ch == '(':
            paren_count += 1
        elif ch == ')':
            paren_count -= 1
        elif ch == '|' and paren_count == 0:
            if current:
                sub_patterns.append(current)
                current = ""
            continue
        current += ch
    if current:
        sub_patterns.append(current)

    valid = []
    for sub in (s.strip() for s in sub_patterns):
        if not sub:
            continue
        test_sub = f"({sub})" if not (sub.startswith('(') and sub.endswith(')')) else sub
        try:
            re.compile(test_sub, re.IGNORECASE)
            valid.append(test_sub)
        except re.error as e:
            print(f"Warning: Invalid sub-pattern '{sub}': {e}")
    return valid

bmc_patterns = []
try:
    with open(bmc_dict_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                _, pattern = line.strip().split("\t")
                for sub_pattern in parse_regex_pattern(pattern):
                    try:
                        bmc_patterns.append(re.compile(sub_pattern, re.IGNORECASE))
                    except re.error as e:
                        print(f"Warning: Failed to compile sub-pattern '{sub_pattern}': {e}")
            except ValueError as e:
                print(f"Warning: Skipping malformed line in {bmc_dict_file}: {line.strip()} ({e})")
    print(f"Loaded {len(bmc_patterns)} SNAIL regex patterns")
except FileNotFoundError:
    print(f"Warning: {bmc_dict_file} not found, is_bmc_db will be 0")
except UnicodeDecodeError as e:
    print(f"Error: Failed to decode {bmc_dict_file}: {e}, is_bmc_db will be 0")

if not bmc_patterns:
    fallback_patterns = [
        r"AAindex", r"AmiGO", r"ArkDB", r"ADAPT", r"ASAP", r"ALIGN",
        r"Amino\s+Acid\s+Index\s+[Dd]ata[Bb]ase",
        r"Annotation\s+[Dd]ata[Bb]ase\s+for\s+Affymetrix\s+Probesets\s+and\s+Transcripts",
        r"The\s+Ark"
    ]
    for pat in fallback_patterns:
        try:
            bmc_patterns.append(re.compile(pat, re.IGNORECASE))
        except re.error as e:
            print(f"Warning: Failed to compile fallback pattern '{pat}': {e}")
    print(f"Using {len(bmc_patterns)} fallback SNAIL patterns")

def is_bmc_db(phrase: str) -> int:
    if not bmc_patterns:
        return 0
    for pattern in bmc_patterns:
        if pattern.fullmatch(phrase):
            return 1
    return 0

# ----------------------------
# Acronyms (SCOWL + web)
# ----------------------------
scowl_dir = "scowl"
acronyms = set()
if os.path.isdir(scowl_dir):
    for filename in os.listdir(scowl_dir):
        if filename.startswith("english-abbreviations"):
            with open(os.path.join(scowl_dir, filename), "r", encoding="latin1") as f:
                acronyms.update(line.strip().upper() for line in f if line.strip())

url = "https://www.allacronyms.com/A/abbreviations"
try:
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')
    all_acronyms = [item.text.strip() for item in soup.select("a[href*='/abbreviation/']") if item.text.isupper()]
    acronyms.update(all_acronyms)
except requests.RequestException as e:
    print(f"Warning: Failed to fetch acronyms from {url}: {e}")

def is_english_acronym(word: str) -> int:
    return 1 if word.upper() in acronyms else 0

# ----------------------------
# Bioconductor names
# ----------------------------
bioconductor_file = "bioconductor_packages.txt"
try:
    with open(bioconductor_file, "r", encoding="utf-8") as f:
        bioconductor_names = set(line.strip().lower() for line in f if line.strip())
except FileNotFoundError:
    print(f"Warning: {bioconductor_file} not found, using default set")
    bioconductor_names = {'limma', 'deseq2', 'edger', 'biomart'}

def is_bioconductor_name(word: str) -> int:
    return 1 if word.lower() in bioconductor_names else 0

# ----------------------------
# Bio acronyms file
# ----------------------------
acronyms_file = "acronyms.dic"

def load_bio_acronyms(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if lines and lines[0].isdigit():
                acronym_lines = lines[1:]
            else:
                acronym_lines = lines
            acrs = set(line.upper() for line in acronym_lines if line)
            if not acrs:
                raise ValueError(f"No valid acronyms found in {file_path}")
            return acrs
    except (FileNotFoundError, ValueError, UnicodeDecodeError) as e:
        print(f"Warning: Error loading {file_path}: {e}, using default set")
        return {'BLAST', 'PDB', 'NCBI', 'DBSNP'}

bio_acronyms = load_bio_acronyms(acronyms_file)

def is_bio_acronym(word: str) -> int:
    return 1 if word.upper() in bio_acronyms else 0

# ----------------------------
# Feature helpers
# ----------------------------
lemmatizer = WordNetLemmatizer()
en_dict = enchant.Dict("en_US")

def is_english_word(word: str) -> int:
    if not word or word.strip() == "":
        return 0
    return 1 if en_dict.check(word.lower()) else 0

def normalize_word(word: str):
    variants = [
        word.lower(),
        word.lower().replace("-", ""),
        word.lower().replace("-", " "),
        lemmatizer.lemmatize(word.lower()),
        lemmatizer.lemmatize(word.lower().replace("-", "")),
        lemmatizer.lemmatize(word.lower().replace("-", " ")),
    ]
    return set(variants)

def find_exact_tag_positions(soup: BeautifulSoup, full_text: str, words):
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

    def variants_of(token_set):
        out = set()
        for w in token_set:
            out.update(normalize_word(w))
        return out

    good_variants = variants_of(GOOD_WORDS)
    weak_variants = variants_of(WEAK_WORDS)
    blacklist_variants = variants_of(BLACKLISTED_WORDS)

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

# ----------------------------
# Core pipeline
# ----------------------------
def evaluate_pmc_text(label_encoder: LabelEncoder, txt_file: str):
    def safe_bool(val):
        return int(bool(val)) if not isinstance(val, bool) else int(val)

    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    articles = content.split("=" * 100)

    all_test_features = []
    all_test_labels = []

    for article in articles:
        match = re.search(r"Full Text:\s*(.+)", article, re.DOTALL)
        if not match:
            continue
        full_text = match.group(1).strip()

        soup = BeautifulSoup(full_text, 'html.parser')
        clean_text = soup.get_text(separator=' ')
        if len(clean_text) < 50:
            continue

        all_words = [re.sub(r'[^a-zA-Z0-9]', '', w) for w in clean_text.split() if w]
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
            if i in processed_indices:
                continue
            feature = extract_features(all_words, pos_dict, matched_phrases, word, i, 1)
            all_test_features.append(feature)
            all_test_labels.append("other")  # unknown => default to "other"
            processed_indices.add(i)

    test_data_with_labels = [{**f, 'label': lbl} for f, lbl in zip(all_test_features, all_test_labels)]
    test_df = __import__("pandas").DataFrame(test_data_with_labels)

    # Keep words of reasonable length
    test_df = test_df[test_df['word'].apply(lambda x: 1 <= len(x) <= 15)]
    # Binary features
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

def predict_and_evaluate(model, X_test, y_test, test_features_df, test_labels, label_encoder, fp_out: str):
    y_pred = model.predict(X_test)

    dbsoft_label = label_encoder.transform(['dbsoft'])[0]
    precision = precision_score(y_test, y_pred, pos_label=dbsoft_label)
    recall = recall_score(y_test, y_pred, pos_label=dbsoft_label)
    f1 = f1_score(y_test, y_pred, pos_label=dbsoft_label)
    cm = confusion_matrix(y_test, y_pred)

    # Decode predictions
    pred_labels = label_encoder.inverse_transform(y_pred)
    true_labels = label_encoder.inverse_transform(y_test)

    false_positives = [
        {'word': f['word']}
        for f, pred, true in zip(test_features_df.to_dict(orient='records'), pred_labels, true_labels)
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
    return f1

# ----------------------------
# Entry point
# ----------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_csv>")
        print("Example: python script.py PCB_articles.txt test.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    loaded_model = joblib.load("best_xgb_model.joblib")
    label_encoder = LabelEncoder()

    X_test, y_test, test_df, test_labels = evaluate_pmc_text(label_encoder, input_file)

    predict_and_evaluate(
        loaded_model,
        X_test,
        y_test,
        test_df,
        test_labels,
        label_encoder,
        output_file
    )

if __name__ == "__main__":
    main()
