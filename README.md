# 🐌 SNAIL_XGBoost

**SNAIL** (Software NAme Identification from Literature) is a high-performance tool designed to identify software and database mentions (dbsoft) in biomedical literature.  
It combines rule-based linguistic features, pattern matching, and a fine-tuned XGBoost model to achieve robust classification—even under extreme class imbalance.

> **SNAIL: Software Name Identification from Literature towards Automated Bioinformatic Resources Collection and Categorization**


---

## 📦 Features

- Hybrid approach combining:
  - spaCy pattern matching
  - regex-based dictionary (BMC patterns)
  - windowed linguistic signal extraction (e.g., headwords, acronyms)
- Classifies words/phrases as `dbsoft` vs `other`
- Trained using SMOTE oversampling and 10-fold cross-validation
- Outputs predictions and false positives to CSV
- Supports bulk evaluation on raw PMC-style text inputs

---

## 🔧 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

```bash
SNAIL_XGBoost/
│
├── SNAIL.py                     # Main pipeline script
├── best_xgb_model.joblib        # Trained XGBoost model
├── requirements.txt             # Python dependencies
├── PCB_articles.txt             # Example input file
├── bioconductor_packages.txt    # Known Bioconductor package names
├── linnaeus.regex.dic.BMCPaper.tsv  # Regex patterns from BMC dictionary
├── acronyms.dic                 # Custom bio-related acronyms
├── scowl/                       # SCOWL dictionary files
│   └── english-words.*          # Word lists
│   └── english-abbreviations.* # Acronym lists

```bash
python SNAIL.py <input_text_file> <output_csv_file>
python SNAIL.py PCB_articles.txt test.csv