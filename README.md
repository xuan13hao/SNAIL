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
python SNAIL.py <input_text_file> <output_csv_file>
