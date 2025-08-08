# 🐌 SNAIL_XGBoost

**SNAIL** (Software NAme Identification from Literature) is a high-performance tool designed to identify software and database mentions (dbsoft) in biomedical literature.  
It combines rule-based linguistic features, pattern matching, and a fine-tuned XGBoost model to achieve robust classification—even under extreme class imbalance.

> **Citation**:  
> *SNAIL: Software Name Identification from Literature towards Automated Bioinformatic Resources Collection and Categorization*
---

## 📦 Features
- Hybrid approach combining:
  - spaCy pattern matching
  - windowed linguistic signal extraction (e.g., headwords, acronyms)
- Classifies words/phrases as `dbsoft` vs `other`
- Outputs predictions to CSV
---

## 🔧 Requirements
- Python 3.8 or newer
- pip & setuptools
- Dependencies (installed automatically):
  - `beautifulsoup4`
  - `pandas`
  - `scikit-learn`
  - `scipy`
  - `nltk`
  - `spacy`
  - `pyenchant`
  - `requests`
  - `joblib`
- **One-time spaCy model download**:
```bash
python -m spacy download en_core_web_sm
git clone https://github.com/xuan13hao/SNAIL.git
cd SNAIL
pip install -e .
snail-xgb <input_txt_file> <output_csv_file>


