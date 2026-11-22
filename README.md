# AI-Based Legal Case Classification and Advocate Recommendation System Using NLP and Machine Learning
**MTech Internship Project - Academy of Skill Development (AI/ML & Data Science)**

> Educational prototype. Uses a **synthetic dataset** and **fictional advocates**. It does not provide legal advice.

## 1. Problem Statement
Many people cannot describe their problem in legal terms or do not know which advocate to approach.
This system takes a case described in plain language, identifies the legal category with NLP and machine
learning, recommends the best advocates, and lets clients and advocates track cases on a dashboard.

## 2. Modules
1. **Case Understanding (NLP + ML):** text cleaning, TF-IDF (unigrams + bigrams), supervised classification into 8 categories:
   Consumer Protection, Property & Tenancy, Family & Matrimonial, Criminal, Labour & Employment,
   Cyber Crime, Motor Accident, Banking & Cheque Bounce.
2. **Advocate Recommendation:** content-based weighted scoring (max 100):
   50 specialization match (30 if secondary) + 15 experience + 15 rating + 10 same city + 10 language match.
3. **Voice input:** the client can speak the case (8 Indian languages); speech-to-text fills the case box, which can be edited before analysis.
4. **Dashboard (Streamlit + SQLite):** client case filing, case tracking, advocate dashboard (update status,
   accept open cases) and a model-results page.

## 3. Dataset (synthetic)
`generate_data.py` creates 2,400 case descriptions (300 per category) from templates with random slot filling,
lower-casing, generic filler sentences, and 20% "confusing" sentences borrowed from other categories to make the
task realistic. It also creates 60 fictional advocate profiles (specialization, experience, rating, city, languages).
Limitation: results on real legal text would be lower. Real data and a transformer model (BERT / IndicBERT) are future work.

## 4. Results (random_state = 42, 80/20 split, 5-fold CV)
| Model (tuned) | Accuracy | Macro-F1 |
|---|---|---|
| **Linear SVM (calibrated)** (best) | **0.9042** | **0.9042** |
| Logistic Regression | 0.9000 | 0.9000 |

Cross-validation of all 5 models gave macro-F1 between 0.891 and 0.896 (Multinomial NB, Random Forest, Complement NB,
Linear SVM, Logistic Regression). Strongest categories: Labour & Employment (F1 0.95) and Motor Accident (0.92);
hardest: Criminal (0.87) and Consumer Protection (0.88). The top influential words are meaningful
(e.g. "cheque, bounced, loan" for Banking; "salary, employer" for Labour; "accident, insurance" for Motor Accident).

## 5. Setup and Run (Windows / Linux / Mac)
```bash
# 1. Install Python 3.9+ (python.org). On Windows tick "Add Python to PATH".
# 2. Open a terminal inside this project folder
python -m venv venv
# Windows:      venv\Scripts\activate
# Linux / Mac:  source venv/bin/activate
pip install -r requirements.txt

python generate_data.py      # 3. creates data/legal_cases.csv and data/advocates.csv
python main.py               # 4. trains, evaluates, saves model, creates graphs in outputs/ (1-2 minutes)
streamlit run app.py         # 5. opens the dashboard at http://localhost:8501
python test_app.py           # optional: automated test of the app
```

## 6. How to demo the dashboard
1. **Client: New Case** tab: enter name `Asha`, city, language. Either click the microphone, speak, and press *Convert voice to text*, or type a case such as
   *"My landlord is not returning my security deposit of Rs 50,000 after I vacated the flat."* then click **Analyse my case**.
   The category, confidence and top 3 advocates are shown and the case is assigned.
2. **Client: Track My Cases** tab: type the same name to see the status (`Demo Client` has sample cases).
3. **Advocate Dashboard** tab: pick the assigned advocate and change the case status (Accepted, Under Review,
   Hearing Scheduled, Closed). Untick auto-assign while filing to create an Open case, then accept it as an advocate.
4. **Model Results** tab: metrics, confusion matrix, per-category F1, top words.

## 7. Folder Structure
```
lawzo_project/
|-- generate_data.py     # synthetic data
|-- core.py              # text cleaning, classifier loader, recommender, SQLite case database
|-- main.py              # full ML pipeline + end-to-end demo
|-- app.py               # Streamlit dashboard
|-- test_app.py          # automated app test
|-- requirements.txt, README.md
|-- data/                # legal_cases.csv, advocates.csv, cases.db (created on first app run)
|-- models/case_classifier.joblib
`-- outputs/             # graphs (PNG) and result tables (CSV)
```

## 8. Future Scope
Real anonymised legal data, IndicBERT/BERT for Indian languages, voice input, collaborative filtering from client
feedback, authentication, real advocate onboarding, notifications.

## 9. Voice Input Notes
Uses Streamlit `st.audio_input` (browser microphone) and the `SpeechRecognition` library (Google web speech recognizer). It needs an internet connection and microphone permission in the browser (use Chrome or Edge, and allow the mic when asked). Choose the language in the dropdown before recording. Always check the converted text before analysing.
