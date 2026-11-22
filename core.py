"""Core logic shared by main.py and app.py: classifier, advocate recommender, case database."""
import re, sqlite3, datetime
import joblib
import pandas as pd

MODEL_PATH = "models/case_classifier.joblib"
ADV_PATH = "data/advocates.csv"
DB_PATH = "data/cases.db"
STATUSES = ["Assigned", "Accepted", "Under Review", "Hearing Scheduled", "Closed"]

def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"rs\.?\s?[\d,\.]+(\s?(lakh|crore))?", " amount ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ------------------------------------------------------------ classifier
_model = None
def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_category(description: str, top_k: int = 3):
    """Returns list of (category, probability) sorted best first."""
    m = load_model()
    proba = m.predict_proba([clean_text(description)])[0]
    order = proba.argsort()[::-1][:top_k]
    return [(m.classes_[i], float(proba[i])) for i in order]

# ------------------------------------------------------------ recommender
def load_advocates() -> pd.DataFrame:
    return pd.read_csv(ADV_PATH, keep_default_na=False)

def recommend_advocates(category: str, city: str = "", language: str = "English", top_n: int = 3) -> pd.DataFrame:
    """Score = 50 specialization + 15 experience + 15 rating + 10 city + 10 language (max 100)."""
    adv = load_advocates()
    def score(r):
        s = 0.0
        if r.primary_specialization == category:
            s += 50
        elif r.secondary_specialization == category:
            s += 30
        s += 15 * min(r.experience_years, 20) / 20
        s += 15 * (r.rating / 5)
        if city and r.city.lower() == city.lower():
            s += 10
        if language and language in r.languages.split(";"):
            s += 10
        return round(s, 1)
    adv["match_score"] = adv.apply(score, axis=1)
    adv = adv[(adv.primary_specialization == category) | (adv.secondary_specialization == category)]
    return adv.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)

# ------------------------------------------------------------ database
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS cases(
            case_id INTEGER PRIMARY KEY AUTOINCREMENT, client_name TEXT, city TEXT, language TEXT,
            description TEXT, category TEXT, confidence REAL, advocate_id TEXT, advocate_name TEXT,
            status TEXT, created TEXT, updated TEXT)""")

def add_case(client_name, city, language, description, category, confidence, advocate_id=None, advocate_name=None):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    status = "Assigned" if advocate_id else "Open (No advocate yet)"
    with _conn() as c:
        cur = c.execute("""INSERT INTO cases(client_name,city,language,description,category,confidence,
                           advocate_id,advocate_name,status,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (client_name, city, language, description, category, confidence,
                         advocate_id, advocate_name, status, now, now))
        return cur.lastrowid

def get_cases(client_name=None, advocate_id=None, unassigned_category=None) -> pd.DataFrame:
    q, p = "SELECT * FROM cases WHERE 1=1", []
    if client_name:
        q += " AND lower(client_name)=lower(?)"; p.append(client_name)
    if advocate_id:
        q += " AND advocate_id=?"; p.append(advocate_id)
    if unassigned_category:
        q += " AND advocate_id IS NULL AND category=?"; p.append(unassigned_category)
    with _conn() as c:
        return pd.read_sql_query(q + " ORDER BY case_id DESC", c, params=p)

def update_status(case_id, status):
    with _conn() as c:
        c.execute("UPDATE cases SET status=?, updated=? WHERE case_id=?",
                  (status, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), case_id))

def assign_case(case_id, advocate_id, advocate_name):
    with _conn() as c:
        c.execute("UPDATE cases SET advocate_id=?, advocate_name=?, status='Assigned', updated=? WHERE case_id=?",
                  (advocate_id, advocate_name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), case_id))

def seed_demo_cases():
    """Adds a few sample cases the first time the app runs so the dashboard is not empty."""
    init_db()
    if len(get_cases()) > 0:
        return
    demo = [("Demo Client", "Bengaluru", "English", "My landlord is not returning my security deposit of Rs 50,000 after I vacated the flat."),
            ("Demo Client", "Bengaluru", "Hindi", "I lost Rs 75,000 in an online UPI fraud after clicking a fake link."),
            ("Asha", "Mumbai", "English", "The washing machine I ordered online stopped working within a week and the seller refuses refund."),
            ("Ravi", "Hyderabad", "Telugu", "A cheque of Rs 2 lakh given to me by my business partner bounced due to insufficient funds.")]
    for name, city, lang, desc in demo:
        cat, p = predict_category(desc, 1)[0]
        adv = recommend_advocates(cat, city, lang, 1).iloc[0]
        cid = add_case(name, city, lang, desc, cat, p, adv.advocate_id, adv["name"])
    update_status(1, "Under Review"); update_status(2, "Accepted"); update_status(4, "Hearing Scheduled")


# ------------------------------------------------------------ voice input
VOICE_LANGS = {"English": "en-IN", "Hindi": "hi-IN", "Kannada": "kn-IN", "Tamil": "ta-IN",
               "Telugu": "te-IN", "Marathi": "mr-IN", "Bengali": "bn-IN", "Urdu": "ur-IN"}

def transcribe_audio(wav_bytes: bytes, language: str = "English"):
    """Speech-to-text using Google's free web recognizer (needs internet). Returns (text, error)."""
    import io
    import speech_recognition as sr
    r = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as src:
            audio = r.record(src)
        return r.recognize_google(audio, language=VOICE_LANGS.get(language, "en-IN")), None
    except sr.UnknownValueError:
        return None, "Could not understand the audio. Please speak clearly and try again."
    except sr.RequestError:
        return None, "Speech service unreachable. Check your internet connection."
    except Exception as e:
        return None, f"Audio error: {e}"
