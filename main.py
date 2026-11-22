"""
AI-Based Legal Case Classification and Advocate Recommendation System
MTech Internship Project - Academy of Skill Development (AI/ML & Data Science)
Pipeline: load data -> EDA -> text cleaning -> TF-IDF -> model comparison (5-fold CV) -> tuning
          -> test evaluation -> explainability (top words) -> save -> recommender demo
Run: python generate_data.py   (once)   then   python main.py
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             confusion_matrix, classification_report)
from core import clean_text, recommend_advocates, predict_category, MODEL_PATH

warnings.filterwarnings("ignore")
SEED, OUT = 42, "outputs"
os.makedirs(OUT, exist_ok=True); os.makedirs("models", exist_ok=True)
sns.set_style("whitegrid")

# ---------------------------------------------------------------- 1. DATA
df = pd.read_csv("data/legal_cases.csv")
df["clean"] = df["description"].apply(clean_text)
df["words"] = df["clean"].str.split().str.len()
print("=" * 70); print("1. DATASET OVERVIEW (synthetic)"); print("=" * 70)
print(f"Samples: {len(df)} | Categories: {df.category.nunique()} | Missing: {int(df.isnull().sum().sum())}")
print(f"Average words per case: {df.words.mean():.1f}")
print(df.category.value_counts().to_string())

plt.figure(figsize=(9, 4.5))
sns.countplot(y="category", data=df, order=df.category.value_counts().index, palette="viridis")
plt.title("Cases per Legal Category"); plt.tight_layout(); plt.savefig(f"{OUT}/01_category_distribution.png", dpi=150); plt.close()
plt.figure(figsize=(7, 4)); sns.histplot(df.words, bins=25, color="teal")
plt.title("Case Description Length (words)"); plt.tight_layout(); plt.savefig(f"{OUT}/02_text_length.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 2. SPLIT
X_train, X_test, y_train, y_test = train_test_split(df.clean, df.category, test_size=0.2,
                                                    stratify=df.category, random_state=SEED)
print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

# ---------------------------------------------------------------- 3. MODEL COMPARISON
def pipe(clf):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)), ("clf", clf)])

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, random_state=SEED),
    "Linear SVM (calibrated)": CalibratedClassifierCV(LinearSVC(random_state=SEED), cv=3),
    "Multinomial NB": MultinomialNB(),
    "Complement NB": ComplementNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
print("\n" + "=" * 70); print("2. MODEL COMPARISON (5-fold CV, training set)"); print("=" * 70)
for n, m in models.items():
    acc = cross_val_score(pipe(m), X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    f1 = cross_val_score(pipe(m), X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    rows.append([n, acc.mean(), acc.std(), f1.mean()])
cvdf = pd.DataFrame(rows, columns=["Model", "CV Accuracy", "CV Acc Std", "CV Macro-F1"]).sort_values("CV Macro-F1", ascending=False).reset_index(drop=True)
print(cvdf.round(4).to_string(index=False)); cvdf.round(4).to_csv(f"{OUT}/model_comparison_cv.csv", index=False)
plt.figure(figsize=(8, 4)); sns.barplot(x="CV Macro-F1", y="Model", data=cvdf, palette="magma")
plt.xlim(cvdf["CV Macro-F1"].min() - 0.03, 1.0); plt.title("Cross-Validated Macro-F1"); plt.tight_layout()
plt.savefig(f"{OUT}/03_model_comparison.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 4. TUNING
print("\n" + "=" * 70); print("3. HYPERPARAMETER TUNING (GridSearchCV)"); print("=" * 70)
grids = {
    "Logistic Regression": (LogisticRegression(max_iter=3000, random_state=SEED), {"clf__C": [1, 5, 10, 50], "tfidf__ngram_range": [(1, 1), (1, 2)]}),
    "Linear SVM (calibrated)": (CalibratedClassifierCV(LinearSVC(random_state=SEED), cv=3), {"clf__estimator__C": [0.1, 0.5, 1, 5], "tfidf__ngram_range": [(1, 1), (1, 2)]}),
}
tuned = {}
for n, (est, g) in grids.items():
    gs = GridSearchCV(pipe(est), g, cv=cv, scoring="f1_macro", n_jobs=-1).fit(X_train, y_train)
    tuned[n] = gs.best_estimator_
    print(f"{n}: best CV macro-F1 = {gs.best_score_:.4f} | {gs.best_params_}")

# ---------------------------------------------------------------- 5. TEST EVALUATION
print("\n" + "=" * 70); print("4. FINAL EVALUATION ON UNSEEN TEST SET"); print("=" * 70)
res = []
for n, m in tuned.items():
    p = m.predict(X_test)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, p, average="macro")
    res.append([n, accuracy_score(y_test, p), pr, rc, f1])
resdf = pd.DataFrame(res, columns=["Model", "Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"]).sort_values("Macro-F1", ascending=False).reset_index(drop=True)
print(resdf.round(4).to_string(index=False)); resdf.round(4).to_csv(f"{OUT}/final_test_results.csv", index=False)
best_name = resdf.loc[0, "Model"]; best = tuned[best_name]; pred = best.predict(X_test)
print(f"\nBEST MODEL: {best_name}\n"); print(classification_report(y_test, pred))

labels = sorted(df.category.unique())
cm = confusion_matrix(y_test, pred, labels=labels)
plt.figure(figsize=(9, 7)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix - {best_name}"); plt.xticks(rotation=40, ha="right")
plt.tight_layout(); plt.savefig(f"{OUT}/04_confusion_matrix.png", dpi=150); plt.close()
pf = pd.Series(f1_score(y_test, pred, average=None, labels=labels), index=labels).sort_values()
plt.figure(figsize=(8, 4.5)); pf.plot(kind="barh", color="seagreen"); plt.xlim(0.5, 1.0)
plt.title("Per-Category F1-score"); plt.tight_layout(); plt.savefig(f"{OUT}/05_per_class_f1.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 6. EXPLAINABILITY (top words, Logistic Regression)
lr = tuned["Logistic Regression"]
vec, clf = lr.named_steps["tfidf"], lr.named_steps["clf"]
fn = np.array(vec.get_feature_names_out())
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
tw = {}
for ax, (i, c) in zip(axes.ravel(), enumerate(clf.classes_)):
    idx = np.argsort(clf.coef_[i])[-8:]
    ax.barh(fn[idx], clf.coef_[i][idx], color="steelblue"); ax.set_title(c, fontsize=10)
    tw[c] = ", ".join(fn[np.argsort(clf.coef_[i])[::-1][:8]])
plt.suptitle("Most Influential Words per Category (Logistic Regression)"); plt.tight_layout()
plt.savefig(f"{OUT}/06_top_words.png", dpi=140); plt.close()
pd.Series(tw, name="top_words").to_csv(f"{OUT}/top_words_per_category.csv")
print("Top words per category:"); [print(f"  {k}: {v}") for k, v in tw.items()]

# ---------------------------------------------------------------- 7. SAVE (refit best on all data for deployment)
final = pipe(best.named_steps["clf"]).set_params(**{k: v for k, v in best.get_params().items()
                                                    if k.startswith("tfidf__")})
final.fit(df.clean, df.category)
joblib.dump(final, MODEL_PATH); print(f"\nSaved model -> {MODEL_PATH}")

# ---------------------------------------------------------------- 8. END-TO-END DEMO
print("\n" + "=" * 70); print("5. END-TO-END DEMO: case text -> category -> advocates"); print("=" * 70)
demo = [("My landlord is not returning my security deposit of Rs 50,000 after I vacated the flat.", "Bengaluru", "Kannada"),
        ("Someone hacked my Instagram account and is blackmailing me with morphed photos.", "Mumbai", "Hindi"),
        ("I received a legal notice under Section 138 for a bounced cheque.", "Delhi", "English"),
        ("My employer terminated me without notice and has not paid my salary for three months.", "Pune", "Marathi")]
out = []
for text, city, lang in demo:
    cat, p = predict_category(text, 1)[0]
    recs = recommend_advocates(cat, city, lang, 3)
    print(f"\nCase: {text}\n -> Category: {cat} (confidence {p*100:.1f}%) | client city={city}, language={lang}")
    print(recs[["advocate_id", "name", "city", "experience_years", "rating", "match_score"]].to_string(index=False))
    for _, r in recs.iterrows():
        out.append(dict(case=text, predicted_category=cat, confidence=round(p, 3), advocate_id=r.advocate_id,
                        advocate=r["name"], city=r.city, match_score=r.match_score))
pd.DataFrame(out).to_csv(f"{OUT}/sample_recommendations.csv", index=False)
print("\nDone. Graphs/tables are in 'outputs'. Now run:  streamlit run app.py")
