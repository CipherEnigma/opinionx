
import re, string, pickle

def load_models(base_path):
    tfidf = pickle.load(open(f"{base_path}/tfidf_vectorizer.pkl", "rb"))
    lr    = pickle.load(open(f"{base_path}/logistic_regression.pkl", "rb"))
    svm   = pickle.load(open(f"{base_path}/linear_svm.pkl",          "rb"))
    return tfidf, lr, svm

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def trust_score(text):
    words = text.split()
    score = 1.0
    if len(words) < 5:   score -= 0.30
    if len(words) < 15:  score -= 0.10
    caps = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps > 0.5:       score -= 0.20
    uniq = len(set(words)) / max(len(words), 1)
    if uniq < 0.4:       score -= 0.15
    if text.count("!") > 5: score -= 0.10
    return round(max(0.0, min(1.0, score)), 2)

def predict(text, tfidf, svm, lr, model_name="svm"):
    clean = clean_text(text)
    vec   = tfidf.transform([clean])
    if model_name == "svm":
        pred       = svm.predict(vec)[0]
        conf_score = svm.decision_function(vec)[0]
        confidence = min(abs(conf_score) / 3, 1.0)
    else:
        pred       = lr.predict(vec)[0]
        confidence = float(max(lr.predict_proba(vec)[0]))
    return {
        "sentiment":   "Positive" if pred == 1 else "Negative",
        "label":       int(pred),
        "confidence":  round(confidence, 3),
        "trust_score": trust_score(text),
        "word_count":  len(text.split())
    }
