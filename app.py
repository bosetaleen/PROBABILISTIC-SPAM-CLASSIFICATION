from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download nltk data if needed
nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()
templates = Jinja2Templates(directory="templates")

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english')]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# Load your existing model and vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Home page (GET)
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "spam_proba": None}
    )

# Prediction (POST)
@app.post("/", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    message = form.get("message")

    # Preprocess and vectorize
    transformed = transform_text(message)
    vector = tfidf.transform([transformed])

    # Predict probabilities
    spam_prob = model.predict_proba(vector)[0][1]  # Probability spam

    spam_keywords = ["loan", "win", "prize", "offer", "buy now", "urgent", "limited time","discount"]

    # Check if any spam keyword exists
    if any(word in message.lower() for word in spam_keywords):
        spam_prob_boosted = max(spam_prob, 0.7)  # force at least 70% spam
    else:
        spam_prob_boosted = spam_prob


    # Final probabilities
    spam_proba = int(spam_prob_boosted * 100)
    ham_proba = 100 - spam_proba

    # Verdict
    if spam_proba > 40:
        verdict = "Likely Spam"
    elif spam_proba > 30:
        verdict = "Borderline"
    else:
        verdict = "Likely Not Spam"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "message": message,
            "spam_proba": spam_proba,
            "ham_proba": ham_proba,
            "verdict": verdict
        }
    )
