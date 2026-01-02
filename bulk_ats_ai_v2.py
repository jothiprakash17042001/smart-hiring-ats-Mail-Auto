import os
import re
import nltk
import PyPDF2
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from sentence_transformers import SentenceTransformer, util

# ---------------- SETUP ----------------
nltk.download("stopwords")
from nltk.corpus import stopwords
STOPWORDS = stopwords.words("english")

model = SentenceTransformer("all-MiniLM-L6-v2")

RESUME_FOLDER = "resumes"
GOOGLE_SHEET_NAME = "ATS Resume Results"
WORKSHEET_NAME = "Sheet1"
CREDENTIALS_FILE = "google_creds.json"

# ---------------- TEXT UTILS ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9@.+ ]", " ", text)
    return " ".join(w for w in text.split() if w not in STOPWORDS)

def extract_pdf(path):
    reader = PyPDF2.PdfReader(path)
    return " ".join(page.extract_text() or "" for page in reader.pages)

# ---------------- AI NAME EXTRACTION (NO FILENAME) ----------------
def extract_name_ai(text):
    """
    AI-style name extractor using confidence scoring.
    Only resume content is used.
    """
    blacklist = [
        "resume", "curriculum", "vitae", "profile",
        "engineer", "developer", "scientist", "analyst",
        "email", "phone", "mobile", "linkedin", "github",
        "summary", "objective", "experience"
    ]

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    candidates = []

    for line in lines[:15]:
        score = 0
        words = line.split()

        # Positive signals
        if 2 <= len(words) <= 4:
            score += 2
        if line.isupper() or line.istitle():
            score += 2

        # Negative signals
        if any(b in line.lower() for b in blacklist):
            score -= 4
        if re.search(r"\d", line):
            score -= 2
        if len(line) > 40:
            score -= 2

        candidates.append((line.title(), score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    best_name, best_score = candidates[0]

    return best_name if best_score >= 2 else "Not Found"

# ---------------- AI EMAIL AGENT ----------------
def extract_email_ai(text):
    text_lower = text.lower()
    pattern = r"[a-z0-9._%+-]+@[a-z0-9.-]+\.(com|in|org|net)\b"
    matches = [m.group(0) for m in re.finditer(pattern, text_lower)]

    scored = []
    for email in matches:
        score = 0
        user = email.split("@")[0]

        if len(user) >= 4:
            score += 2
        if user.isdigit():
            score -= 3
        if any(x in email for x in ["linkedin", "github", "resume", "profile"]):
            score -= 4

        scored.append((email, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored and scored[0][1] >= 1 else "Not Found"

# ---------------- PHONE ----------------
def extract_phone(text):
    m = re.search(r"(\+91[\s-]?\d{10}|\b\d{10}\b)", text)
    return m.group(0) if m else "Not Found"

# ---------------- AI SEMANTIC SCORING ----------------
def semantic_similarity(resume, jd):
    r_emb = model.encode(resume, convert_to_tensor=True)
    j_emb = model.encode(jd, convert_to_tensor=True)
    return float(util.cos_sim(r_emb, j_emb)[0][0])

def experience_score(text):
    signals = ["experience", "internship", "project", "deployed", "developed"]
    count = sum(text.count(s) for s in signals)
    return 1 if count >= 8 else 0.6 if count >= 4 else 0.3 if count >= 2 else 0

# ---------------- GOOGLE SHEET ----------------
def get_sheet():
    creds = Credentials.from_service_account_file(
        CREDENTIALS_FILE,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    client = gspread.authorize(creds)
    sheet = client.open(GOOGLE_SHEET_NAME).worksheet(WORKSHEET_NAME)

    if not sheet.row_values(1):
        sheet.append_row(
            ["Candidate_Name", "Email", "Phone_Number", "Final_ATS_Score", "Decision"]
        )

    return sheet

# ---------------- LOAD JD ----------------
with open("job_description.txt", encoding="utf-8") as f:
    jd_clean = clean_text(f.read())

sheet = get_sheet()
existing_emails = set(sheet.col_values(2)[1:])

results = []

# ---------------- PROCESS ALL RESUMES ----------------
for file in sorted(os.listdir(RESUME_FOLDER)):
    if not file.lower().endswith(".pdf"):
        continue

    text = extract_pdf(os.path.join(RESUME_FOLDER, file))

    email = extract_email_ai(text)
    if email == "Not Found" or email in existing_emails:
        continue

    name = extract_name_ai(text)
    phone = extract_phone(text)

    resume_clean = clean_text(text)

    sem_sim = semantic_similarity(resume_clean, jd_clean)
    exp = experience_score(resume_clean)

    final_score = round((sem_sim * 70) + (exp * 30), 2)
    decision = "SELECTED" if final_score >= 60 and sem_sim >= 0.45 else "REJECTED"

    sheet.append_row([name, email, phone, final_score, decision])
    existing_emails.add(email)

    results.append(
        {
            "Candidate_Name": name,
            "Email": email,
            "Phone_Number": phone,
            "Final_ATS_Score": final_score,
            "Decision": decision,
        }
    )

print("✅ ATS AI AGENT v3 Completed Successfully")
print(pd.DataFrame(results))
# ---------------- END OF SCRIPT ----------------