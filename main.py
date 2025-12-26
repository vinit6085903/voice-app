import os, csv, uuid, json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from passlib.context import CryptContext
import smtplib
from email.message import EmailMessage

# Voice
import torchaudio
import librosa
import soundfile as sf
from speechbrain.inference import EncoderClassifier

# Gemini
import google.generativeai as genai

# ================= ENV ==================
load_dotenv()

# ---------- GEMINI ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing")

genai.configure(api_key=GEMINI_API_KEY)

# ---------- EMAIL ----------
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# ================= APP ==================
app = FastAPI(title="Secure Voice Intelligence System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= EMAIL ==================
def send_token_email(to_email: str, token: str):
    msg = EmailMessage()
    msg["From"] = EMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = "Your Login Token"

    msg.set_content(f"""
Hello,

Your login token:
{token}

Do not share this token.

Secure Voice Intelligence System
""")

    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)

# ================= SECURITY ==================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(p: str):
    return pwd_context.hash(p[:72])

def verify_password(p: str, h: str):
    return pwd_context.verify(p[:72], h)

def generate_token():
    return uuid.uuid4().hex

# ================= FILES ==================
FILES = {
    "users": "users.csv",
    "voices": "voices.csv",
    "records": "records.csv",
    "reminders": "reminders.csv"
}

HEADERS = {
    "users": ["id","name","email","password","token","created_at"],
    "voices": ["user_id","embedding","created_at"],
    "records": ["id","user","department","transcript","summary","timestamp"],
    "reminders": ["id","user","text","remind_at","created_at"]
}

for k, f in FILES.items():
    if not os.path.exists(f):
        with open(f, "w", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow(HEADERS[k])

# ================= HELPERS ==================
def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def append_csv(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def verify_token(token: str):
    user = next((u for u in read_csv(FILES["users"]) if u["token"] == token), None)
    if not user:
        raise HTTPException(401, "Login required")
    return user

def is_future_time(remind_at: str):
    try:
        return datetime.strptime(remind_at, "%Y-%m-%d %H:%M") > datetime.now()
    except:
        return False

def is_duplicate_reminder(user, text, time):
    return any(
        r["user"] == user and r["text"] == text and r["remind_at"] == time
        for r in read_csv(FILES["reminders"])
    )

# ================= GEMINI ==================
def speech_to_text(audio_bytes: bytes, filename: str):
    ext = filename.lower().split(".")[-1]
    mime = {
        "wav":"audio/wav",
        "mp3":"audio/mpeg",
        "mp4":"audio/mp4",
        "ogg":"audio/ogg"
    }.get(ext)

    if not mime:
        raise HTTPException(400, "Unsupported audio format")

    model = genai.GenerativeModel("gemini-2.5-flash")
    return model.generate_content(
        ["Transcribe in ENGLISH", {"mime_type": mime, "data": audio_bytes}]
    ).text or ""

def get_summary(text: str):
    return genai.GenerativeModel("gemini-2.5-flash") \
        .generate_content("Summarize briefly:\n" + text).text or ""

def extract_reminder(text: str):
    try:
        raw = genai.GenerativeModel("gemini-2.5-flash").generate_content(f"""
Detect reminder.
Return JSON only:
{{"found":true,"remind_at":"YYYY-MM-DD HH:MM","message":""}}
Text:{text}
""").text
        return json.loads(raw.replace("```",""))
    except:
        return {"found": False}

# ================= VOICE ==================
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

def get_embedding(audio_bytes):
    path = f"temp_{uuid.uuid4()}.wav"
    with open(path, "wb") as f:
        f.write(audio_bytes)

    y,_ = librosa.load(path, sr=16000, mono=True)
    sf.write(path, librosa.util.normalize(y), 16000)

    signal,_ = torchaudio.load(path)
    os.remove(path)

    return classifier.encode_batch(signal).squeeze().cpu().numpy()

# ================= ROUTES ==================
@app.post("/auth/register")
async def register(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    voice: UploadFile = File(...)
):
    if any(u["email"] == email for u in read_csv(FILES["users"])):
        raise HTTPException(409, "User exists")

    uid = uuid.uuid4().hex
    audio = await voice.read()

    append_csv(FILES["users"], [
        uid, name, email, hash_password(password), "", datetime.now().isoformat()
    ])

    emb = get_embedding(audio) if voice.filename.endswith(".wav") else None
    append_csv(FILES["voices"], [
        uid, json.dumps(emb.tolist() if emb is not None else None),
        datetime.now().isoformat()
    ])

    return {"status": "registered"}

@app.post("/auth/login")
async def login(email: str = Form(...), password: str = Form(...)):
    users = read_csv(FILES["users"])
    user = next((u for u in users if u["email"] == email), None)

    if not user or not verify_password(password, user["password"]):
        raise HTTPException(401, "Invalid credentials")

    token = generate_token()
    for u in users:
        if u["email"] == email:
            u["token"] = token

    with open(FILES["users"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADERS["users"])
        for u in users:
            w.writerow([u[h] for h in HEADERS["users"]])

    send_token_email(email, token)
    return {"token": token}

@app.post("/business/upload")
async def business_upload(
    token: str = Form(...),
    department: str = Form(...),
    audio: UploadFile = File(...)
):
    user = verify_token(token)
    audio_bytes = await audio.read()

    # 1️⃣ Speech → Text
    transcript = speech_to_text(audio_bytes, audio.filename)

    # 2️⃣ Summary
    summary = get_summary(transcript)

    # 3️⃣ Reminder detection
    reminder = extract_reminder(transcript)

    # 4️⃣ Save record
    append_csv(FILES["records"], [
        uuid.uuid4().hex,
        user["email"],
        department,
        transcript,
        summary,
        datetime.now().isoformat()
    ])

    # 5️⃣ Save reminder (if valid)
    if reminder.get("found"):
        msg = reminder.get("message")
        time = reminder.get("remind_at")

        if is_future_time(time) and not is_duplicate_reminder(user["email"], msg, time):
            append_csv(FILES["reminders"], [
                uuid.uuid4().hex,
                user["email"],
                msg,
                time,
                datetime.now().isoformat()
            ])

    return {
        "status": "processed",
        "transcript": transcript,
        "summary": summary,
        "reminder": reminder
    }
@app.get("/")
def root():
    return {"status": "running"}
