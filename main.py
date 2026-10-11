"""EcoSort AI backend v2 - login, per-user analytics, CSV export.
Run: uvicorn backend.main:app --reload"""
import csv, hashlib, io, json, secrets, sqlite3, time, torch, torch.nn as nn
from pathlib import Path
from PIL import Image
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torchvision import models, transforms as T

BASE = Path(__file__).parent
CLASSES = json.load(open(BASE / "classes.json"))
model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(1280, len(CLASSES))
model.load_state_dict(torch.load(BASE / "model.pt", map_location="cpu"))
model.eval()
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([.485, .456, .406], [.229, .224, .225])])

INFO = {  # label -> (bin, recyclable, tip)
    "cardboard": ("Blue bin", True, "Flatten boxes and keep them dry."),
    "glass": ("Green bin", True, "Rinse and remove lids. Don't mix with ceramics."),
    "metal": ("Yellow bin", True, "Rinse cans; crush if possible to save space."),
    "paper": ("Blue bin", True, "Keep clean and dry. Greasy paper is not recyclable."),
    "plastic": ("Red bin", True, "Rinse, check the resin code, and remove caps."),
    "trash": ("Black bin", False, "Not recyclable - dispose with general waste."),
}
CO2 = {"cardboard": .6, "glass": .3, "metal": 3.0, "paper": .9, "plastic": 1.2}  # illustrative kg CO2e per item

db = sqlite3.connect(BASE / "waste.db", check_same_thread=False)
db.executescript("""
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, salt TEXT, pw TEXT);
CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id INTEGER, ts REAL);
CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY, user_id INTEGER, ts REAL, label TEXT, confidence REAL, filename TEXT);
""")
app = FastAPI(title="EcoSort AI")

class Auth(BaseModel):
    email: str
    password: str
    name: str = ""

def pw_hash(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 100_000).hex()

def new_session(uid):
    tok = secrets.token_hex(24)
    db.execute("INSERT INTO sessions VALUES(?,?,?)", (tok, uid, time.time())); db.commit()
    return tok

def current_user(authorization: str = Header(None)):
    tok = (authorization or "").removeprefix("Bearer ")
    r = db.execute("SELECT u.id,u.name FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=?", (tok,)).fetchone()
    if not r:
        raise HTTPException(401, "Please log in.")
    return {"id": r[0], "name": r[1]}

@app.post("/api/register")
def register(a: Auth):
    email = a.email.strip().lower()
    if "@" not in email or len(a.password) < 6:
        raise HTTPException(400, "Enter a valid email and a password of 6+ characters.")
    if db.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
        raise HTTPException(409, "That email is already registered.")
    salt = secrets.token_hex(16)
    cur = db.execute("INSERT INTO users(name,email,salt,pw) VALUES(?,?,?,?)",
                     (a.name.strip() or email.split("@")[0], email, salt, pw_hash(a.password, salt)))
    db.commit()
    return {"token": new_session(cur.lastrowid)}

@app.post("/api/login")
def login(a: Auth):
    r = db.execute("SELECT id,salt,pw FROM users WHERE email=?", (a.email.strip().lower(),)).fetchone()
    if not r or not secrets.compare_digest(r[2], pw_hash(a.password, r[1])):
        raise HTTPException(401, "Wrong email or password.")
    return {"token": new_session(r[0])}

@app.get("/api/me")
def me(u=Depends(current_user)):
    return u

@app.post("/api/predict")
async def predict(file: UploadFile = File(...), u=Depends(current_user)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Please upload an image file (JPG or PNG).")
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "That image could not be read.")
    with torch.no_grad():
        probs = torch.softmax(model(tf(img).unsqueeze(0)), 1)[0]
    conf, idx = probs.topk(min(3, len(CLASSES)))
    top = [{"label": CLASSES[i], "confidence": round(c.item(), 4)} for c, i in zip(conf, idx)]
    label = top[0]["label"]
    db.execute("INSERT INTO scans(user_id,ts,label,confidence,filename) VALUES(?,?,?,?,?)",
               (u["id"], time.time(), label, top[0]["confidence"], file.filename))
    db.commit()
    bin_, recyc, tip = INFO.get(label, ("General bin", False, "Check local guidelines."))
    return {"label": label, "confidence": top[0]["confidence"], "top3": top,
            "bin": bin_, "recyclable": recyc, "tip": tip}

@app.get("/api/stats")
def stats(u=Depends(current_user)):
    q = lambda s: db.execute(s, (u["id"],)).fetchall()
    by = dict(q("SELECT label,COUNT(*) FROM scans WHERE user_id=? GROUP BY label"))
    return {"total": sum(by.values()), "by_class": by,
            "avg_conf": q("SELECT AVG(confidence) FROM scans WHERE user_id=?")[0][0] or 0,
            "co2_kg": round(sum(CO2.get(k, 0) * v for k, v in by.items()), 1),
            "by_day": q("SELECT date(ts,'unixepoch','localtime'),COUNT(*) FROM scans WHERE user_id=? GROUP BY 1 ORDER BY 1 DESC LIMIT 14")[::-1]}

@app.get("/api/history")
def history(u=Depends(current_user)):
    rows = db.execute("SELECT ts,label,confidence,filename FROM scans WHERE user_id=? ORDER BY id DESC LIMIT 50", (u["id"],)).fetchall()
    return [{"time": time.strftime("%d %b %H:%M", time.localtime(r[0])), "label": r[1],
             "confidence": r[2], "file": r[3]} for r in rows]

@app.get("/api/export")
def export(u=Depends(current_user)):
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(["time", "file", "category", "confidence"])
    for r in db.execute("SELECT ts,filename,label,confidence FROM scans WHERE user_id=? ORDER BY id DESC", (u["id"],)):
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[0])), r[1], r[2], round(r[3], 4)])
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=ecosort_history.csv"})

app.mount("/", StaticFiles(directory=BASE.parent / "frontend", html=True), name="ui")
