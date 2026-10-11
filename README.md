# EcoSort AI - Waste Classification System

**Stack:** PyTorch (MobileNetV2 transfer learning) - FastAPI - SQLite - HTML/JS + Chart.js

## Run
1. `pip install -r requirements.txt`
2. Download TrashNet (cardboard, glass, metal, paper, plastic, trash) into `dataset/<class>/*.jpg`
3. `python backend/train.py`   -> creates `backend/model.pt` and `backend/classes.json`
4. `uvicorn backend.main:app --reload`
5. Open http://127.0.0.1:8000  (API docs at /docs)

## API
- `POST /api/predict` - image upload -> class, confidence, top-3, bin, tip
- `GET /api/stats` - totals, per-class counts, daily counts
- `GET /api/history` - recent predictions

## Architecture
Browser -> FastAPI -> MobileNetV2 inference -> SQLite log -> dashboard charts
