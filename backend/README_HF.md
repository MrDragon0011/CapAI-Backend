---
title: CapAI Backend
emoji: 🤽
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# CapAI Backend (Hugging Face Space)

FastAPI service for water polo biomechanical analysis: 33-point MediaPipe Pose
skeleton, per-frame joint kinematics, and a trained YOLOv8 ball detector.

This Space is the deployment target. When pushing here, place the **contents of
the `backend/` folder at the repo root** (Dockerfile, main.py, requirements.txt,
ball_detector.pt) and rename this file to `README.md`.

`POST /analyze` and `GET /health` — see the main repo README for the API contract.
