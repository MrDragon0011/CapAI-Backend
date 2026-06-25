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

`POST /analyze` and `GET /health` — see the main repo README for the API contract.