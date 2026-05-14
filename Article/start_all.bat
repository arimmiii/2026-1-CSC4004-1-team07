@echo off
start cmd /k "python -m uvicorn article_mainapi:app --reload --host 0.0.0.0"
start cmd /k "python article_BE.py"