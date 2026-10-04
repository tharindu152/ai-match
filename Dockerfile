FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/model

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY lecturer_matcher_rf.joblib /model/
COPY encoders/ /model/encoders/

EXPOSE 5000

CMD ["python", "app.py"]
