# LSTM Inference via MLflow + MinIO (FastAPI)

## 1) 설치
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

uvicorn main:app --reload --host 0.0.0.0 --port 8000


curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  --data-binary @sample_request.json


uvicorn callback_server:app --reload --port 9001
# 콜백 예:
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "X": [[... 12x18 ...]],
        "callback_url": "http://localhost:9001/result",
        "metadata": {"req_id":"abc-123"}
      }'

