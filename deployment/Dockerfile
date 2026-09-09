FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY src ./src
COPY reports ./reports

COPY data ./data

COPY model_v1.pkl .
COPY preprocessor_v1.pkl .
COPY model_v1_metadata.json .
COPY WA_Fn-UseC_-Telco-Customer-Churn.csv .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]