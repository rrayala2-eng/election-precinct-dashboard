# Election Precinct Dashboard

## 1. Run the backend

```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

Keep this terminal running.

## 2. Run the frontend

Open a **new terminal**:

```
cd backend\frontend
python -m http.server 5500
```

Keep this terminal running too.

## 3. Open the dashboard

Go to `http://localhost:5500` in your browser.

Pick an office, district, year, and election type, then click **Load results**.