FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
```

**Step 4 — Update requirements.txt**
```
setuptools
flask==2.3.3
numpy==1.26.4
tensorflow==2.20.0
pillow==10.0.0
werkzeug==2.3.7
gunicorn==21.2.0