FROM python:3.10-slim

WORKDIR /App

COPY . /App

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "App.py"]
