FROM python:3.12-slim-bookworm

WORKDIR /app

# Оновлюємо системні пакети Debian, щоб закрити системні вразливості
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]