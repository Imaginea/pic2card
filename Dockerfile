FROM python:3.6.9-slim-buster

RUN pip install --upgrade pip && \
    apt-get update && \
    apt-get install -y --no-install-recommends libsm6 tesseract-ocr && \
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/*

# Cache the requirement.txt
ADD requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

# app addition
ADD . /app

EXPOSE 5050

ENTRYPOINT ["python", "-m", "app.api"]
