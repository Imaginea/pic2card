FROM python:3.6.9-slim-buster

RUN pip install --upgrade pip && \
    apt-get update && apt-get install -y libsm6 tesseract-ocr && \
    apt-get clean

WORKDIR /mystique_app

COPY . .

RUN pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

EXPOSE 5050

CMD ["python", "/mystique_app/app/api.py"]
