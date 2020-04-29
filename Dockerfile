FROM python:3.6.9

RUN python --version

RUN pip install --upgrade pip

WORKDIR /mystique_app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5050

CMD ["python", "/mystique_app/app/api.py"]