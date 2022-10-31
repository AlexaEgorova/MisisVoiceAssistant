FROM python:3.8


WORKDIR /home

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY src .

RUN python3 -c "import nltk; nltk.download('punkt')"

CMD python bot.py
