FROM python:3.8

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /home

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN python3 -c "import nltk; nltk.download('punkt')"

COPY src .

CMD python bot.py
