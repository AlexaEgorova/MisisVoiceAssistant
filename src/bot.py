"""Simple voice assistant helping on questions about NUST MISIS."""
from datetime import datetime
import re
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pytz

from pydantic import BaseModel
from pydantic.fields import Field
from pydantic.env_settings import BaseSettings

from pymongo import MongoClient
from bson.objectid import ObjectId

from deeppavlov import build_model, train_model, Chainer
from deeppavlov.core.common.file import read_json

# Telegram API
# https://docs.python-telegram-bot.org/en/v20.0a4/#installing
from telegram import (
    Update,
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    MessageHandler,
    Filters,
    Updater,
    CommandHandler,
    CallbackContext,
    CallbackQueryHandler
)

# Tokenizer
# https://www.nltk.org/install.html
# import nltk
from nltk.tokenize import word_tokenize

# Voice recognition
# https://github.com/Uberi/speech_recognition#installing
import speech_recognition as sr

# Google text-to-speech
# https://gtts.readthedocs.io/en/latest/#installation
from gtts import gTTS

# Audio convertion
# https://github.com/jiaaro/pydub#installation
from pydub import AudioSegment

# # Ru tokenizer
# nltk.download('punkt')

ADMINS = [
    "87701872",
    "499825068"
]


class EnvSettings(BaseSettings):
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


class MongoDBParams(EnvSettings):
    host: str = Field(..., env="MONGO_HOST")
    port: int = Field(..., env="MONGO_PORT")
    username: str = Field(..., env="MONGO_USER")
    password: str = Field(..., env="MONGO_PASSWORD")
    db: str = 'misis-voa'


class BotConfig(EnvSettings):

    token: str = Field(..., env="BOT_TOKEN")
    tmp_dir: str = Field(..., env="TMP_DIR")
    db: MongoDBParams = MongoDBParams()


class ResultRec(BaseModel):

    id: Optional[str] = Field(None, alias="_id")

    recognized: str
    question: str
    score: float
    answer: str

    user_score: Optional[int] = Field(None)

    chat_id: str
    created_at: datetime

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True

    def to_msg(self):
        return (
            f"`recognized`: _{prepare_text(self.recognized.lower())}_\n"
            f"`question`: _{prepare_text(self.question)}_\n"
            f"`score`: `{str(round(self.score, 3))}`\n"
            f"`answer`: _{prepare_text(self.answer)}_"
        )



def chatter(
    predictor: Chainer,
    question: str
) -> Tuple[str, str, float]:
    """Map a question to a list of answers and return the result.

    Args:
        words (List[str]): Tokenized question.

    Returns:
        Tuple[str, str, float]: Question, Answer, Score.
    """
    resp = predictor([question])

    answer = resp[0][0]
    total_score = resp[1][0]

    if not total_score:
        answer = "Извините, не совсем поняла ваш вопрос."

    return (
        question,
        answer,
        total_score
    )


def recognizer(audio_path: str, language: str = "ru") -> Tuple[str, List[str]]:
    """Speech to text.

    Args:
        audio_path (str): Path to a WAV-file with the audio.
        language (str): The audio language.

    Returns:
        Tuple[str, List[str]]: Recognized text, tokenized text.
    """
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as src:
        audio = r.record(src)
    orig_line = r.recognize_google(
        audio,
        language=language,
    )

    if not isinstance(orig_line, str):
        raise ValueError("KEK")
    orig_line = orig_line.lower()

    # remove mean words
    line = re.sub(r'\w\*{4}', '', orig_line)

    words = word_tokenize(line)
    return orig_line, words


def talker(text: str, path: Path, language: str = "ru") -> Path:
    """Text to speech.

    Args:
        text (str): Text to transform.
        audio_path (str): Where to save MP3-file.
        language (str): The audio language.

    Returns:
        Path: Audio path.
    """
    myobj = gTTS(
        text=text,
        lang=language,
        slow=False
    )
    path = path.with_suffix(".mp3")
    myobj.save(path)
    return path


def converter(
    from_path: Path,
    to_path: Path,
    silence_ms: int = 0,
) -> bool:
    """Audio covertion.

    Args:
        from_path (Path): Input file path.
        to_path (Path): Path to audio in dst format.

    Returns:
        bool: If succeed.
    """
    readers = {
        ".ogg": AudioSegment.from_ogg,
        ".wav": AudioSegment.from_wav,
        ".mp3": AudioSegment.from_mp3,
    }

    suffix = from_path.suffix
    reader = readers.get(suffix, None)
    if reader is None:
        return False

    orig = reader(str(from_path))
    if silence_ms:
        silenced_segment = AudioSegment.silent(duration=silence_ms)
        orig = orig + silenced_segment

    orig.export(
        str(to_path),
        format=to_path.suffix.replace(".", "")
    )
    return True


def prepare_text(text: str) -> str:
    """Format text for Telegram messages."""
    return text\
        .replace("-", r"\-") \
        .replace("+", r"\+") \
        .replace(".", r"\.")\
        .replace("(", r"\(")\
        .replace(")", r"\)")\
        .replace("[", r"\[")\
        .replace("]", r"\]")\
        .replace("´", r"")


def get_keyboard(oid: str) -> InlineKeyboardMarkup:
    """Send a message with two inline buttons attached."""
    keyboard = [
        [
            InlineKeyboardButton("❌", callback_data=f"{oid}_0"),
            InlineKeyboardButton("✅", callback_data=f"{oid}_1"),
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    return reply_markup


class VOABot:
    """Voice Bot."""

    def __init__(self, config: BotConfig):
        """Create voice bot."""
        self.name = "live-tg-bot"

        self.config = config
        self.updater = Updater(
            token=config.token,
            use_context=True
        )
        self.bot = Bot(
            token=config.token
        )

        self._db: Optional[MongoClient] = None

        self._predictor: Optional[Chainer] = None

        self._tmp_dir = Path(config.tmp_dir).resolve()
        self._tmp_dir.parent.mkdir(exist_ok=True, parents=True)

    def _get_mongo_client(self) -> MongoClient:
        """Return mongodb connection."""
        if self._db:
            return self._db
        url = "mongodb://{username}:{password}@{host}:{port}".format(
            username=self.config.db.username,
            password=self.config.db.password,
            host=self.config.db.host,
            port=self.config.db.port,
        )
        self._db = MongoClient(
            host=url,
            tz_aware=True,
            tzinfo=pytz.utc
        )
        return self._db

    def _tg_callback_start(self, update: Update, context: CallbackContext):
        """/start."""
        if not update.effective_chat:
            return

        chat_id = update.effective_chat.id

        msg = f"/start by {chat_id}"
        logging.info(msg)

        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=(
                r"Successfully started"
            ),
            parse_mode="MarkdownV2"
        )

    def _tg_callback_voice(self, update: Update, context: CallbackContext):
        """Voice command."""
        chat = update.effective_chat
        if not chat:
            return
        logging.warning("/voice by %s", chat.id)

        chat_id = chat.id

        message = update.message
        if not message or not message.voice:
            print("No voice message obtained")
            return

        file_info = self.bot.get_file(message.voice.file_id)
        print(file_info)

        path = self._tmp_dir / str(
            chat_id
        ).replace("-", "_") / f"{round(time.time())}.ogg"
        path.parent.mkdir(exist_ok=True, parents=True)
        file_info.download(str(path))
        msg = "Audio is being processed"
        context.bot.send_message(
            chat_id=chat.id,
            text=msg,
            parse_mode="MarkdownV2"
        )
        self._tg_handle_voice(
            chat_id=chat.id,
            path=path,
            context=context,
        )
        print("done")

    def _tg_handle_voice(
        self,
        chat_id: int,
        path: Path,
        context: CallbackContext
    ) -> None:
        """Voice processing."""
        self._predictor = self._init_model()
        new_path = path.with_suffix(".wav")  # WAV for recognition
        converted = converter(
            path,
            new_path,
            silence_ms=1000
        )
        if not converted:
            return None

        line, words = recognizer(str(new_path))
        question, answer, score = chatter(
            self._predictor,
            " ".join(words)
        )
        out_path = new_path.parent / f"a_{new_path.name}"
        out_path = talker(answer, out_path)
        new_out_path = out_path.with_suffix(".ogg")  # voice msgs are in OGG
        converted = converter(
            out_path,
            new_out_path
        )
        if not converted:
            return None

        if not new_out_path.exists():
            return None

        db = self._get_mongo_client()[self.config.db.db]

        rec = ResultRec(
            recognized=line,
            question=question,
            score=score,
            answer=answer,
            chat_id=chat_id,
            created_at=datetime.now(pytz.utc)
        )
        msg = rec.to_msg()

        result = db["live_recs"].insert_one(
            rec.dict(exclude={"id": True})
        )
        oid = result.inserted_id

        context.bot.send_message(
            chat_id=chat_id,
            text=msg,
            parse_mode="MarkdownV2",
            reply_markup=get_keyboard(oid)
        )
        msg += f"\n`chat_id`: {chat_id}"

        for admin in ADMINS:
            if chat_id == int(admin):
                continue
            self.bot.send_message(
                chat_id=int(admin),
                text=msg,
                parse_mode="MarkdownV2"
            )
        context.bot.send_voice(
            chat_id=chat_id,
            voice=new_out_path.open("rb")
        )

    def _tg_callback_button(
        self,
        update: Update,
        context: CallbackContext
    ) -> None:
        """Parse the CallbackQuery and updates the message text."""
        query = update.callback_query
        assert query is not None
        assert query.data is not None

        db = self._get_mongo_client()[self.config.db.db]

        oid, user_score = query.data.split("_")
        print("HERE", oid, user_score)
        db["live_recs"].update_one(
            filter={"_id": ObjectId(oid)},
            update={"$set": {"user_score": int(user_score)}}
        )
        query.answer("Спасибо за оценку!")
        # await query.edit_message_text(text=f"Selected option: {query.data}")

    def _init_model(self) -> Chainer:
        if self._predictor is not None:
            return self._predictor
        model_config = read_json('./data/tfidf_logreg_autofaq_misis.json')
        try:
            self._predictor = build_model(model_config, load_trained=True)
        except FileNotFoundError:
            self._predictor = train_model(model_config, download=True)
        return self._predictor

    def start(self):
        """Start Bot."""
        dispatcher = self.updater.dispatcher

        start_handler = CommandHandler(
            'start',
            self._tg_callback_start
        )
        voice_handler = MessageHandler(
            Filters.voice,
            self._tg_callback_voice
        )
        button_handler = CallbackQueryHandler(
            self._tg_callback_button
        )

        dispatcher.add_handler(start_handler)
        dispatcher.add_handler(voice_handler)
        dispatcher.add_handler(button_handler)

        self._predictor = self._init_model()

        for admin in ADMINS:
            self.bot.send_message(
                chat_id=admin,
                text="🫡"
            )

        self.updater.start_polling()
        self.updater.idle()


if __name__ == "__main__":
    bot = VOABot(
        config=BotConfig()
    )
    bot.start()
