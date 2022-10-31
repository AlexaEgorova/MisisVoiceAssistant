"""Simple voice assistant helping on questions about NUST MISIS"""
import re
import time
import logging
from pathlib import Path
from typing import List, Tuple

# Telegram API
# https://docs.python-telegram-bot.org/en/v20.0a4/#installing
from telegram import (
    Update,
    Bot
)
from telegram.ext import (
    MessageHandler,
    Filters,
    Updater,
    CommandHandler,
    CallbackContext
)

# Tokenizer
# https://www.nltk.org/install.html
import nltk
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

# Ru tokenizer
nltk.download('punkt')


QUESTIONS: Dict[str, str] = {}
ANSWERS: Dict[str, str] = {}

def chatter(words: List[str]) -> Tuple[str, str, float]:
    """Map a question to a list of answers and return the result.

    Args:
        words (List[str]): Tokenized question.

    Returns:
        Tuple[str, str, float]: Question, Answer, Score.
    """
        max_score = 0.0
    total_score = 0.0
    best_key = None
    for key, value in QUESTIONS.items():
        score = 0.0
        for word in words:
            if word in value:
                score += 1
            else:
                score -= 0.5
        if score > max_score:
            max_score = score
            _sc = max_score / len(value)
            if _sc > 0.3:
                best_key = key
                total_score = _sc

    q = "Unrecognized"
    a = "Unrecognized"
    if best_key is not None:
        a = ANSWERS[best_key]
        q = " ".join(QUESTIONS[best_key])
    return q, a, total_score


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


def converter(from_path: Path, to_path: Path) -> bool:
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
    orig.export(
        str(to_path),
        format=to_path.suffix.replace(".", "")
    )
    return True


def prepare_text(text: str) -> str:
    """Formatting for Telegram messages"""
    return text\
        .replace("-", r"\-") \
        .replace("+", r"\+") \
        .replace(".", r"\.")\
        .replace("(", r"\(")\
        .replace(")", r"\)")\
        .replace("[", r"\[")\
        .replace("]", r"\]")\
        .replace("´", r"")


class VOABot:
    """Voice Bot."""

    def __init__(self, token: str, tmp_dir: str):
        """Create voice bot."""
        self.name = "live-tg-bot"

        self.updater = Updater(
            token=token,
            use_context=True
        )
        self.bot = Bot(
            token=token
        )

        self._tmp_dir = Path(tmp_dir).resolve()
        self._tmp_dir.parent.mkdir(exist_ok=True)

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
        new_path = path.with_suffix(".wav")  # WAV for recognition
        converted = converter(
            path,
            new_path
        )
        if not converted:
            return None

        line, words = recognizer(str(new_path))
        question, answer, score = chatter(words)
        out_path = new_path.parent / f"a_{new_path.name}"
        out_path = talker(answer, out_path)
        new_out_path = out_path.with_suffix(".ogg")  # voice messages are in OGG
        converted = converter(
            out_path,
            new_out_path
        )
        if not converted:
            return None

        if not new_out_path.exists():
            return None

        msg = (
            f"`recognized`: _{prepare_text(line.lower())}_\n"
            f"`question`: _{prepare_text(question)}_\n"
            f"`score`: `{str(round(score, 3))}`\n"
            f"`answer`: _{prepare_text(answer)}_"
        )
        context.bot.send_message(
            chat_id=chat_id,
            text=msg,
            parse_mode="MarkdownV2"
        )
        context.bot.send_voice(
            chat_id=chat_id,
            voice=new_out_path.open("rb")
        )

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

        dispatcher.add_handler(start_handler)
        dispatcher.add_handler(voice_handler)

        self.bot.send_message(
            chat_id="87701872",
            text="Bot is up and running : ]"
        )

        self.updater.start_polling()
        self.updater.idle()


if __name__ == "__main__":
    token = "5742923020:AAEkGi6m06F4hb4p-KaLrMD_K-UFhBnG5b8"
    tmp_dir = "D://tmp"

    bot = VOABot(
        token=token,
        tmp_dir=tmp_dir,
    )
    bot.start()
