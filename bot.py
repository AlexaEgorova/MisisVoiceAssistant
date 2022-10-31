"""Simple voice assistant helping on questions about NUST MISIS"""

def chatter(words: List[str]) -> Tuple[str, str, float]:
    """Map a question to a list of answers and return the result.

    Args:
        words (List[str]): Tokenized question.

    Returns:
        Tuple[str, str, float]: Question, Answer, Score.
    """
    pass


def recognizer(audio_path: str, language: str = "ru") -> Tuple[str, List[str]]:
    """Speech to text.

    Args:
        audio_path (str): Path to a WAV-file with the audio.
        language (str): The audio language.

    Returns:
        Tuple[str, List[str]]: Recognized text, tokenized text.
    """
    pass


def talker(text: str, path: Path, language: str = "ru") -> Path:
    """Text to speech.

    Args:
        text (str): Text to transform.
        audio_path (str): Where to save MP3-file.
        language (str): The audio language.

    Returns:
        Path: Audio path.
    """
    pass


def converter(from_path: Path, to_path: Path) -> bool:
    """Audio covertion.

    Args:
        from_path (Path): Input file path.
        to_path (Path): Path to audio in dst format.

    Returns:
        bool: If succeed.
    """
    pass


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
        pass

    def _tg_callback_start(self, update: Update, context: CallbackContext):
        """/start."""
        pass

    def _tg_callback_voice(self, update: Update, context: CallbackContext):
        """Voice command."""
        pass

    def _tg_handle_voice(
        self,
        chat_id: int,
        path: Path,
        context: CallbackContext
    ) -> None:
        """Voice processing."""
        pass

    def start(self):
        """Start Bot."""
        pass


if __name__ == "__main__":
    pass
