from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

def remove_punctuation(message: str) -> str:
    """Remove punctuation from a message."""
    return "".join([c for c in message if c not in string.punctuation])


def remove_stopwords(message: str, stop_word: set) -> list:
    """Tokenize and remove stopwords from a message."""
    return [word for word in word_tokenize(message.lower()) if word not in stop_word]


def stem_messages(message: list, stemmer: PorterStemmer) -> list:
    """Stem a list of messages."""
    return [stemmer.stem(el) for el in message]


def sanitize_messages(messages: list) -> list:
    """Remove punctuation, tokenize and remove stopwords from a list of messages."""
    stemmer = PorterStemmer()
    stop_word = set(stopwords.words('english'))
    return [stem_messages(remove_stopwords(remove_punctuation(message), stop_word), stemmer) for message in messages]