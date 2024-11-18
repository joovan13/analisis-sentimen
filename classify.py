import pickle
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string

nltk.download('punkt')

with open('sentiment_model.pkl', 'rb') as f:
    model, vectorizer, mask = pickle.load(f)


def case_folding(text):
    return text.lower()


def cleansing(text):
    text = re.sub(r'https?:\/\/\S+', '', str(text))
    text = ' '.join(
        re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", str(text)).split())
    text = re.sub(r'(b\'{1,2})', "", str(text))
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = re.sub(r'\d+', '', str(text))
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text


def tokenizing(text):
    return nltk.word_tokenize(str(text))


def normalization(tokens, normal_csv):
    return [normal_csv.get(token, token) for token in tokens]


def stemming(tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(token) for token in tokens]


def remove_stopwords(tokens, stop_words):
    return [word for word in tokens if word not in stop_words]


def preprocess_text(text, normal_csv, stop_words):
    text = case_folding(text)
    text = cleansing(text)
    tokens = tokenizing(text)
    tokens = normalization(tokens, normal_csv)
    tokens = stemming(tokens)
    tokens = remove_stopwords(tokens, stop_words)
    return ' '.join(tokens)


normal_csv = 'text/slang.csv'
stopword_csv = 'text/stopwords.csv'


def load_resources(normal_csv, stopword_csv):
    normalization_df = pd.read_csv(normal_csv)
    normal_csv = dict(
        zip(normalization_df['slang'], normalization_df['formal']))

    stop_words = pd.read_csv(stopword_csv)
    stop_words = set(stop_words['stopwords'].tolist())

    return normal_csv, stop_words


def classify(processed_text):
    text_vec = vectorizer.transform([processed_text])
    text_selected = text_vec[:, mask]

    prediction = model.predict(text_selected)[0]
    return prediction


# input_text = 'presiden pak jokowi luar biasa kecerdasan dn prestasi nya bisa menarik prusahaan2 luar untuk mengenjot ekonomi negara dn membukak lapagan krja yg besar buat rakyat indonesia nanti nya'
# print(classify(input_text))
