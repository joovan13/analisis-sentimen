import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import time  # Tambahkan impor modul time

nltk.download('punkt')


def case_folding(text):
    return text.lower()


def cleansing(text):
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = ' '.join(
        re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
    text = re.sub(r'(b\'{1,2})', "", text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenizing(text):
    return word_tokenize(text)


def normalization(tokens, normal_csv):
    return [normal_csv.get(token, token) for token in tokens]


def stemming(tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(token) for token in tokens]


def remove_stopwords(tokens, stopword_csv):
    return [word for word in tokens if word not in stopword_csv]


def preprocess_text(text, normal_csv, stopword_csv):
    text = case_folding(text)
    text = cleansing(text)
    tokens = tokenizing(text)
    tokens = normalization(tokens, normal_csv)
    tokens = stemming(tokens)
    tokens = remove_stopwords(tokens, stopword_csv)
    return ' '.join(tokens)


def split_data(data):
    X = data['full_text'].fillna('')
    y = data['sentimen'].fillna('')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[
                                   'Negatif', 'Netral', 'Positif'])
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, report, conf_matrix, precision, recall, f1


def train_model_nb(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()

    start_time = time.time()
    model.fit(X_train_vec, y_train)
    training_time = time.time() - start_time
    return model, vectorizer, training_time


def test_model_nb(model, vectorizer, X_test):
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    return y_pred


def sentiment_analyze_nb(data):
    X_train, X_test, y_train, y_test = split_data(data)
    model, vectorizer, training_time = train_model_nb(X_train, y_train)
    y_pred = test_model_nb(model, vectorizer, X_test)
    accuracy, report, conf_matrix, precision, recall, f1 = evaluate_model(
        y_test, y_pred)
    return accuracy, report, conf_matrix, precision, recall, f1, training_time


def hitung_ig(texts, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    y = labels
    ig_scores = mutual_info_classif(X, y)
    feature_names = vectorizer.get_feature_names_out()
    ig_results = pd.DataFrame(
        {'Feature': feature_names, 'Information_Gain': ig_scores})
    return ig_results


def select_features(X_train, gain_df, threshold):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    feature_names = vectorizer.get_feature_names_out()
    gain_df = gain_df.set_index('Feature')
    selected_features = gain_df[gain_df['Information_Gain'] > threshold].index
    mask = [feature in selected_features for feature in feature_names]
    X_train_selected = X_train_vec[:, mask]
    return X_train_selected, mask, vectorizer, len(selected_features)


def train_model_ignb(X_train_selected, y_train):
    model = MultinomialNB()

    start_time = time.time()
    model.fit(X_train_selected, y_train)
    training_time = time.time() - start_time
    return model, training_time


def test_model_ignb(model, mask, vectorizer, X_test):
    X_test_vec = vectorizer.transform(X_test)
    X_test_selected = X_test_vec[:, mask]
    y_pred = model.predict(X_test_selected)
    return y_pred


def sentiment_analyze_ignb(data, gain_df, threshold):
    X_train, X_test, y_train, y_test = split_data(data)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    initial_num_features = X_train_vec.shape[1]

    X_train_selected, mask, vectorizer, num_features = select_features(
        X_train, gain_df, threshold)
    model, training_time = train_model_ignb(X_train_selected, y_train)
    y_pred = test_model_ignb(model, mask, vectorizer, X_test)
    accuracy, report, conf_matrix, precision, recall, f1 = evaluate_model(
        y_test, y_pred)

    st.write('====================================')
    st.write("**Nilai Evaluasi**")
    st.write(f"**Fitur sebelum Information Gain:** {initial_num_features}")
    st.write(f"**Fitur setelah Information Gain:** {num_features}")
    st.write(f"**Accuracy:** {accuracy}")
    st.write("**Confusion Matrix:**")
    st.write(conf_matrix)
    st.write(f"**Precision:** {precision}")
    st.write(f"**Recall:** {recall}")
    st.write(f"**F1 Score:** {f1}")
    st.write(
        f"**Waktu Training (Naive Bayes + Information Gain):** {training_time:.4f} detik")

    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump((model, vectorizer, mask), f)


def app():
    st.title('Analisis Sentimen')

    uploaded_file = st.file_uploader('Upload CSV', type='csv')

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file).fillna('')
            normalization_df = pd.read_csv('text/slang.csv')
            normal_csv = dict(
                zip(normalization_df['slang'], normalization_df['formal']))
            stopword_df = pd.read_csv('text/stopwords.csv')
            stopword_csv = set(stopword_df['stopwords'].tolist())

            if data.empty:
                st.error("Data Kosong.")
            else:
                st.write("Data Preview:")
                st.write(data.head())

                if 'full_text' not in data.columns:
                    st.error("Data harus memiliki kolom 'full_text'.")
                else:
                    if st.button('Preprocess Data'):
                        data['full_text'] = data['full_text'].apply(
                            preprocess_text, args=(normal_csv, stopword_csv)).fillna('')
                        st.write("Data After Preprocessing:")
                        st.write(data.head())
                        st.session_state.preprocessed_data = data
                        st.session_state.preprocess_done = True

        except pd.errors.EmptyDataError:
            st.error("Data file is empty or not valid.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.session_state.get('preprocess_done', False):
        analysis_choice = st.radio(
            "Choose Analysis Method:", ('Naive Bayes Classifier', 'Naive Bayes dan Information Gain'))

        if analysis_choice == 'Naive Bayes dan Information Gain':
            ig_results = hitung_ig(
                st.session_state.preprocessed_data['full_text'], st.session_state.preprocessed_data['sentimen'])
            if not ig_results.empty:
                threshold = 0.0002

        if st.button('Analyze'):
            if 'preprocessed_data' in st.session_state:
                data = st.session_state.preprocessed_data
                if analysis_choice == 'Naive Bayes Classifier':
                    accuracy, report, conf_matrix, precision, recall, f1, training_time = sentiment_analyze_nb(
                        data)
                    st.write(f"**Accuracy:** {accuracy}")
                    st.write("**Confusion Matrix:**")
                    st.write(conf_matrix)
                    st.write(f"**Precision:** {precision}")
                    st.write(f"**Recall:** {recall}")
                    st.write(f"**F1 Score:** {f1}")
                    st.write(
                        f"**Waktu Training (Naive Bayes):** {training_time:.4f} detik")
                elif analysis_choice == 'Naive Bayes dan Information Gain':
                    sentiment_analyze_ignb(data, ig_results, threshold)


if __name__ == "__main__":
    app()
