import streamlit as st
import classify


@st.cache_data
def preprocess_text_cached(text, normal_csv, stopword_csv):
    normalization_dict, stop_words = classify.load_resources(
        normal_csv, stopword_csv)
    return classify.preprocess_text(text, normalization_dict, stop_words)


@st.cache_data
def classify_text_cached(processed_text):
    return classify.classify(processed_text)


normal_csv = 'text/slang.csv'
stopword_csv = 'text/stopwords.csv'


def main():
    st.title("Analisis Sentimen")

    user_input = st.text_input("Masukkan Kalimat :")

    if st.button("Analisis"):
        if user_input:
            processed_text = preprocess_text_cached(
                user_input, normal_csv, stopword_csv)

            sentiment = classify_text_cached(processed_text)

            # st.write(f"Teks setelah preprocess : {processed_text}")
            st.write(f"Hasil sentiment : {sentiment}")

        else:
            st.write("Enter a sentence.")


if __name__ == "__main__":
    main()
