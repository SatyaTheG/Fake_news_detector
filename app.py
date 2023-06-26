import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

model_name = 'finalized_model.sav'
model = pickle.load(open(model_name, 'rb'))

vectorizer_name = 'finalized_vectorizer.sav'
vector = pickle.load(open(vectorizer_name, 'rb'))

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def process_text(input_text):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', input_text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    X = vector.transform([review])
    result = model.predict(X)
    final = output_lable(result)
    return final

# Streamlit app
def main():
    st.set_page_config(page_title='FAKE NEWS DETECTOR',
                       page_icon=':clipboard:', layout='wide')
    st.title('FAKE NEWS DETECTION MODEL :sleuth_or_spy:')

    # Text input
    input_text = st.text_area("Enter your article that needs to be checked")

    # Process button
    if st.button("Process"):
        if input_text:
            # Process the input text
            processed_output = process_text(input_text)

            # Display the processed output
            st.write("Predicted Output:")
            st.write(processed_output)

if __name__ == "__main__":
    main()
