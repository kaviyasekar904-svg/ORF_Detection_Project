import streamlit as st
import requests
from bs4 import BeautifulSoup
import pickle
import os

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="AI Fake Job Detection",
    page_icon="🔍",
    layout="centered"
)

# -------------------------
# Load CSS
# -------------------------
def load_css():
    css_file = os.path.join("assets", "styles.css")
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -------------------------
# Load Model
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------
# Title
# -------------------------
st.markdown(
    '<p class="big-title"><h1>🔍 AI Fake Job Detection System</h1></p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Analyze job postings and detect whether they are <b>Fake</b> or <b>Legitimate</b> using Deep Learning</p>',
    unsafe_allow_html=True
)

st.write("")

# -------------------------
# Extract Text From URL
# -------------------------
def extract_text_from_url(url):

    try:

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = " ".join(text.split())

        return text

    except:
        return None


# -------------------------
# URL Input
# -------------------------
url = st.text_input("🔗 Enter Job Posting URL")

# -------------------------
# Analyze Button
# -------------------------
if st.button("Analyze Job"):

    if url == "":
        st.warning("⚠ Please enter a job URL")

    else:

        with st.spinner("Analyzing job posting..."):

            text = extract_text_from_url(url)

            if text is None:

                st.error("❌ Could not extract job description from the URL")

            else:

                vector = vectorizer.transform([text])

                prediction = model.predict(vector)[0]
                probability = model.predict_proba(vector)[0]

                fake_prob = probability[1]
                real_prob = probability[0]

                st.write("")

                if prediction == 1:

                    st.markdown(
                        f'''
                        <div class="result-box fake">
                        ⚠ FAKE JOB DETECTED<br><br>
                        Confidence: {fake_prob:.2f}
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f'''
                        <div class="result-box real">
                        ✅ LEGIT JOB POSTING<br><br>
                        Confidence: {real_prob:.2f}
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

                st.write("")

                st.subheader("Prediction Probabilities")

                st.progress(float(real_prob))
                st.write(f"Legit Job Probability: **{real_prob:.2f}**")

                st.progress(float(fake_prob))
                st.write(f"Fake Job Probability: **{fake_prob:.2f}**")