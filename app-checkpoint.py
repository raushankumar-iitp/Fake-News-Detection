import streamlit as st
import joblib
import wikipedia

# Page configuration
st.set_page_config(
    page_title="Fake News Detection App",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# --- Sidebar ---
st.sidebar.title("🧠 About This App")
st.sidebar.info("""
This app uses a **Machine Learning model (Logistic Regression)**  
and **Wikipedia API** to verify if a news article is **Fake** or **True**.
""")
st.sidebar.markdown("👨‍💻 *Developed by Raushan Kumar — IIT Patna*")

# --- Main page title ---
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>📰 Fake News Detection App</h1>
    <p style='text-align: center; color: #555;'>Enter a news headline or paragraph below and find out if it's REAL or FAKE!</p>
    """,
    unsafe_allow_html=True
)

# --- Input section ---
user_input = st.text_area("✏️ Enter News Text Here:", height=180)

# --- Prediction button with Wikipedia check ---
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        # ---- 1) Wikipedia search & summary ----
        wiki_found = False
        wiki_info = None
        wiki_pages = []

        try:
            wiki_pages = wikipedia.search(user_input, results=5)
        except Exception:
            wiki_pages = []

        if not wiki_pages:
            short_query = " ".join(user_input.split()[:5])
            try:
                wiki_pages = wikipedia.search(short_query, results=5)
            except:
                wiki_pages = []

        for title in wiki_pages[:3]:
            try:
                page_summary = wikipedia.summary(title, sentences=3, auto_suggest=False, redirect=True)
                if page_summary:
                    wiki_found = True
                    wiki_info = {"title": title, "summary": page_summary}
                    break
            except Exception:
                continue

        if wiki_found and wiki_info:
            st.info(f"📚 Wikipedia match: **{wiki_info['title']}**")
            st.write(wiki_info['summary'])
        else:
            st.info("ℹ️ No clear Wikipedia match found for the query.")

        # ---- 2) ML model prediction ----
        try:
            input_data = vectorizer.transform([user_input])
            prediction = model.predict(input_data)[0]
        except Exception as e:
            st.error("Model or vectorizer error: " + str(e))
            prediction = None

        # ---- 3) Display ML prediction ----
        if prediction is not None:
            if prediction == "TRUE":
                st.success("✅ The ML model predicts: **TRUE**")
            else:
                st.error("🚨 The ML model predicts: **FAKE**")

        # ---- 4) Combined recommendation ----
        recommendation = None
        if wiki_found and wiki_info:
            words = [w.lower() for w in user_input.split() if len(w) > 3]
            matches = sum(1 for w in words if w in wiki_info['summary'].lower())
            if matches >= 1:
                recommendation = "✅ Wikipedia evidence found supporting the news."
            else:
                recommendation = "⚠️ Wikipedia page found, but content doesn’t strongly match."
        else:
            recommendation = "🚨 No Wikipedia evidence found. Only ML model result available."

        st.markdown("---")
        st.subheader("🧩 Combined Recommendation:")
        st.info(recommendation)

# --- Footer ---
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray; font-size: 13px;'>
    Built with ❤️ using Streamlit | Hackathon 2025
    </p>
    """,
    unsafe_allow_html=True
)
