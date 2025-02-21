import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")

# ✅ Load Models and Vectorizer
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "XGBoost": joblib.load("xgboost_model.pkl"),
    }

@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")

models = load_models()
vectorizer = load_vectorizer()

# ✅ Header with Centered Title
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color:#FFA500;">📝 Customer Sentiment Analysis</h1>
        <h4 style="color:#666;">Analyze customer reviews with AI-powered sentiment detection.</h4>
    </div>
    """,
    unsafe_allow_html=True,
)

# ✅ Sidebar: Model Selection and App Info
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3771/3771518.png", width=100)
st.sidebar.title("📊 Sentiment Analysis App")

st.sidebar.markdown(
    """
    **Analyze Customer Feedback with AI**
    
    - 📝 Paste a customer review
    - 🔥 Select a Model: Logistic Regression, Random Forest, or XGBoost
    - 🚀 Click 'Analyze Sentiment'
    - 🎯 Get Instant Sentiment Predictions!
    
    ---
    """,
    unsafe_allow_html=True,
)

# ✅ Model Selection Dropdown
selected_model = st.sidebar.selectbox("🔍 Choose a Model", list(models.keys()))

# ✅ Example Reviews
example_reviews = [
    "I absolutely love this product! It's amazing. 😊",
    "Terrible experience. Waste of money! 😡",
    "The service was okay, not great but not bad either.",
    "This is my favorite brand! The quality is top-notch. 💖",
    "I had a really bad experience with this company. Never again!"
]

selected_example = st.sidebar.selectbox("🔍 Try an Example Review", ["Type your own..."] + example_reviews)

# ✅ User Input Section
st.markdown("<h3 style='color:#FFA500;'>✍️ Enter a Customer Review:</h3>", unsafe_allow_html=True)
if selected_example != "Type your own...":
    user_input = st.text_area("", selected_example, height=120)
else:
    user_input = st.text_area("", "", height=120)

# ✅ Sentiment Analysis Button
if st.button("🚀 Analyze Sentiment"):
    if user_input.strip():
        # Vectorize Input Text
        input_vector = vectorizer.transform([user_input])

        # Load Selected Model and Predict
        model = models[selected_model]
        prediction_prob = model.predict_proba(input_vector)[0]  # Get probability scores
        prediction = model.predict(input_vector)[0]

        # Display Result
        sentiment = "😊 Positive" if prediction == 1 else "😡 Negative"
        confidence = max(prediction_prob) * 100  # Convert to percentage

        st.markdown(
            f"""
            <div style="text-align: center;">
                <h2 style="color: {'#28a745' if prediction == 1 else '#dc3545'};">
                    {sentiment}
                </h2>
                <h4>Confidence: {confidence:.2f}%</h4>
                <h5>Model Used: <strong>{selected_model}</strong></h5>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.warning("⚠️ Please enter a review before analyzing!")

# ✅ Footer
st.markdown(
    """
    <hr>
    <div style="text-align: center; font-size: 14px;">
        Built with ❤️ using Streamlit | AI-Powered Sentiment Analysis 🚀
    </div>
    """,
    unsafe_allow_html=True,
)
