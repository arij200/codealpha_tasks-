import random
import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK packages download
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# --- ADVANCED UI SYSTEM (Glassmorphism & Professional Dark Theme) ---
st.set_page_config(page_title="Smile FAQ Assistant", page_icon="😊", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    h1 {
        color: #00ff87 !important;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #a0aec0;
        text-align: center;
        margin-bottom: 30px;
        font-size: 14px;
    }
    .stChatInput {
        border-radius: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


class AdvancedSmileBot:

    def __init__(self):
        # Professional & Clean FAQ Dataset
        self.faq_data = {
            "describe your purpose and platform": [
                "I am a smart FAQ assistant designed to bring a smile to your face while solving queries! 😊",
                "My purpose is to help you navigate through learning schedules and platform features seamlessly.",
                "I am here to ensure you have a smooth and happy learning journey!",
            ],
            "about intellipaat online learning platform courses": [
                "Intellipaat is a leading global online professional training provider.",
                "It offers high-quality certification courses in AI, Data Science, Cloud Computing, and DevOps.",
                "With Intellipaat, you get lifetime access to course materials and 24/7 technical support.",
            ],
            "about session schedule timings and date": [
                "The upcoming premium interactive session is scheduled for 14th August 2026.",
                "Get ready! The session is highly interactive, packed with live industry use-cases.",
                "You can access the link from your student dashboard on the scheduled date.",
            ],
            "how to contact support or trainers": [
                "You can reach out to our dedicated support team via the 'Help' section on the portal.",
                "Our mentors are available 24/7 to resolve any conceptual doubts you face.",
            ],
        }

        self.faq_questions = list(self.faq_data.keys())
        self.cleaned_faq_questions = [
            self.preprocess_text(q) for q in self.faq_questions
        ]
        
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_faq_questions)

    def preprocess_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        cleaned_tokens = [w for w in tokens if w not in stop_words]
        return " ".join(cleaned_tokens)

    def match_reply(self, user_reply):
        # Safety & Profanity Guardrail
        inappropriate_words = ["sex", "porn", "abuse", "fuck", "bitch"]
        if any(word in user_reply.lower() for word in inappropriate_words):
            return "😊 Let's keep our conversation professional and productive. I am here to assist you with educational queries!"

        # Greetings Handler
        greetings = ["hi", "hello", "hey", "greetings", "good morning"]
        if user_reply.lower().strip() in greetings:
            return "Hello! Hope you are doing great today. 😊 How can I help you with your courses or schedule?"

        cleaned_user_reply = self.preprocess_text(user_reply)
        user_vector = self.vectorizer.transform([cleaned_user_reply])

        similarity_scores = cosine_similarity(user_vector, self.tfidf_matrix)
        best_match_idx = similarity_scores.argmax()
        highest_score = similarity_scores[0][best_match_idx]

        if highest_score < 0.15:
            return self.fallback_intent()

        matched_question = self.faq_questions[best_match_idx]
        return f"😊 {random.choice(self.faq_data[matched_question])}"

    def fallback_intent(self):
        return "I want to give you the perfect answer, but I couldn't find that specific topic. Try asking about sessions or courses! 😊"


# --- STREAMLIT EXECUTION FLOW ---

st.markdown("<h1>😊 Smile AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your Next-Gen Smart FAQ Partner for Learning & Schedules</div>", unsafe_allow_html=True)

# Session states initialization
if "smile_bot" not in st.session_state:
    st.session_state.smile_bot = AdvancedSmileBot()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome! I am your Smile Assistant. Ask me anything about your platform or sessions! 😊"}
    ]

# Bot status check (band hai ya chal raha hai)
if "is_active" not in st.session_state:
    st.session_state.is_active = True

# --- AGAR BOT USER NE BAND (EXIT) KAR DIYA HAI ---
if not st.session_state.is_active:
    st.info("🔒 Chat Session Closed.")
    st.success("Thank you for using Smile AI Assistant! Have a wonderful day ahead! 😊👋")
    
    # Restart karne ka button
    if st.button("🔄 Start New Chat"):
        st.session_state.is_active = True
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Welcome back! How can I help you today? 😊"}
        ]
        st.rerun()

else:
    # Render historic conversation stream
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User interactive text streaming (Input Box)
    if user_input := st.chat_input("Type your question here... (or type 'exit' / 'quit' to close)"):

        # Check for Exit Condition
        if user_input.lower().strip() in ["quit", "exit", "close", "bye"]:
            st.session_state.is_active = False
            st.rerun()
            
        else:
            # Render User Message
            with st.chat_message("user"):
                st.write(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Trigger Advanced matching module
            bot_response = st.session_state.smile_bot.match_reply(user_input)

            # Render Bot Payload
            with st.chat_message("assistant"):
                st.write(bot_response)
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})