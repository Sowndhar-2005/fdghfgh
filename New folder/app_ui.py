import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="GradeGuard Predictor", page_icon="🎓", layout="centered")

# Inject Custom CSS for premium UI
st.markdown("""
<style>
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #7c6aff;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #7c6aff, #a855f7);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 12px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(124, 106, 255, 0.4);
    }
    .success-badge {
        padding: 20px;
        background-color: rgba(34, 211, 164, 0.1);
        border: 1px solid #22d3a4;
        border-radius: 10px;
        color: #22d3a4;
        font-size: 1.8rem;
        font-weight: 800;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 0 20px rgba(34, 211, 164, 0.2);
    }
    .fail-badge {
        padding: 20px;
        background-color: rgba(244, 91, 139, 0.1);
        border: 1px solid #f45b8b;
        border-radius: 10px;
        color: #f45b8b;
        font-size: 1.8rem;
        font-weight: 800;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 0 20px rgba(244, 91, 139, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🎓 GradeGuard Predictor")
st.markdown("A Machine Learning powered tool utilizing **Logistic Regression** to predict academic success based on student metrics.")
st.markdown("---")

# --- MODEL TRAINING ---
@st.cache_resource
def load_and_train_model():
    # Load data
    data = pd.read_csv("stu_dataset.csv")
    data['Result_num'] = data['Result'].map({'Fail': 0, 'Pass': 1})
    
    # Features & Target
    X = data[['Study Hours', 'Attendance (%)', 'Assignments']]
    y = data['Result_num']
    
    # Note: Dataset is small so we ensure a consistent reproducible split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Test model accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, acc, data

with st.spinner('Initializing Logistic Regression Model...'):
    model, accuracy, raw_data = load_and_train_model()

# --- MAIN UI ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📊 Your Academic Metrics")
    study = st.slider("📚 Study Hours per Day", min_value=0.0, max_value=12.0, value=5.0, step=0.5)
    attendance = st.slider("🗓️ Attendance (%)", min_value=0, max_value=100, value=75, step=1)
    assignments = st.slider("📝 Assignments Completed", min_value=0, max_value=5, value=3, step=1)

with col2:
    st.subheader("⚙️ Model Details")
    st.markdown(f"**Algorithm:** `Logistic Regression`")
    st.metric(label="Validation Accuracy", value=f"{accuracy * 100:.0f}%")
    
    st.info("The model trains dynamically on `stu_dataset.csv` in the background.")

st.markdown("<br>", unsafe_allow_html=True)

# --- PREDICTION TRIGGER ---
if st.button("Predict My Result"):
    # Reshape input to 2D array as required by scikit-learn
    input_data = [[study, attendance, assignments]]
    
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    confidence = max(probabilities[0])
    
    st.markdown("### Model Prediction")
    if prediction[0] == 1:
        st.markdown('<div class="success-badge">🎉 RESULT: PASS</div>', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown('<div class="fail-badge">⚠️ RESULT: FAIL</div>', unsafe_allow_html=True)
    
    st.markdown(f"<p style='text-align: center; margin-top: 10px; color: gray;'>Prediction Confidence: <strong>{confidence * 100:.1f}%</strong></p>", unsafe_allow_html=True)
    st.progress(float(confidence))

st.markdown("---")

with st.expander("📂 View Training Dataset", expanded=False):
    st.dataframe(raw_data.drop(columns=['Result_num']), use_container_width=True)
