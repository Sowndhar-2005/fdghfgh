# GradeGuard: Academic Outcome Predictor (Python ML)

GradeGuard is an AI-powered Machine Learning project built purely in Python. It evaluates a student's academic metrics to predict whether they will Pass or Fail utilizing Logistic Regression.

## Project Structure
- `app_ui.py`: The main Streamlit Web Application. Run this for a beautiful, browser-based graphical user interface.
- `main.app.py`: The CLI terminal script that handles data loading, model training, and interactive terminal text prediction.
- `stu_dataset.csv`: A robust 250-row synthetic dataset containing historical student records (Study Hours, Attendance, Assignments, Result).
- `report.md`: A detailed report explaining the machine learning architecture, feature set, and model characteristics.

## How It Works
The project utilizes the `scikit-learn` library to perform binary classification. It evaluates three independent numerical features:
1. **Study Hours** (per day)
2. **Attendance (%)**
3. **Assignments** (completed)

It outputs a binary Result (Pass = 1, Fail = 0).

## How to Run

### Prerequisites
Make sure you have Python installed on your system. You will also need `streamlit`, `pandas`, and `scikit-learn`.

Open a terminal and install the required modules:
```bash
pip install streamlit pandas scikit-learn
```

### Method 1: Interactive Web UI (Recommended)
We built a beautiful GUI powered by Streamlit that trains the ML model dynamically.
1. Navigate to the project directory.
2. Run the application:
   ```bash
   streamlit run app_ui.py
   ```
3. Your default web browser will automatically open the UI.
4. Drag the sliders to see instant Confidence Probabilities and Pass/Fail predictions.

### Method 2: Terminal / Command-Line
If you want to test the raw algorithm in your terminal:
1. Navigate to the project directory.
2. Run the script:
   ```bash
   python main.app.py
   ```
3. Enter your metrics when prompted:
   ```text
   Enter Study Hours: 4
   Enter Attendance %: 80
   Enter Assignments completed: 3
   ```
   The model will output `Result: PASS` or `Result: FAIL`.
