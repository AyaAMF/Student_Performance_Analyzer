import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "users.db"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "student_performance_model.pkl"
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.html("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99,102,241,.15),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(168,85,247,.12),
            transparent 28%
        ),
        #070a12;
}

/* Main */

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0c1020 0%,
            #080b14 100%
        );

    border-right:
        1px solid rgba(255,255,255,.07);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* Buttons */

.stButton > button {
    width: 100%;

    border-radius: 12px;

    border:
        1px solid rgba(129,140,248,.30);

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        );

    color: white;

    font-weight: 700;

    padding:
        .70rem
        1rem;

    transition:
        all .2s ease;
}

.stButton > button:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 10px 30px
        rgba(79,70,229,.30);

    border-color:
        rgba(165,180,252,.70);
}


/* Inputs */

.stTextInput input,
.stNumberInput input {

    background:
        #111827 !important;

    color:
        white !important;

    border:
        1px solid rgba(255,255,255,.09)
        !important;

    border-radius:
        12px !important;
}


/* Select */

div[data-baseweb="select"] > div {

    background:
        #111827 !important;

    border-radius:
        12px !important;

    border:
        1px solid rgba(255,255,255,.09)
        !important;
}


/* Cards */

.card {

    background:
        linear-gradient(
            145deg,
            rgba(18,24,39,.96),
            rgba(10,14,25,.96)
        );

    border:
        1px solid rgba(255,255,255,.07);

    border-radius:
        20px;

    padding:
        26px;

    margin-bottom:
        20px;

    box-shadow:
        0 20px 60px
        rgba(0,0,0,.18);
}


/* Hero */

.hero {

    background:
        radial-gradient(
            circle at 85% 15%,
            rgba(124,58,237,.25),
            transparent 32%
        ),
        radial-gradient(
            circle at 10% 90%,
            rgba(59,130,246,.12),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #111827,
            #0b1020
        );

    border:
        1px solid
        rgba(129,140,248,.20);

    border-radius:
        26px;

    padding:
        35px;

    margin-bottom:
        25px;
}


/* Logo */

.logo {

    font-size:
        26px;

    font-weight:
        800;

    letter-spacing:
        -1px;
}


/* Gradient text */

.gradient-text {

    background:
        linear-gradient(
            90deg,
            #818cf8,
            #c084fc,
            #38bdf8
        );

    -webkit-background-clip:
        text;

    -webkit-text-fill-color:
        transparent;
}


/* Subtitle */

.subtitle {

    color:
        #94a3b8;

    line-height:
        1.7;

    margin-top:
        8px;
}


/* Metric */

.metric-card {

    background:
        linear-gradient(
            145deg,
            #111827,
            #0d1320
        );

    border:
        1px solid
        rgba(255,255,255,.07);

    border-radius:
        17px;

    padding:
        21px;
}


.metric-label {

    color:
        #94a3b8;

    font-size:
        13px;
}


.metric-value {

    color:
        white;

    font-size:
        28px;

    font-weight:
        800;

    margin-top:
        5px;
}


/* Auth */

.auth-container {

    max-width:
        470px;

    margin:
        7vh auto;
}


.auth-header {

    text-align:
        center;

    margin-bottom:
        28px;
}


.auth-icon {

    font-size:
        62px;

    margin-bottom:
        10px;
}


.auth-title {

    font-size:
        34px;

    font-weight:
        800;
}


.auth-subtitle {

    color:
        #94a3b8;

    margin-top:
        8px;
}


/* Result */

.result-box {

    background:
        radial-gradient(
            circle at 50% 0%,
            rgba(99,102,241,.22),
            transparent 45%
        ),
        #101625;

    border:
        1px solid
        rgba(129,140,248,.25);

    border-radius:
        25px;

    padding:
        45px;

    text-align:
        center;

    margin:
        20px 0;
}


.result-score {

    font-size:
        78px;

    font-weight:
        800;

    line-height:
        1;

    margin:
        18px 0;
}


.result-status {

    color:
        #a5b4fc;

    font-size:
        20px;

    font-weight:
        700;
}


/* Step */

.step {

    background:
        #111827;

    border:
        1px solid
        rgba(255,255,255,.06);

    border-radius:
        14px;

    padding:
        14px;

    text-align:
        center;
}


.step-active {

    background:
        linear-gradient(
            135deg,
            #312e81,
            #581c87
        );

    border-color:
        rgba(129,140,248,.50);
}


/* Small text */

.muted {

    color:
        #64748b;

    font-size:
        13px;
}


/* Divider */

hr {

    border-color:
        rgba(255,255,255,.07);
}

</style>
""")


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "prediction_step" not in st.session_state:
    st.session_state.prediction_step = 1

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "profile_data" not in st.session_state:
    st.session_state.profile_data = {}

if "academic_data" not in st.session_state:
    st.session_state.academic_data = {}


# ============================================================
# DATABASE
# ============================================================

def get_connection():

    return sqlite3.connect(
        DB_PATH
    )


def initialize_database():

    conn = get_connection()

    cursor = conn.cursor()

    # Users

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id
            INTEGER
            PRIMARY KEY
            AUTOINCREMENT,

            username
            TEXT
            UNIQUE
            NOT NULL,

            email
            TEXT
            UNIQUE,

            password
            TEXT
            NOT NULL
        )
    """)

    # Predictions

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (

            id
            INTEGER
            PRIMARY KEY
            AUTOINCREMENT,

            username
            TEXT
            NOT NULL,

            predicted_grade
            REAL
            NOT NULL,

            performance
            TEXT
            NOT NULL,

            created_at
            TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Fix old database

    cursor.execute(
        "PRAGMA table_info(users)"
    )

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if "email" not in columns:

        cursor.execute(
            "ALTER TABLE users ADD COLUMN email TEXT"
        )

    conn.commit()

    conn.close()


initialize_database()


# ============================================================
# PASSWORD
# ============================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# ============================================================
# CREATE USER
# ============================================================

def create_user(
    username,
    email,
    password
):

    conn = get_connection()

    try:

        conn.execute(
            """
            INSERT INTO users
            (
                username,
                email,
                password
            )
            VALUES (?, ?, ?)
            """,
            (
                username.strip(),
                email.strip().lower(),
                hash_password(password)
            )
        )

        conn.commit()

        return True

    except sqlite3.IntegrityError:

        return False

    finally:

        conn.close()


# ============================================================
# LOGIN
# ============================================================

def login_user(
    email,
    password
):

    conn = get_connection()

    result = conn.execute(
        """
        SELECT username
        FROM users
        WHERE email = ?
        AND password = ?
        """,
        (
            email.strip().lower(),
            hash_password(password)
        )
    ).fetchone()

    conn.close()

    if result:

        return result[0]

    return None


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    username,
    grade,
    performance
):

    conn = get_connection()

    conn.execute(
        """
        INSERT INTO prediction_history
        (
            username,
            predicted_grade,
            performance
        )
        VALUES (?, ?, ?)
        """,
        (
            username,
            float(grade),
            performance
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# GET HISTORY
# ============================================================

def get_history(username):

    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            id,
            predicted_grade,
            performance,
            created_at
        FROM prediction_history
        WHERE username = ?
        ORDER BY id DESC
        """,
        conn,
        params=(username,)
    )

    conn.close()

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


try:

    model = load_model()

    MODEL_READY = True

except Exception:

    model = None

    MODEL_READY = False


# ============================================================
# FEATURES
# ============================================================

CATEGORICAL_FEATURES = [

    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic"

]


NUMERICAL_FEATURES = [

    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences"

]


FEATURES = (
    CATEGORICAL_FEATURES
    +
    NUMERICAL_FEATURES
)


# ============================================================
# PERFORMANCE
# ============================================================

def get_performance(
    score
):

    if score >= 15:

        return "Very Good"

    if score >= 10:

        return "Passing Range"

    return "Below Passing"


def get_message(
    score
):

    if score >= 15:

        return (
            "The model predicts strong academic performance."
        )

    if score >= 10:

        return (
            "The model predicts performance within the passing range."
        )

    return (
        "The model predicts a score below the passing range."
    )


# ============================================================
# AUTH PAGE
# ============================================================

def auth_page():

    st.html("""
    <div class="auth-container">

        <div class="auth-header">

            <div class="auth-icon">
                🎓
            </div>

            <div class="auth-title">

                <span class="gradient-text">
                    Student Performance AI
                </span>

            </div>

            <div class="auth-subtitle">

                Intelligent academic performance prediction

            </div>

        </div>

    </div>
    """)

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

    if st.session_state.auth_mode == "login":

        st.markdown(
            "### Welcome Back 👋"
        )

        st.caption(
            "Login to access your AI dashboard."
        )

        email = st.text_input(
            "Email",
            placeholder="example@email.com",
            key="login_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button(
            "Login →",
            key="login_button"
        ):

            if not email or not password:

                st.warning(
                    "Please enter your email and password."
                )

            else:

                username = login_user(
                    email,
                    password
                )

                if username:

                    st.session_state.logged_in = True

                    st.session_state.username = username

                    st.session_state.page = "Dashboard"

                    st.rerun()

                else:

                    st.error(
                        "Invalid email or password."
                    )

        st.divider()

        st.write(
            "Don't have an account?"
        )

        if st.button(
            "Create Account",
            key="go_signup"
        ):

            st.session_state.auth_mode = "signup"

            st.rerun()

    # --------------------------------------------------------
    # SIGNUP
    # --------------------------------------------------------

    else:

        st.markdown(
            "### Create Account 🚀"
        )

        st.caption(
            "Create your account and start using the AI platform."
        )

        username = st.text_input(
            "Username",
            placeholder="Your username",
            key="signup_username"
        )

        email = st.text_input(
            "Email",
            placeholder="example@email.com",
            key="signup_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="signup_password"
        )

        confirm = st.text_input(
            "Confirm Password",
            type="password",
            key="signup_confirm"
        )

        if st.button(
            "Create Account →",
            key="signup_button"
        ):

            if not username or not email or not password:

                st.warning(
                    "Please complete all fields."
                )

            elif password != confirm:

                st.error(
                    "Passwords do not match."
                )

            elif len(password) < 6:

                st.error(
                    "Password must contain at least 6 characters."
                )

            elif create_user(
                username,
                email,
                password
            ):

                st.success(
                    "Account created successfully! 🎉"
                )

                st.session_state.auth_mode = "login"

                st.rerun()

            else:

                st.error(
                    "Username or email already exists."
                )

        st.divider()

        st.write(
            "Already have an account?"
        )

        if st.button(
            "Back to Login",
            key="back_login"
        ):

            st.session_state.auth_mode = "login"

            st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.html("""
        <div style="
            padding:10px 5px 20px 5px;
        ">

            <div class="logo">

                🎓
                <span class="gradient-text">
                    Student AI
                </span>

            </div>

            <div class="muted">
                Performance Intelligence Platform
            </div>

        </div>
        """)

        st.divider()

        # Navigation

        if st.button(
            "🏠 Dashboard",
            key="nav_dashboard"
        ):

            st.session_state.page = "Dashboard"

            st.rerun()

        if st.button(
            "🤖 AI Prediction",
            key="nav_prediction"
        ):

            st.session_state.page = "Prediction"

            st.rerun()

        if st.button(
            "📊 Analytics",
            key="nav_analytics"
        ):

            st.session_state.page = "Analytics"

            st.rerun()

        if st.button(
            "👤 Profile",
            key="nav_profile"
        ):

            st.session_state.page = "Profile"

            st.rerun()

        st.divider()

        st.html(f"""
        <div style="
            background:#111827;
            border:1px solid rgba(255,255,255,.07);
            border-radius:15px;
            padding:15px;
        ">

            <div class="muted">
                Logged in as
            </div>

            <div style="
                color:white;
                font-weight:700;
                margin-top:5px;
            ">
                {st.session_state.username}
            </div>

        </div>
        """)

        st.write("")

        if st.button(
            "🚪 Logout",
            key="logout"
        ):

            st.session_state.logged_in = False

            st.session_state.username = ""

            st.session_state.page = "Dashboard"

            st.session_state.prediction = None

            st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():

    history = get_history(
        st.session_state.username
    )

    total = len(history)

    if total:

        average = history[
            "predicted_grade"
        ].mean()

        latest = history.iloc[0][
            "predicted_grade"
        ]

    else:

        average = 0

        latest = 0


    # HERO

    st.html(f"""
    <div class="hero">

        <div class="logo">

            Welcome back,

            <span class="gradient-text">
                {st.session_state.username}
            </span>

            👋

        </div>

        <div class="subtitle">

            Your AI-powered student performance
            analysis platform.

        </div>

    </div>
    """)


    # METRICS

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-label">
                Total Predictions
            </div>

            <div class="metric-value">
                {total}
            </div>

        </div>
        """)

    with c2:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-label">
                Average Prediction
            </div>

            <div class="metric-value">
                {average:.1f}
            </div>

        </div>
        """)

    with c3:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-label">
                Latest Prediction
            </div>

            <div class="metric-value">
                {latest:.1f}
            </div>

        </div>
        """)

    with c4:

        status = (
            "Ready"
            if MODEL_READY
            else "Error"
        )

        st.html(f"""
        <div class="metric-card">

            <div class="metric-label">
                AI Model
            </div>

            <div class="metric-value">
                {status}
            </div>

        </div>
        """)


    st.write("")


    # ACTION CARDS

    left, right = st.columns(2)

    with left:

        st.html("""
        <div class="card">

            <h2>
                🤖 AI Prediction
            </h2>

            <p class="subtitle">

                Enter student information and let
                the trained machine learning model
                estimate the final grade.

            </p>

        </div>
        """)

        if st.button(
            "Start Prediction →",
            key="dashboard_start"
        ):

            st.session_state.page = "Prediction"

            st.session_state.prediction_step = 1

            st.rerun()


    with right:

        st.html("""
        <div class="card">

            <h2>
                📊 Analytics
            </h2>

            <p class="subtitle">

                Explore prediction history,
                average scores and performance trends.

            </p>

        </div>
        """)

        if st.button(
            "Open Analytics →",
            key="dashboard_analytics"
        ):

            st.session_state.page = "Analytics"

            st.rerun()


# ============================================================
# PREDICTION PAGE
# ============================================================

def prediction_page():

    st.title(
        "🤖 AI Student Prediction"
    )

    st.caption(
        "Complete the profile to generate an AI prediction."
    )


    # --------------------------------------------------------
    # STEPS
    # --------------------------------------------------------

    labels = [
        ("1", "Profile"),
        ("2", "Academic"),
        ("3", "Lifestyle"),
        ("4", "Result")
    ]

    columns = st.columns(4)

    for index, (number, label) in enumerate(
        labels,
        start=1
    ):

        with columns[index - 1]:

            if (
                st.session_state.prediction_step
                == index
            ):

                css_class = (
                    "step step-active"
                )

            else:

                css_class = "step"

            st.html(f"""
            <div class="{css_class}">

                <b>
                    {number}
                </b>

                <br>

                <span class="muted">
                    {label}
                </span>

            </div>
            """)


    st.write("")


    # ========================================================
    # STEP 1
    # ========================================================

    if st.session_state.prediction_step == 1:

        st.markdown(
            "## 👤 Student Profile"
        )

        c1, c2 = st.columns(2)

        with c1:

            school = st.selectbox(
                "School",
                [
                    "GP",
                    "MS"
                ]
            )

            sex = st.selectbox(
                "Sex",
                [
                    "F",
                    "M"
                ]
            )

            age = st.number_input(
                "Age",
                min_value=15,
                max_value=25,
                value=17
            )

            address = st.selectbox(
                "Address",
                [
                    "U",
                    "R"
                ]
            )

            famsize = st.selectbox(
                "Family Size",
                [
                    "LE3",
                    "GT3"
                ]
            )

        with c2:

            Pstatus = st.selectbox(
                "Parent Status",
                [
                    "A",
                    "T"
                ]
            )

            guardian = st.selectbox(
                "Guardian",
                [
                    "mother",
                    "father",
                    "other"
                ]
            )

            reason = st.selectbox(
                "Reason",
                [
                    "course",
                    "home",
                    "reputation",
                    "other"
                ]
            )

            Mjob = st.selectbox(
                "Mother Job",
                [
                    "teacher",
                    "health",
                    "services",
                    "at_home",
                    "other"
                ]
            )

            Fjob = st.selectbox(
                "Father Job",
                [
                    "teacher",
                    "health",
                    "services",
                    "at_home",
                    "other"
                ]
            )


        st.write("")

        if st.button(
            "Continue →",
            key="profile_continue"
        ):

            st.session_state.profile_data = {

                "school":
                    school,

                "sex":
                    sex,

                "age":
                    age,

                "address":
                    address,

                "famsize":
                    famsize,

                "Pstatus":
                    Pstatus,

                "guardian":
                    guardian,

                "reason":
                    reason,

                "Mjob":
                    Mjob,

                "Fjob":
                    Fjob

            }

            st.session_state.prediction_step = 2

            st.rerun()


    # ========================================================
    # STEP 2
    # ========================================================

    elif st.session_state.prediction_step == 2:

        st.markdown(
            "## 📚 Academic Information"
        )

        c1, c2 = st.columns(2)

        with c1:

            Medu = st.slider(
                "Mother Education",
                0,
                4,
                2
            )

            Fedu = st.slider(
                "Father Education",
                0,
                4,
                2
            )

            traveltime = st.slider(
                "Travel Time",
                1,
                4,
                2
            )

            studytime = st.slider(
                "Study Time",
                1,
                4,
                2
            )

        with c2:

            failures = st.slider(
                "Previous Failures",
                0,
                3,
                0
            )

            schoolsup = st.selectbox(
                "School Support",
                [
                    "yes",
                    "no"
                ]
            )

            famsup = st.selectbox(
                "Family Support",
                [
                    "yes",
                    "no"
                ]
            )

            paid = st.selectbox(
                "Extra Paid Classes",
                [
                    "yes",
                    "no"
                ]
            )

            higher = st.selectbox(
                "Higher Education",
                [
                    "yes",
                    "no"
                ]
            )


        st.write("")


        c1, c2 = st.columns(2)

        with c1:

            if st.button(
                "← Back",
                key="academic_back"
            ):

                st.session_state.prediction_step = 1

                st.rerun()

        with c2:

            if st.button(
                "Continue →",
                key="academic_continue"
            ):

                st.session_state.academic_data = {

                    "Medu":
                        Medu,

                    "Fedu":
                        Fedu,

                    "traveltime":
                        traveltime,

                    "studytime":
                        studytime,

                    "failures":
                        failures,

                    "schoolsup":
                        schoolsup,

                    "famsup":
                        famsup,

                    "paid":
                        paid,

                    "higher":
                        higher

                }

                st.session_state.prediction_step = 3

                st.rerun()


    # ========================================================
    # STEP 3
    # ========================================================

    elif st.session_state.prediction_step == 3:

        st.markdown(
            "## 🌱 Lifestyle & Social Information"
        )

        c1, c2 = st.columns(2)

        with c1:

            famrel = st.slider(
                "Family Relationship",
                1,
                5,
                4
            )

            freetime = st.slider(
                "Free Time",
                1,
                5,
                3
            )

            goout = st.slider(
                "Going Out",
                1,
                5,
                3
            )

            Dalc = st.slider(
                "Workday Lifestyle",
                1,
                5,
                1
            )

            Walc = st.slider(
                "Weekend Lifestyle",
                1,
                5,
                1
            )

        with c2:

            health = st.slider(
                "Health",
                1,
                5,
                3
            )

            absences = st.number_input(
                "Absences",
                min_value=0,
                max_value=100,
                value=5
            )

            activities = st.selectbox(
                "Activities",
                [
                    "yes",
                    "no"
                ]
            )

            nursery = st.selectbox(
                "Nursery",
                [
                    "yes",
                    "no"
                ]
            )

            internet = st.selectbox(
                "Internet",
                [
                    "yes",
                    "no"
                ]
            )

            romantic = st.selectbox(
                "Romantic Relationship",
                [
                    "yes",
                    "no"
                ]
            )


        st.write("")


        c1, c2 = st.columns(2)

        with c1:

            if st.button(
                "← Back",
                key="lifestyle_back"
            ):

                st.session_state.prediction_step = 2

                st.rerun()


        with c2:

            if st.button(
                "Generate AI Prediction 🚀",
                key="generate"
            ):

                if not MODEL_READY:

                    st.error(
                        "Model file was not found."
                    )

                    st.info(
                        f"Expected model path: {MODEL_PATH}"
                    )

                else:

                    data = {}

                    data.update(
                        st.session_state.profile_data
                    )

                    data.update(
                        st.session_state.academic_data
                    )

                    data.update({

                        "famrel":
                            famrel,

                        "freetime":
                            freetime,

                        "goout":
                            goout,

                        "Dalc":
                            Dalc,

                        "Walc":
                            Walc,

                        "health":
                            health,

                        "absences":
                            absences,

                        "activities":
                            activities,

                        "nursery":
                            nursery,

                        "internet":
                            internet,

                        "romantic":
                            romantic

                    })


                    # Missing features

                    for feature in FEATURES:

                        if feature not in data:

                            if (
                                feature
                                in CATEGORICAL_FEATURES
                            ):

                                data[feature] = "no"

                            else:

                                data[feature] = 0


                    # Correct order

                    input_df = pd.DataFrame(
                        [
                            [
                                data[col]
                                for col in FEATURES
                            ]
                        ],
                        columns=FEATURES
                    )


                    try:

                        prediction = float(
                            model.predict(
                                input_df
                            )[0]
                        )

                        prediction = float(
                            np.clip(
                                prediction,
                                0,
                                20
                            )
                        )


                        performance = (
                            get_performance(
                                prediction
                            )
                        )


                        st.session_state.prediction = (
                            prediction
                        )


                        save_prediction(
                            st.session_state.username,
                            prediction,
                            performance
                        )


                        st.session_state.prediction_step = 4

                        st.rerun()


                    except Exception as error:

                        st.error(
                            "Prediction failed."
                        )

                        st.exception(
                            error
                        )


    # ========================================================
    # STEP 4
    # ========================================================

    elif st.session_state.prediction_step == 4:

        prediction = (
            st.session_state.prediction
        )

        performance = (
            get_performance(
                prediction
            )
        )

        message = (
            get_message(
                prediction
            )
        )


        st.html(f"""

        <div class="result-box">

            <div class="muted">
                AI PREDICTED FINAL GRADE
            </div>

            <div class="result-score">

                {prediction:.2f}

                <span style="
                    font-size:24px;
                    color:#64748b;
                ">
                    / 20
                </span>

            </div>

            <div class="result-status">
                {performance}
            </div>

            <div class="subtitle">
                {message}
            </div>

        </div>

        """)


        c1, c2, c3 = st.columns(3)


        with c1:

            if st.button(
                "🔄 New Prediction",
                key="new_prediction"
            ):

                st.session_state.prediction = None

                st.session_state.prediction_step = 1

                st.rerun()


        with c2:

            if st.button(
                "📊 Analytics",
                key="prediction_analytics"
            ):

                st.session_state.page = "Analytics"

                st.rerun()


        with c3:

            if st.button(
                "🏠 Dashboard",
                key="prediction_dashboard"
            ):

                st.session_state.page = "Dashboard"

                st.rerun()


# ============================================================
# ANALYTICS
# ============================================================

def analytics_page():

    st.title(
        "📊 Analytics"
    )

    st.caption(
        "Your AI prediction history and performance insights."
    )


    history = get_history(
        st.session_state.username
    )


    if history.empty:

        st.info(
            "No predictions yet. Generate your first prediction."
        )

        if st.button(
            "Start Prediction →"
        ):

            st.session_state.page = "Prediction"

            st.session_state.prediction_step = 1

            st.rerun()

        return


    average = (
        history[
            "predicted_grade"
        ].mean()
    )

    highest = (
        history[
            "predicted_grade"
        ].max()
    )

    lowest = (
        history[
            "predicted_grade"
        ].min()
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Predictions",
            len(history)
        )


    with c2:

        st.metric(
            "Average",
            f"{average:.2f}/20"
        )


    with c3:

        st.metric(
            "Highest",
            f"{highest:.2f}/20"
        )


    with c4:

        st.metric(
            "Lowest",
            f"{lowest:.2f}/20"
        )


    st.write("")


    # Trend

    st.subheader(
        "📈 Prediction Trend"
    )

    chart = history.copy()

    chart["created_at"] = pd.to_datetime(
        chart["created_at"]
    )

    chart = chart.sort_values(
        "created_at"
    )

    chart = chart.set_index(
        "created_at"
    )

    st.line_chart(
        chart[
            "predicted_grade"
        ]
    )


    st.write("")


    # Distribution

    st.subheader(
        "📊 Performance Distribution"
    )

    distribution = pd.DataFrame({

        "Performance": [

            "Very Good",

            "Passing Range",

            "Below Passing"

        ],

        "Count": [

            int(
                (
                    history["performance"]
                    == "Very Good"
                ).sum()
            ),

            int(
                (
                    history["performance"]
                    == "Passing Range"
                ).sum()
            ),

            int(
                (
                    history["performance"]
                    == "Below Passing"
                ).sum()
            )

        ]

    })


    st.bar_chart(
        distribution.set_index(
            "Performance"
        )
    )


    st.write("")


    # History

    st.subheader(
        "🕘 Prediction History"
    )

    table = history.copy()

    table["predicted_grade"] = (
        table[
            "predicted_grade"
        ].round(2)
    )

    table["created_at"] = (
        pd.to_datetime(
            table["created_at"]
        )
        .dt.strftime(
            "%Y-%m-%d %H:%M"
        )
    )


    table = table.rename(
        columns={

            "predicted_grade":
                "Predicted Grade",

            "performance":
                "Performance",

            "created_at":
                "Date"

        }
    )


    table = table[
        [
            "Date",
            "Predicted Grade",
            "Performance"
        ]
    ]


    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )


    csv = table.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )


# ============================================================
# PROFILE
# ============================================================

def profile_page():

    st.title(
        "👤 Profile"
    )

    conn = get_connection()

    user = conn.execute(
        """
        SELECT
            username,
            email
        FROM users
        WHERE username = ?
        """,
        (
            st.session_state.username,
        )
    ).fetchone()

    conn.close()


    if user:

        username = user[0]

        email = user[1]

    else:

        username = (
            st.session_state.username
        )

        email = ""


    st.markdown(
        "### Account Information"
    )


    c1, c2 = st.columns(2)


    with c1:

        st.text_input(
            "Username",
            value=username,
            disabled=True
        )


    with c2:

        st.text_input(
            "Email",
            value=email or "",
            disabled=True
        )


    history = get_history(
        st.session_state.username
    )


    st.write("")


    st.markdown(
        "### 📊 Your Activity"
    )


    c1, c2 = st.columns(2)


    with c1:

        st.metric(
            "Total Predictions",
            len(history)
        )


    with c2:

        if len(history):

            st.metric(
                "Average Grade",
                f"""
                {
                    history["predicted_grade"].mean()
                :.2f}
                /20
                """
            )

        else:

            st.metric(
                "Average Grade",
                "—"
            )


# ============================================================
# MAIN
# ============================================================

if not st.session_state.logged_in:

    auth_page()

else:

    render_sidebar()


    if st.session_state.page == "Dashboard":

        dashboard()


    elif st.session_state.page == "Prediction":

        prediction_page()


    elif st.session_state.page == "Analytics":

        analytics_page()


    elif st.session_state.page == "Profile":

        profile_page()