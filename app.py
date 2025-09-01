import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Rectangle
import matplotlib.cm as cm
import time
import json
from io import BytesIO
import base64

# Import the ML models and utilities from your code
from ml_models import *
from data_generator import DataGenerator
from visualization_utils import *

# Page configuration
st.set_page_config(
    page_title="VisiML - ML Algorithm Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, clean styling
st.markdown("""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Typography */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #e85d04;
        text-align: center;
        margin-bottom: 1rem;
        animation: slideInDown 0.8s ease-out;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 400;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
        animation: slideInUp 0.8s ease-out 0.2s both;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #1a1a1a;
        border-right: 1px solid #2d2d2d;
    }
    
    /* Sidebar section boxes */
    .css-1d391kg .stMarkdown h3 {
        background: #8b4000 !important;
        color: white !important;
        border: 2px solid #e85d04 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        margin: 1rem 0 0.5rem 0 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar selectbox styling */
    .css-1d391kg .stSelectbox > div > div {
        background: #8b4000 !important;
        border: 2px solid #e85d04 !important;
        color: white !important;
    }
    
    .css-1d391kg .stSelectbox label {
        color: white !important;
    }
    
    /* Button styling - Fix visibility */
    .stButton > button {
        background: #e85d04 !important;
        color: white !important;
        border: 1px solid #2d2d2d !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(232, 93, 4, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: #d63384 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(232, 93, 4, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Tab styling - Fix visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent !important;
        padding-bottom: 1rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #2d2d2d !important;
        color: #94a3b8 !important;
        border: 1px solid #3d3d3d !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #3d3d3d !important;
        color: #e85d04 !important;
        border-color: #e85d04 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #e85d04 !important;
        color: white !important;
        border-color: #e85d04 !important;
    }
    
    /* Remove tab content border/highlighting */
    .stTabs [data-baseweb="tab-panel"] {
        border: none !important;
        background: transparent !important;
        padding-top: 1rem !important;
    }
    
    /* Remove tab highlight indicator */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    
    /* Data tabs styling */
    .css-1cpxqw2 .stTabs [data-baseweb="tab"] {
        background: #2d2d2d !important;
        color: #94a3b8 !important;
        border: 1px solid #3d3d3d !important;
    }
    
    .css-1cpxqw2 .stTabs [aria-selected="true"] {
        background: #e85d04 !important;
        color: white !important;
        border-color: #e85d04 !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: #e85d04 !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1px solid #2d2d2d !important;
        background: #1a1a1a !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #e85d04 !important;
        box-shadow: 0 0 0 2px rgba(232, 93, 4, 0.2) !important;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: #1a1a1a !important;
        border: 1px solid #2d2d2d !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        animation: fadeInUp 0.6s ease-out !important;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px) !important;
        border-color: #e85d04 !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: #e85d04 !important;
        border-radius: 4px !important;
    }
    
    /* Success/Error message styling */
    .stAlert {
        border-radius: 8px !important;
        border: 1px solid #e85d04 !important;
        animation: slideInRight 0.5s ease-out !important;
        background: rgba(232, 93, 4, 0.15) !important;
    }
    
    .stAlertContainer {
        background: rgba(232, 93, 4, 0.15) !important;
        border: 1px solid #e85d04 !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stAlertContentInfo"] {
        color: white !important;
    }
    
    [data-testid="stAlertContainer"] {
        background: rgba(232, 93, 4, 0.15) !important;
        border: 1px solid #e85d04 !important;
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.15) !important;
        color: white !important;
        border: 1px solid #10b981 !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.15) !important;
        color: white !important;
        border: 1px solid #ef4444 !important;
    }
    
    .stInfo {
        background: rgba(232, 93, 4, 0.15) !important;
        color: white !important;
        border: 1px solid #e85d04 !important;
        border-radius: 8px !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.15) !important;
        color: white !important;
        border: 1px solid #f59e0b !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #2d2d2d !important;
        border: 1px solid #3d3d3d !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        color: white !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #3d3d3d !important;
        border-color: #e85d04 !important;
    }
    
    /* Modern scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #e85d04;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #d63384;
    }
    
    /* Keyframe animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from { 
            opacity: 0; 
            transform: translateY(20px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    @keyframes slideInDown {
        from { 
            opacity: 0; 
            transform: translateY(-20px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    @keyframes slideInUp {
        from { 
            opacity: 0; 
            transform: translateY(20px); 
    
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    @keyframes slideInRight {
        from { 
            opacity: 0; 
            transform: translateX(20px); 
        }
        to { 
    
            opacity: 1; 
            transform: translateX(0); 
        }
    }
    
    /* Remove white outlines */
    * {
        outline: none !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        display: flex !important;
        gap: 16px !important;
        justify-content: center !important;
    }
    
    .stRadio > div > label {
        background: #2d2d2d !important;
        color: #94a3b8 !important;
        border: 1px solid #3d3d3d !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        font-weight: 500 !important;
    }
    
    .stRadio > div > label:hover {
        background: #3d3d3d !important;
        color: #e85d04 !important;
        border-color: #e85d04 !important;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: #e85d04 !important;
        color: white !important;
        border-color: #e85d04 !important;
    }
    
    /* Custom card styling for main content columns */
    [data-testid="column"]:nth-child(1) {
        background: #2b2b2b !important;
        border: 1px solid #3d3d3d !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-right: 0.5rem !important;
    }
    
    [data-testid="column"]:nth-child(2) {
        background: #2b2b2b !important;
        border: 1px solid #3d3d3d !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-left: 0.5rem !important;
    }
    
    /* Reduce gap between training buttons */
    .stColumns > div {
        gap: 0.25rem !important;
    }
    
    /* Fix metrics card display */
    [data-testid="metric-container"] {
        background: #2b2b2b !important;
        border: 1px solid #3d3d3d !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        animation: fadeInUp 0.6s ease-out !important;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px) !important;
        border-color: #e85d04 !important;
    }
    
    
    /* Metrics card styling */
    .metrics-card {
        background: #2b2b2b !important;
        border: 1px solid #3d3d3d !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
    }
    
    @keyframes slideInFromRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: #2d2d2d !important;
        border: 1px solid #3d3d3d !important;
        color: white !important;
        border-radius: 8px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #e85d04 !important;
        box-shadow: 0 0 0 2px rgba(232, 93, 4, 0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def generate_data(task_type, pattern, n_samples, noise):
    """Generate synthetic data based on user configuration"""
    if task_type == "Regression":
        X, y = DataGenerator.generate_regression_data(
            n_samples=n_samples,
            n_features=1 if pattern != 'polynomial' else 2,
            noise=noise,
            function_type=pattern
        )
    else:
        X, y = DataGenerator.generate_classification_data(
            n_samples=n_samples,
            n_features=2,
            n_classes=2 if pattern != 'spiral' else 3,
            pattern=pattern
        )
    
    # Split data
    split_idx = int(0.8 * len(y))
    indices = np.random.permutation(len(y))
    
    st.session_state.X_train = X[indices[:split_idx]]
    st.session_state.y_train = y[indices[:split_idx]]
    st.session_state.X_test = X[indices[split_idx:]]
    st.session_state.y_test = y[indices[split_idx:]]

from datetime import datetime

# Simple in-app log in place of notifications
if 'logs' not in st.session_state:
    st.session_state.logs = []  # list of {time, level, message}

def log_event(message: str, level: str = 'info'):
    ts = datetime.now().strftime('%H:%M:%S')
    st.session_state.logs.append({'time': ts, 'level': level, 'message': message})
    if len(st.session_state.logs) > 200:
        st.session_state.logs = st.session_state.logs[-200:]

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'iteration' not in st.session_state:
    st.session_state.iteration = 0
if 'training_status' not in st.session_state:
    st.session_state.training_status = ""

def set_status(message: str):
    """Set the single-line training status message shown under Training Metrics."""
    st.session_state.training_status = message
    # Don't mirror to Activity Log to avoid duplicate entries; callers should log explicitly when needed.

# Header
st.markdown('<h1 class="main-header">VisiML - Machine Learning Algorithm Visualizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Learn Machine Learning by Visualizing How Algorithms Work!</p>', unsafe_allow_html=True)

# No toasts; logs are shown inline

# Sidebar for configuration with modern styling
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 0; margin: 0;'>
        <h2 style='color: white; font-family: Inter, sans-serif; font-weight: 700; margin: 0; padding: 0;'>
            Configuration Panel
        </h2>
    </div>
    <hr style='border: none; height: 1px; background: #e85d04; margin: 1rem 0;'>
    """, unsafe_allow_html=True)
    
    # Task selection
    st.markdown("### 1️⃣ Select Task Type")
    task_type = st.selectbox(
        "What type of ML task do you want to explore?",
        ["Regression", "Classification"],
        help="Regression predicts continuous values (e.g., house prices), Classification predicts categories (e.g., spam/not spam)"
    )
    
    # Model selection based on task type
    st.markdown("### 2️⃣ Select Algorithm")
    if task_type == "Regression":
        model_options = {
            "Linear Regression": "Simple linear relationship between features and target",
            "Polynomial Regression": "Non-linear relationships using polynomial features"
        }
    else:
        model_options = {
            "Logistic Regression": "Linear model for binary/multi-class classification",
            "Naive Bayes": "Probabilistic classifier based on Bayes' theorem",
            "Decision Tree": "Tree-like model of decisions",
            "Random Forest": "Ensemble of decision trees",
            "Support Vector Machine (SVM)": "Finds optimal hyperplane for separation",
            "K-Nearest Neighbors (KNN)": "Classifies based on nearest neighbors"
        }
    
    model_type = st.selectbox(
        "Choose an algorithm to visualize:",
        list(model_options.keys()),
        help="Each algorithm has different strengths and use cases"
    )
    
    # Display algorithm description
    st.info(model_options[model_type])
    
    # Data configuration
    st.markdown("### 3️⃣ Configure Data")
    
    data_tab1, data_tab2 = st.tabs(["GENERATE", "UPLOAD"])
    
    with data_tab1:
        n_samples = st.slider("Number of samples", 50, 1000, 200, 50,
                             help="More samples = more training data")
        
        if task_type == "Regression":
            pattern_options = {
                "linear": "Simple linear relationship",
                "polynomial": "Quadratic relationship",
                "sinusoidal": "Wave-like pattern",
                "exponential": "Exponential growth",
                "logarithmic": "Logarithmic relationship"
            }
        else:
            pattern_options = {
                "blobs": "Separated clusters",
                "moons": "Two interleaving half-circles",
                "circles": "Concentric circles",
                "spiral": "Spiral pattern",
                "xor": "XOR pattern (linearly non-separable)"
            }
        
        pattern = st.selectbox("Data pattern", list(pattern_options.keys()),
                              help="Different patterns test different algorithm capabilities")
        
        st.info(f"📊 {pattern_options[pattern]}")
        
        noise = st.slider("Noise level", 0.0, 1.0, 0.1, 0.05,
                         help="Add randomness to make data more realistic")
        
        if st.button(" Generate Data", type="primary"):
            generate_data(task_type, pattern, n_samples, noise)
            log_event("Data generated successfully!", level='success')
    
    with data_tab2:
        st.info("📤 Upload your own CSV file with features and target column")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:", df.head())
            # TODO: Implement custom data loading

# Helper functions (moved to top to avoid NameError)
def get_model_parameters(model_type, task_type):
    """Get parameter configuration for each model"""
    params = {}
    
    if model_type == "Linear Regression":
        params = {
            'learning_rate': {
                'type': 'slider',
                'label': 'Learning Rate',
                'min': 0.001,
                'max': 1.0,
                'default': 0.01,
                'step': 0.001,
                'help': 'Controls how big steps we take during optimization. Too high = instability, too low = slow learning'
            },
            'n_iterations': {
                'type': 'slider',
                'label': 'Number of Iterations',
                'min': 10,
                'max': 5000,
                'default': 1000,
                'step': 10,
                'help': 'How many times to update the model weights'
            },
            'optimizer': {
                'type': 'selectbox',
                'label': 'Optimizer',
                'options': ['sgd', 'momentum', 'adam', 'rmsprop'],
                'default': 'sgd',
                'help': 'Algorithm used to update weights. Adam and RMSprop are adaptive methods'
            },
            'regularization': {
                'type': 'selectbox',
                'label': 'Regularization',
                'options': ['None', 'l2', 'l1', 'elastic'],
                'default': 'None',
                'help': 'Prevents overfitting by penalizing large weights'
            }
        }
    
    elif model_type == "Polynomial Regression":
        params = {
            'degree': {
                'type': 'slider',
                'label': 'Polynomial Degree',
                'min': 1,
                'max': 10,
                'default': 2,
                'step': 1,
                'help': 'Degree of polynomial features. Higher = more complex curves'
            },
            'learning_rate': {
                'type': 'slider',
                'label': 'Learning Rate',
                'min': 0.001,
                'max': 1.0,
                'default': 0.01,
                'step': 0.001,
                'help': 'Controls optimization step size'
            },
            'n_iterations': {
                'type': 'slider',
                'label': 'Number of Iterations',
                'min': 10,
                'max': 5000,
                'default': 1000,
                'step': 10,
                'help': 'Training iterations'
            },
            'regularization': {
                'type': 'selectbox',
                'label': 'Regularization',
                'options': ['None', 'l2', 'l1'],
                'default': 'l2',
                'help': 'Prevents overfitting with polynomial features'
            }
        }
    
    elif model_type == "Logistic Regression":
        params = {
            'learning_rate': {
                'type': 'slider',
                'label': 'Learning Rate',
                'min': 0.001,
                'max': 1.0,
                'default': 0.01,
                'step': 0.001,
                'help': 'Controls gradient descent step size'
            },
            'n_iterations': {
                'type': 'slider',
                'label': 'Number of Iterations',
                'min': 10,
                'max': 5000,
                'default': 1000,
                'step': 10,
                'help': 'Maximum training iterations'
            },
            'penalty': {
                'type': 'selectbox',
                'label': 'Penalty',
                'options': ['l2', 'l1', 'None'],
                'default': 'l2',
                'help': 'Regularization penalty to prevent overfitting'
            },
            'C': {
                'type': 'slider',
                'label': 'Regularization Strength (C)',
                'min': 0.01,
                'max': 100.0,
                'default': 1.0,
                'step': 0.01,
                'help': 'Inverse of regularization strength. Lower = more regularization'
            }
        }
    
    elif model_type == "Naive Bayes":
        params = {
            'distribution': {
                'type': 'selectbox',
                'label': 'Distribution Type',
                'options': ['gaussian', 'multinomial', 'bernoulli'],
                'default': 'gaussian',
                'help': 'Distribution assumption for features'
            },
            'alpha': {
                'type': 'slider',
                'label': 'Smoothing Parameter (Alpha)',
                'min': 0.01,
                'max': 10.0,
                'default': 1.0,
                'step': 0.01,
                'help': 'Laplace smoothing parameter'
            }
        }
    
    elif model_type == "Decision Tree":
        params = {
            'max_depth': {
                'type': 'slider',
                'label': 'Maximum Depth',
                'min': 1,
                'max': 20,
                'default': 5,
                'step': 1,
                'help': 'Maximum depth of the tree. Limits overfitting'
            },
            'min_samples_split': {
                'type': 'slider',
                'label': 'Min Samples to Split',
                'min': 2,
                'max': 50,
                'default': 2,
                'step': 1,
                'help': 'Minimum samples required to split a node'
            },
            'min_samples_leaf': {
                'type': 'slider',
                'label': 'Min Samples per Leaf',
                'min': 1,
                'max': 50,
                'default': 1,
                'step': 1,
                'help': 'Minimum samples required at each leaf'
            },
            'criterion': {
                'type': 'selectbox',
                'label': 'Split Criterion',
                'options': ['gini', 'entropy'],
                'default': 'gini',
                'help': 'Measure of impurity for splits'
            }
        }
    
    elif model_type == "Random Forest":
        params = {
            'n_estimators': {
                'type': 'slider',
                'label': 'Number of Trees',
                'min': 10,
                'max': 200,
                'default': 100,
                'step': 10,
                'help': 'Number of trees in the forest'
            },
            'max_depth': {
                'type': 'slider',
                'label': 'Maximum Depth',
                'min': 1,
                'max': 20,
                'default': 10,
                'step': 1,
                'help': 'Maximum depth of individual trees'
            },
            'min_samples_split': {
                'type': 'slider',
                'label': 'Min Samples to Split',
                'min': 2,
                'max': 20,
                'default': 2,
                'step': 1,
                'help': 'Minimum samples required to split'
            },
            'max_features': {
                'type': 'selectbox',
                'label': 'Max Features',
                'options': ['sqrt', 'log2', 'auto'],
                'default': 'sqrt',
                'help': 'Number of features for best split'
            }
        }
    
    elif model_type == "Support Vector Machine (SVM)":
        params = {
            'C': {
                'type': 'slider',
                'label': 'Regularization (C)',
                'min': 0.01,
                'max': 100.0,
                'default': 1.0,
                'step': 0.01,
                'help': 'Penalty parameter. Higher C = less regularization'
            },
            'kernel': {
                'type': 'selectbox',
                'label': 'Kernel Type',
                'options': ['rbf', 'linear', 'poly', 'sigmoid'],
                'default': 'rbf',
                'help': 'Kernel function for non-linear mapping'
            },
            'gamma': {
                'type': 'selectbox',
                'label': 'Gamma',
                'options': ['scale', 'auto'],
                'default': 'scale',
                'help': 'Kernel coefficient for RBF, poly and sigmoid'
            }
        }
    
    elif model_type == "K-Nearest Neighbors (KNN)":
        params = {
            'n_neighbors': {
                'type': 'slider',
                'label': 'Number of Neighbors (K)',
                'min': 1,
                'max': 50,
                'default': 5,
                'step': 1,
                'help': 'Number of nearest neighbors to consider'
            },
            'weights': {
                'type': 'selectbox',
                'label': 'Weight Function',
                'options': ['uniform', 'distance'],
                'default': 'uniform',
                'help': 'Weight function for neighbors'
            },
            'metric': {
                'type': 'selectbox',
                'label': 'Distance Metric',
                'options': ['euclidean', 'manhattan', 'minkowski'],
                'default': 'euclidean',
                'help': 'Distance metric for finding neighbors'
            }
        }
    
    return params

def start_training(model_type, task_type, params, notify: bool = True):
    """Initialize and start model training.
    notify: when True, emits a 'model created' toast; when False, stays silent (used for param updates).
    """
    # Create model instance based on type
    model_map = {
        'Linear Regression': LinearRegression,
        'Polynomial Regression': PolynomialRegression,
        'Logistic Regression': LogisticRegression,
        'Naive Bayes': NaiveBayes,
        'Decision Tree': DecisionTree,
        'Random Forest': RandomForest,
        'Support Vector Machine (SVM)': SVM,
        'K-Nearest Neighbors (KNN)': KNN
    }
    
    # Clean parameters (convert 'None' strings to None, etc.)
    clean_params = {}
    for k, v in params.items():
        if v == 'None':
            clean_params[k] = None
        elif v == 'True':
            clean_params[k] = True
        elif v == 'False':
            clean_params[k] = False
        else:
            clean_params[k] = v
    
    # Create model
    try:
        model_class = model_map[model_type]
        st.session_state.model = model_class(**clean_params)
        st.session_state.training_started = True
        st.session_state.iteration = 0
        st.session_state.training_complete = False
        if notify:
            log_event(f"{model_type} model created successfully!", level='success')
        set_status(f"{model_type} model created")
    except Exception as e:
        st.error(f"❌ Error creating model: {str(e)}")
        st.error(f"Parameters: {clean_params}")
        return

def train_step():
    """Perform one training step"""
    # This simulates iterative training for visualization
    # In practice, you'd call your model's training method here
    if st.session_state.model and hasattr(st.session_state.model, 'partial_fit'):
        st.session_state.model.partial_fit(st.session_state.X_train, st.session_state.y_train)

def display_training_metrics(placeholder, model, task_type):
    """Display real-time training metrics inside an expander like other sections."""
    with placeholder.container():
        with st.expander("📊 Training Metrics", expanded=True):
            if task_type == "Regression":
                if hasattr(model, 'history') and 'costs' in model.history and model.history['costs']:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Current Loss", f"{model.history['costs'][-1]:.4f}")
                    with col2:
                        if len(model.history['costs']) > 1:
                            delta = model.history['costs'][-1] - model.history['costs'][-2]
                            st.metric("Loss Change", f"{delta:.4f}")
                else:
                    st.write("Metrics will appear after the first iteration.")
            else:
                if hasattr(model, 'history') and 'accuracies' in model.history and model.history['accuracies']:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Current Accuracy", f"{model.history['accuracies'][-1]:.2%}")
                    with col2:
                        if 'costs' in model.history and model.history['costs']:
                            st.metric("Current Loss", f"{model.history['costs'][-1]:.4f}")
                else:
                    st.write("Metrics will appear after the first iteration.")

    # Training status expander removed; status updates will be logged to Activity Log

def create_data_visualization(X_train, y_train, X_test, y_test, task_type):
    """Create initial data visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if task_type == "Regression":
        ax1.scatter(X_train[:, 0], y_train, alpha=0.6, label='Training data')
        ax1.set_xlabel('Feature')
        ax1.set_ylabel('Target')
        ax1.set_title('Training Data')
        ax1.legend()
        
        ax2.scatter(X_test[:, 0], y_test, alpha=0.6, color='orange', label='Test data')
        ax2.set_xlabel('Feature')
        ax2.set_ylabel('Target')
        ax2.set_title('Test Data')
        ax2.legend()
    else:
        # Plot training data
        for class_idx in np.unique(y_train):
            mask = y_train == class_idx
            ax1.scatter(X_train[mask, 0], X_train[mask, 1], alpha=0.6, label=f'Class {class_idx}')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.set_title('Training Data')
        ax1.legend()
        
        # Plot test data
        for class_idx in np.unique(y_test):
            mask = y_test == class_idx
            ax2.scatter(X_test[mask, 0], X_test[mask, 1], alpha=0.6, label=f'Class {class_idx}')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        ax2.set_title('Test Data')
        ax2.legend()
    
    plt.tight_layout()
    return fig

def create_model_visualization(model, X_train, y_train, X_test, y_test, task_type, model_type):
    """Create model-specific visualization during training"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_type} - {task_type}', fontsize=16)
    
    if task_type == "Regression":
        # Plot 1: Data and predictions
        ax1 = axes[0, 0]
        ax1.scatter(X_train[:, 0], y_train, alpha=0.6, label='Training Data', color='blue')
        ax1.scatter(X_test[:, 0], y_test, alpha=0.6, label='Test Data', color='orange')
        
        if hasattr(model, 'predict') and model.is_fitted:
            X_range = np.linspace(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5, 100).reshape(-1, 1)
            if X_train.shape[1] > 1:
                X_pred = np.zeros((100, X_train.shape[1]))
                X_pred[:, 0] = X_range.ravel()
                for i in range(1, X_train.shape[1]):
                    X_pred[:, i] = X_train[:, i].mean()
            else:
                X_pred = X_range
            
            try:
                y_pred = model.predict(X_pred)
                ax1.plot(X_range, y_pred, 'r-', linewidth=2, label='Model Prediction')
            except:
                pass
        
        ax1.set_xlabel('Feature')
        ax1.set_ylabel('Target')
        ax1.set_title('Predictions vs Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss curve
        ax2 = axes[0, 1]
        if hasattr(model, 'history') and 'costs' in model.history and model.history['costs']:
            ax2.plot(model.history['costs'], 'b-', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No training history\navailable', ha='center', va='center')
            ax2.set_title('Training Loss')
        
        # Plot 3: Residuals (if model is fitted)
        ax3 = axes[1, 0]
        if hasattr(model, 'predict') and model.is_fitted:
            try:
                y_pred = model.predict(X_test)
                residuals = y_test - y_pred
                ax3.scatter(y_pred, residuals, alpha=0.6)
                ax3.axhline(y=0, color='red', linestyle='--')
                ax3.set_xlabel('Predicted Values')
                ax3.set_ylabel('Residuals')
                ax3.set_title('Residual Plot')
                ax3.grid(True, alpha=0.3)
            except:
                ax3.text(0.5, 0.5, 'Model not trained', ha='center', va='center')
                ax3.set_title('Residual Plot')
        else:
            ax3.text(0.5, 0.5, 'Model not trained', ha='center', va='center')
            ax3.set_title('Residual Plot')
        
        # Plot 4: Model parameters
        ax4 = axes[1, 1]
        if hasattr(model, 'history') and 'weights' in model.history and model.history['weights']:
            weights_array = np.array(model.history['weights'])
            for i in range(min(weights_array.shape[1], 5)):  # Show max 5 weights
                ax4.plot(weights_array[:, i], label=f'Weight {i}', linewidth=2)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Weight Value')
            ax4.set_title('Weight Evolution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No weight history\navailable', ha='center', va='center')
            ax4.set_title('Weight Evolution')
    
    else:  # Classification
        # Plot 1: Decision boundary
        ax1 = axes[0, 0]
        if X_train.shape[1] >= 2:
            # Create mesh for decision boundary
            h = 0.02
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Plot decision boundary if model is fitted
            if hasattr(model, 'predict') and model.is_fitted:
                try:
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    ax1.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                except:
                    pass
            
            # Plot data points
            unique_classes = np.unique(y_train)
            colors = ['red', 'blue', 'green', 'purple', 'orange']
            for idx, class_val in enumerate(unique_classes):
                mask = y_train == class_val
                ax1.scatter(X_train[mask, 0], X_train[mask, 1], 
                           c=colors[idx % len(colors)], label=f'Class {class_val}', 
                           alpha=0.7, edgecolors='black')
            
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
            ax1.set_title('Decision Boundary')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'Need 2D data for\ndecision boundary', ha='center', va='center')
            ax1.set_title('Decision Boundary')
        
        # Plot 2: Training progress
        ax2 = axes[0, 1]
        if hasattr(model, 'history') and 'costs' in model.history and model.history['costs']:
            ax2.plot(model.history['costs'], 'b-', linewidth=2, label='Loss')
            if 'accuracies' in model.history and model.history['accuracies']:
                ax2_twin = ax2.twinx()
                ax2_twin.plot(model.history['accuracies'], 'g-', linewidth=2, label='Accuracy')
                ax2_twin.set_ylabel('Accuracy', color='g')
                ax2_twin.tick_params(axis='y', labelcolor='g')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss', color='b')
            ax2.set_title('Training Progress')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No training history\navailable', ha='center', va='center')
            ax2.set_title('Training Progress')
        
        # Plot 3: Confusion Matrix (if model is fitted)
        ax3 = axes[1, 0]
        if hasattr(model, 'predict') and model.is_fitted:
            try:
                y_pred = model.predict(X_test)
                unique_labels = np.unique(np.concatenate([y_test, y_pred]))
                cm = np.zeros((len(unique_labels), len(unique_labels)))
                
                for i, true_label in enumerate(unique_labels):
                    for j, pred_label in enumerate(unique_labels):
                        cm[i, j] = np.sum((y_test == true_label) & (y_pred == pred_label))
                
                im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
                ax3.set_xticks(np.arange(len(unique_labels)))
                ax3.set_yticks(np.arange(len(unique_labels)))
                ax3.set_xticklabels(unique_labels)
                ax3.set_yticklabels(unique_labels)
                
                # Add text annotations
                for i in range(len(unique_labels)):
                    for j in range(len(unique_labels)):
                        ax3.text(j, i, f'{int(cm[i, j])}', ha="center", va="center")
                
                ax3.set_xlabel('Predicted')
                ax3.set_ylabel('Actual')
                ax3.set_title('Confusion Matrix')
            except:
                ax3.text(0.5, 0.5, 'Model not trained', ha='center', va='center')
                ax3.set_title('Confusion Matrix')
        else:
            ax3.text(0.5, 0.5, 'Model not trained', ha='center', va='center')
            ax3.set_title('Confusion Matrix')
        
        # Plot 4: Model-specific info
        ax4 = axes[1, 1]
        if hasattr(model, 'predict') and model.is_fitted:
            try:
                y_pred = model.predict(X_test)
                accuracy = np.mean(y_pred == y_test)
                ax4.bar(['Accuracy'], [accuracy], color='lightblue', edgecolor='black')
                ax4.set_ylim(0, 1)
                ax4.set_ylabel('Score')
                ax4.set_title('Model Performance')
                ax4.text(0, accuracy + 0.05, f'{accuracy:.2%}', ha='center')
            except:
                ax4.text(0.5, 0.5, 'Model not trained', ha='center', va='center')
                ax4.set_title('Model Performance')
        else:
            ax4.text(0.5, 0.5, 'Model not trained', ha='center', va='center')
            ax4.set_title('Model Performance')
    
    plt.tight_layout()
    return fig

def display_performance_metrics(model, X_test, y_test, task_type):
    """Display model performance metrics"""
    if not hasattr(model, 'is_fitted') or not model.is_fitted:
        st.warning("⚠️ Model needs to be trained before displaying performance metrics")
        return
    
    predictions = model.predict(X_test)
    
    col1, col2, col3 = st.columns(3)
    
    if task_type == "Regression":
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))
        r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        
        with col1:
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with col2:
            st.metric("Root MSE", f"{rmse:.4f}")
        with col3:
            st.metric("R² Score", f"{r2:.4f}")
    else:
        accuracy = np.mean(predictions == y_test)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        # Add confusion matrix visualization

def display_model_insights(model, model_type, task_type):
    """Display model-specific insights and explanations"""
    st.subheader(f"🔍 Understanding {model_type}")
    
    # Model-specific insights
    if model_type == "Linear Regression":
        st.write("""
        **What you learned:**
        - How gradient descent optimizes the weights
        - The effect of learning rate on convergence
        - How regularization prevents overfitting
        - The importance of feature scaling
        """)
        
        if hasattr(model, 'weights'):
            st.write("**Model Weights:**")
            for i, w in enumerate(model.weights):
                st.write(f"- Feature {i}: {w:.4f}")
            st.write(f"- Bias: {model.bias:.4f}")
    
    # Add insights for other models...

def display_learning_resources(model_type, task_type):
    """Display educational resources for the selected model"""
    st.subheader("📚 Learn More")
    
    resources = {
        "Linear Regression": {
            "concepts": ["Gradient Descent", "Cost Function", "Normal Equation", "Regularization"],
            "tutorials": [
                "Andrew Ng's Machine Learning Course - Linear Regression",
                "StatQuest - Linear Regression",
                "3Blue1Brown - Gradient Descent"
            ],
            "papers": ["Least Squares Estimation - Gauss (1809)"]
        },
        # Add resources for other models...
    }
    
    if model_type in resources:
        st.write("**Key Concepts:**")
        for concept in resources[model_type]["concepts"]:
            st.write(f"- {concept}")
        
        st.write("\n**Recommended Tutorials:**")
        for tutorial in resources[model_type]["tutorials"]:
            st.write(f"- {tutorial}")

def render_activity_log(placeholder):
    """Render the Activity Log inside the provided placeholder so we can call it after logging events."""
    logs = st.session_state.get('logs', [])
    with placeholder.container():
        with st.expander("📝 Activity Log", expanded=False):
            if not logs:
                st.write("No activity yet.")
            else:
                st.markdown(
                    """
                    <style>
                    #activity-log-container { height: 420px; overflow-y: auto; padding-right: 6px; }
                    #activity-log-container::-webkit-scrollbar { width: 8px; }
                    #activity-log-container::-webkit-scrollbar-thumb { background: #3d3d3d; border-radius: 4px; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                entries = []
                for entry in reversed(logs[-100:]):  # show last 100
                    color = {
                        'success': '#10b981',
                        'warning': '#f59e0b',
                        'error': '#ef4444',
                        'info': '#60a5fa'
                    }.get(entry.get('level', 'info'), '#60a5fa')
                    entries.append(
                        f"<div style='padding:6px 10px;border-left:3px solid {color};margin:4px 0;background:rgba(255,255,255,0.02);border-radius:6px;'>"
                        f"<span style='opacity:0.75;font-size:0.85rem'>{entry['time']}</span> "
                        f"<span style='color:{color};text-transform:uppercase;font-size:0.75rem;margin-left:6px'>{entry.get('level','info')}</span> "
                        f"<div style='margin-top:2px'>{entry['message']}</div>"
                        f"</div>"
                    )
                st.markdown("<div id='activity-log-container'>" + "".join(entries) + "</div>", unsafe_allow_html=True)

def display_training_summary(model_type, task_type):
    """Display educational summary after training completion"""
    # Small helper to format metrics
    def maybe(val, fmt):
        try:
            return fmt.format(val)
        except Exception:
            return str(val)

    params = st.session_state.get('last_params', {})
    model = st.session_state.model
    final_loss = None
    accuracy = None
    if hasattr(model, 'history') and isinstance(getattr(model, 'history'), dict):
        costs = model.history.get('costs')
        if costs:
            final_loss = costs[-1]
    # quick accuracy for classification if predict exists
    if task_type == 'Classification' and hasattr(model, 'predict') and st.session_state.get('X_test') is not None:
        try:
            y_pred = model.predict(st.session_state.X_test)
            accuracy = float(np.mean(y_pred == st.session_state.y_test))
        except Exception:
            pass

    # Hardcoded teaching content by algorithm
    summaries = {
        'Linear Regression': {
            'overview': 'Linear Regression models the relationship between input features and a continuous target as a straight line (or hyperplane).',
            'how_it_works': [
                'We assume y ≈ w·x + b (weights times features plus a bias).',
                'A loss function (Mean Squared Error) measures how far predictions are from real values.',
                'Gradient descent iteratively updates weights to reduce the loss.'
            ],
            'what_you_did': [
                'Generated a synthetic regression dataset.',
                f'Trained a {model_type} model using iterative optimization.',
                'Observed the loss curve converging as training progressed.'
            ],
            'interpret_plots': [
                'Predictions vs Data: the red line shows the fitted relationship.',
                'Training Loss: should trend downward and flatten when converged.',
                'Residual Plot: points scattered around 0 indicate a good unbiased fit.',
                'Weight Evolution: shows how parameters stabilize over iterations.'
            ],
            'insights': [
                'Learning rate controls step size—too large can diverge, too small is slow.',
                'Regularization (L1/L2) can improve generalization by penalizing large weights.'
            ]
        },
        'Logistic Regression': {
            'overview': 'Logistic Regression is a linear classifier that outputs probabilities using the sigmoid function.',
            'how_it_works': [
                'We model P(y=1|x) = sigmoid(w·x + b).',
                'Binary cross-entropy loss encourages correct probabilities.',
                'Gradient descent updates parameters to maximize likelihood.'
            ],
            'what_you_did': [
                'Generated a classification dataset.',
                f'Trained a {model_type} classifier and monitored the loss/accuracy.'
            ],
            'interpret_plots': [
                'Decision Boundary: area where the model switches class predictions.',
                'Training Progress: loss down; accuracy up over iterations.',
                'Confusion Matrix: counts of correct/incorrect predictions by class.'
            ],
            'insights': [
                'Feature scaling helps optimization.',
                'Regularization prevents overfitting when classes overlap.'
            ]
        }
    }

    content = summaries.get(model_type, {
        'overview': f'{model_type} is a classic algorithm for {task_type.lower()} with intuitive behavior.',
        'how_it_works': ['The algorithm optimizes a task-specific objective function.'],
        'what_you_did': [f'Trained {model_type} on a synthetic dataset and reviewed key plots.'],
        'interpret_plots': ['Use the provided plots to understand fit quality and learning dynamics.'],
        'insights': ['Tune hyperparameters to balance bias and variance.']
    })

    # Card styling
    st.markdown(
        """
        <style>
        .summary-card { background:#1f1f1f; border:1px solid #3d3d3d; border-radius:12px; padding:16px; margin-top:14px; }
        .summary-card h4 { color:#e85d04; margin:0 0 10px 0; }
        .summary-section { margin:10px 0; }
        .summary-section h5 { color:#cbd5e1; margin:0 0 6px 0; }
        .summary-kv { display:flex; gap:16px; flex-wrap:wrap; margin:6px 0 12px 0; }
        .kv-pill { background:#2b2b2b; border:1px solid #3d3d3d; padding:6px 10px; border-radius:999px; color:#e2e8f0; font-size:0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Build a single HTML block so the wrapper and its content are one element
    facts = [
        f"Task: {task_type}",
        f"Algorithm: {model_type}",
    ]
    if final_loss is not None:
        facts.append(f"Final Loss: {maybe(final_loss, '%.4f')}")
    if accuracy is not None:
        facts.append(f"Accuracy: {maybe(accuracy*100, '%.2f')}%")
    if params:
        facts.append("Params: " + ", ".join([f"{k}={v}" for k,v in list(params.items())[:6]]))

    def li(items):
        return "".join([f"<li>{x}</li>" for x in items])

    html = f"""
    <div class="summary-card">
      <h4>🧠 Learning Summary</h4>
      <div class="summary-kv">{''.join([f"<span class='kv-pill'>{f}</span>" for f in facts])}</div>
      <div class="summary-section"><h5>What is this?</h5>{content['overview']}</div>
      <div class="summary-section"><h5>How it works</h5><ul>{li(content.get('how_it_works', []))}</ul></div>
      <div class="summary-section"><h5>What you did here</h5><ul>{li(content.get('what_you_did', []))}</ul></div>
      <div class="summary-section"><h5>Interpreting the plots</h5><ul>{li(content.get('interpret_plots', []))}</ul></div>
      <div class="summary-section"><h5>More insights</h5><ul>{li(content.get('insights', []))}</ul></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def export_results(model, model_type, task_type, params):
    """Export training results and model"""
    st.subheader("💾 Export Your Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Plots"):
            # Generate and download plots
            fig = create_model_visualization(model, st.session_state.X_train, 
                                          st.session_state.y_train, st.session_state.X_test, 
                                          st.session_state.y_test, task_type, model_type)
            
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label="Download Plot",
                data=buf,
                file_name=f"{model_type.lower().replace(' ', '_')}_visualization.png",
                mime="image/png"
            )
            plt.close()
    
    with col2:
        if st.button("📄 Download Report"):
            report = generate_training_report(model, model_type, task_type, params)
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"{model_type.lower().replace(' ', '_')}_report.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("💻 Generate Code"):
            code = generate_model_code(model_type, params)
            st.download_button(
                label="Download Code",
                data=code,
                file_name=f"{model_type.lower().replace(' ', '_')}_code.py",
                mime="text/plain"
            )

def generate_training_report(model, model_type, task_type, params):
    """Generate a comprehensive training report"""
    report = {
        "model_type": model_type,
        "task_type": task_type,
        "parameters": params,
        "training_samples": len(st.session_state.y_train),
        "test_samples": len(st.session_state.y_test),
        "final_performance": {}
    }
    
    if hasattr(model, 'history'):
        report["training_history"] = {
            k: v[-10:] if isinstance(v, list) else v  # Last 10 values
            for k, v in model.history.items()
            if k in ['costs', 'accuracies', 'iterations']
        }
    
    return json.dumps(report, indent=2)

def generate_model_code(model_type, params):
    """Generate Python code for the trained model"""
    code = f"""# VisiML Generated Code
# Model: {model_type}

import numpy as np
from sklearn.model_selection import train_test_split

# Model parameters
params = {params}

# Load your data here
# X, y = load_your_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
"""
    
    if model_type == "Linear Regression":
        code += """from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
"""
    
    return code

# Main content area

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📊 Visualization Area")
    
    if st.session_state.X_train is not None:
        # Display data statistics
        with st.expander("📈 Data Statistics", expanded=True):
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Training Samples", len(st.session_state.y_train))
            with col_stat2:
                st.metric("Test Samples", len(st.session_state.y_test))
            with col_stat3:
                if task_type == "Classification":
                    n_classes = len(np.unique(st.session_state.y_train))
                    st.metric("Number of Classes", n_classes)
                else:
                    st.metric("Target Range", f"[{st.session_state.y_train.min():.2f}, {st.session_state.y_train.max():.2f}]")
        
        # Visualization placeholder
        viz_placeholder = st.empty()
        
        # Plot initial data or model visualization
        if st.session_state.training_complete and st.session_state.model and hasattr(st.session_state, 'force_viz_refresh') and st.session_state.force_viz_refresh:
            # Force refresh with updated model
            fig = create_model_visualization(st.session_state.model, st.session_state.X_train, 
                                           st.session_state.y_train, st.session_state.X_test, 
                                           st.session_state.y_test, task_type, model_type)
            viz_placeholder.pyplot(fig)
            st.session_state.force_viz_refresh = False  # Reset flag
            plt.close()
        elif st.session_state.training_complete and st.session_state.model:
            # Show model visualization
            fig = create_model_visualization(st.session_state.model, st.session_state.X_train, 
                                           st.session_state.y_train, st.session_state.X_test, 
                                           st.session_state.y_test, task_type, model_type)
            viz_placeholder.pyplot(fig)
            plt.close()
        else:
            # While training is in progress, avoid re-plotting the raw data to prevent flicker
            if st.session_state.training_started:
                pass  # keep current visualization intact during training
            else:
                # Plot initial data only when not training
                fig = create_data_visualization(st.session_state.X_train, st.session_state.y_train, 
                                              st.session_state.X_test, st.session_state.y_test, 
                                              task_type)
                viz_placeholder.pyplot(fig)
                plt.close()
        
        # Activity Log placeholder (we will fill/update after training actions too)
        activity_log_placeholder = st.empty()
        render_activity_log(activity_log_placeholder)

        # Learning Summary placeholder directly below Activity Log in col1
        summary_placeholder = st.empty()
        if st.session_state.get('training_complete') and st.session_state.get('model') is not None:
            try:
                with summary_placeholder.container():
                    display_training_summary(model_type, task_type)
            except Exception as e:
                st.error(f"Summary error: {e}")
    else:
        st.info("👈 Please generate or upload data using the sidebar")

with col2:
    st.markdown("## ⚙️ Hyperparameters")
    
    if model_type:
        params_dict = get_model_parameters(model_type, task_type)
        
        with st.expander("🎛️ Adjust Hyperparameters", expanded=True):
            user_params = {}
            for param_name, param_config in params_dict.items():
                if param_config['type'] == 'slider':
                    user_params[param_name] = st.slider(
                        param_config['label'],
                        param_config['min'],
                        param_config['max'],
                        param_config['default'],
                        param_config.get('step', None),
                        help=param_config['help']
                    )
                elif param_config['type'] == 'selectbox':
                    user_params[param_name] = st.selectbox(
                        param_config['label'],
                        param_config['options'],
                        index=param_config['options'].index(param_config['default']),
                        help=param_config['help']
                    )
                elif param_config['type'] == 'checkbox':
                    user_params[param_name] = st.checkbox(
                        param_config['label'],
                        value=param_config['default'],
                        help=param_config['help']
                    )
        
        # Check if parameters have changed and auto-update model
        if 'last_params' not in st.session_state:
            st.session_state.last_params = {}
        
        # Detect parameter changes (aggregate all changes into a single toast)
        params_changed = False
        changed_items = []
        for param_name, param_value in user_params.items():
            if param_name not in st.session_state.last_params or st.session_state.last_params[param_name] != param_value:
                params_changed = True
                old_val = st.session_state.last_params.get(param_name, 'None')
                changed_items.append(f"{param_name}: {old_val} → {param_value}")
        if params_changed and changed_items:
            log_event("Parameters updated: " + "; ".join(changed_items), level="info")
            # Immediately refresh the Activity Log so the change is visible
            try:
                render_activity_log(activity_log_placeholder)
            except Exception:
                pass
        
        # If parameters changed and we have a trained model, show warning and recreate model
        if params_changed and st.session_state.training_complete:
            st.session_state.last_params = user_params.copy()
            if st.session_state.X_train is not None:
                # Automatically recreate and retrain model with new parameters
                try:
                    # Silent model (no 'created' toast) during param-change retrain
                    start_training(model_type, task_type, user_params, notify=False)
                    # Complete training immediately
                    if st.session_state.model:
                        st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
                        st.session_state.training_complete = True
                        st.session_state.force_viz_refresh = True  # Flag to force visualization refresh
                        # Final single log after retrain
                        log_event("Model retrained successfully with updated parameters", level='success')
                        # Refresh log and summary so UI reflects the new model immediately
                        try:
                            render_activity_log(activity_log_placeholder)
                        except Exception:
                            pass
                        try:
                            summary_placeholder.empty()
                            with summary_placeholder.container():
                                display_training_summary(model_type, task_type)
                        except Exception:
                            pass
                        
                except Exception as e:
                    st.error(f"Error updating model: {str(e)}")
        elif not st.session_state.training_complete:
            st.session_state.last_params = user_params.copy()
        
        # Training controls
        
        # Training buttons side by side with minimal gap (robust selector)
        st.markdown("""
        <style>
        /* Tighten the exact row that contains the Start/Stop buttons */
        div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"] .stButton) {
            gap: 0.5rem !important;
            column-gap: 0.5rem !important;
        }
        div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"] .stButton) > div[data-testid="stColumn"] {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        div[data-testid="stElementContainer"]:has(> .stButton) { margin-bottom: 0 !important; }
        </style>
        """, unsafe_allow_html=True)
        
        cols = st.columns(2)
        with cols[0]:
            start_disabled = (st.session_state.X_train is None) or st.session_state.training_started
            if st.button("▶️ Start Training", type="primary", disabled=start_disabled, key="start_training"):
                # Start only if not already in progress to avoid duplicate logs
                if not st.session_state.training_started:
                    start_training(model_type, task_type, user_params, notify=True)
                    log_event("Training started", level='info')
                    set_status("Training started")
                # No immediate rerun; Streamlit will rerun naturally after button click.
        with cols[1]:
            if st.button("⏹️ Stop Training", disabled=not st.session_state.training_started, key="stop_training"):
                st.session_state.training_started = False
                log_event("Training stopped", level='warning')
                set_status("Training stopped")
        
        # Training progress
        if st.session_state.training_started and st.session_state.model:
            progress_placeholder = st.empty()
            metrics_placeholder = st.empty()
            # Show the metrics card immediately (will show status even if metrics empty)
            display_training_metrics(metrics_placeholder, st.session_state.model, task_type)
            
            # Train the model completely instead of step-by-step for better stability
            if not st.session_state.training_complete:
                training_error = None
                loading_placeholder = st.empty()
                
                with loading_placeholder:
                    with st.spinner('Training model...'):
                        try:
                            st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
                            st.session_state.training_complete = True
                        except Exception as e:
                            training_error = str(e)
                            st.session_state.training_started = False
                
                # Clear loading placeholder
                loading_placeholder.empty()
                        
                # Outside spinner: show results or error
                if training_error:
                    st.error(f"Training failed: {training_error}")
                else:
                    # Final training message
                    log_event("Training completed!", level='success')
                    set_status("Training completed successfully")
                    st.session_state.training_started = False  # mark as stopped after completion
                    
                    # Update visualization and metrics after training just once
                    fig = create_model_visualization(
                        st.session_state.model,
                        st.session_state.X_train,
                        st.session_state.y_train,
                        st.session_state.X_test,
                        st.session_state.y_test,
                        task_type,
                        model_type,
                    )
                    viz_placeholder.pyplot(fig)
                    display_training_metrics(metrics_placeholder, st.session_state.model, task_type)
                    # Re-render the Activity Log so the latest messages are visible immediately
                    render_activity_log(activity_log_placeholder)
                    # Render/Update the Learning Summary immediately after initial training
                    try:
                        summary_placeholder.empty()
                    except Exception:
                        # If placeholder wasn't created (edge case), create it now under col1
                        try:
                            summary_placeholder = st.empty()
                        except Exception:
                            summary_placeholder = None
                    if summary_placeholder is not None:
                        try:
                            with summary_placeholder.container():
                                display_training_summary(model_type, task_type)
                        except Exception:
                            pass
                    # No forced rerun; normal rerun is enough and prevents duplicate renders/logs.
    
    st.markdown("</div>", unsafe_allow_html=True)
# (Learning Summary is rendered under Activity Log in col1 via summary_placeholder)

 # Footer section removed per request

