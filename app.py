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

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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

# Header
st.markdown('<h1 class="main-header">VisiML - Machine Learning Algorithm Visualizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Learn Machine Learning by Visualizing How Algorithms Work!</p>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.markdown("## 🎯 Configuration Panel")
    
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
    
    data_tab1, data_tab2 = st.tabs(["Generate Data", "Upload Data"])
    
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
        
        if st.button("🎲 Generate Data", type="primary"):
            generate_data(task_type, pattern, n_samples, noise)
            st.success("✅ Data generated successfully!")
    
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

def start_training(model_type, task_type, params):
    """Initialize and start model training"""
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
        st.success(f"✅ {model_type} model created successfully!")
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
    """Display real-time training metrics"""
    with placeholder.container():
        if task_type == "Regression":
            if hasattr(model, 'history') and 'costs' in model.history:
                col1, col2 = st.columns(2)
                with col1:
                    if model.history['costs']:
                        st.metric("Current Loss", f"{model.history['costs'][-1]:.4f}")
                with col2:
                    if len(model.history['costs']) > 1:
                        delta = model.history['costs'][-1] - model.history['costs'][-2]
                        st.metric("Loss Change", f"{delta:.4f}")
        else:
            if hasattr(model, 'history') and 'accuracies' in model.history:
                col1, col2 = st.columns(2)
                with col1:
                    if model.history['accuracies']:
                        st.metric("Current Accuracy", f"{model.history['accuracies'][-1]:.2%}")
                with col2:
                    if 'costs' in model.history and model.history['costs']:
                        st.metric("Current Loss", f"{model.history['costs'][-1]:.4f}")

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
            # Plot initial data
            fig = create_data_visualization(st.session_state.X_train, st.session_state.y_train, 
                                          st.session_state.X_test, st.session_state.y_test, 
                                          task_type)
            viz_placeholder.pyplot(fig)
            plt.close()
    else:
        st.info("👈 Please generate or upload data using the sidebar")

with col2:
    st.markdown("## ⚙️ Hyperparameters")
    
    if model_type:
        params_dict = get_model_parameters(model_type, task_type)
        
        with st.expander("🎛️ Adjust Hyperparameters", expanded=True):
            st.info("💡 **Tip**: Hover over each parameter to see what it does!")
            
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
        
        # Detect parameter changes
        params_changed = False
        for param_name, param_value in user_params.items():
            if param_name not in st.session_state.last_params or st.session_state.last_params[param_name] != param_value:
                params_changed = True
                st.warning(f"🔄 Parameter '{param_name}' changed from {st.session_state.last_params.get(param_name, 'None')} to {param_value}")
                break
        
        # If parameters changed and we have a trained model, show warning and recreate model
        if params_changed and st.session_state.training_complete:
            st.session_state.last_params = user_params.copy()
            if st.session_state.X_train is not None:
                # Automatically recreate and retrain model with new parameters
                try:
                    start_training(model_type, task_type, user_params)
                    # Complete training immediately
                    if st.session_state.model:
                        st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
                        st.session_state.training_complete = True
                        st.session_state.force_viz_refresh = True  # Flag to force visualization refresh
                        st.success(f"🔄 Model updated! New K = {st.session_state.model.n_neighbors}")
                        st.info("🔄 Model updated with new parameters!")
                        
                except Exception as e:
                    st.error(f"Error updating model: {str(e)}")
        elif not st.session_state.training_complete:
            st.session_state.last_params = user_params.copy()
        
        # Training controls
        st.markdown("### 🚀 Training Controls")
        
        # Show current model info if available
        if st.session_state.training_complete and st.session_state.model:
            st.info(f"Current Model: K = {getattr(st.session_state.model, 'n_neighbors', 'N/A')}")
        
        col_train1, col_train2 = st.columns(2)
        with col_train1:
            if st.button("▶️ Start Training", type="primary", disabled=st.session_state.X_train is None):
                start_training(model_type, task_type, user_params)
        
        with col_train2:
            if st.button("⏹️ Stop Training", disabled=not st.session_state.training_started):
                st.session_state.training_started = False
        
        # Training progress
        if st.session_state.training_started and st.session_state.model:
            progress_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            # Train the model completely instead of step-by-step for better stability
            if not st.session_state.training_complete:
                with st.spinner('Training model...'):
                    try:
                        st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
                        st.session_state.training_complete = True
                        st.success("✅ Training completed!")
                        
                        # Update visualization after training
                        fig = create_model_visualization(st.session_state.model, st.session_state.X_train, 
                                                       st.session_state.y_train, st.session_state.X_test, 
                                                       st.session_state.y_test, task_type, model_type)
                        viz_placeholder.pyplot(fig)
                        plt.close()
                        
                        # Display metrics
                        display_training_metrics(metrics_placeholder, st.session_state.model, task_type)
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        st.session_state.training_started = False

# Bottom section - Results and Insights
if st.session_state.training_complete and st.session_state.model:
    st.markdown("## 📊 Results & Insights")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Metrics", "🎯 Model Insights", 
                                       "📚 Learning Resources", "💾 Export Results"])
    
    with tab1:
        display_performance_metrics(st.session_state.model, st.session_state.X_test, 
                                  st.session_state.y_test, task_type)
    
    with tab2:
        display_model_insights(st.session_state.model, model_type, task_type)
    
    with tab3:
        display_learning_resources(model_type, task_type)
    
    with tab4:
        export_results(st.session_state.model, model_type, task_type, user_params)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Made with ❤️ for ML Education | VisiML - Making Machine Learning Visual and Intuitive</p>
    <p>Star ⭐ this project on <a href='https://github.com/your-username/VisiML'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)

