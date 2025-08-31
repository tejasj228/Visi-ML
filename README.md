# VisiML - Interactive Machine Learning Visualization Tool 🧠

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)

VisiML is an interactive educational tool designed to help students and practitioners understand machine learning algorithms through real-time visualization. Watch how algorithms learn, adjust hyperparameters on the fly, and see the immediate impact on model performance.

![VisiML Demo](docs/images/visiml-demo.gif)

## 🌟 Features

### Supported Algorithms

#### Regression

- **Linear Regression** - Multiple optimizers (SGD, Adam, RMSprop), regularization options
- **Polynomial Regression** - Automatic feature engineering with degree control

#### Classification

- **Logistic Regression** - Binary and multi-class support
- **Naive Bayes** - Gaussian, Multinomial, and Bernoulli variants
- **Decision Tree** - Visual tree structure, feature importance
- **Random Forest** - Ensemble visualization, OOB score tracking
- **Support Vector Machine** - Kernel methods, margin visualization
- **K-Nearest Neighbors** - Distance metrics, Voronoi regions

### Key Features

- 🎯 **Real-time Training Visualization** - Watch models learn iteration by iteration
- 🎛️ **Interactive Hyperparameter Tuning** - Adjust parameters and see immediate effects
- 📊 **Comprehensive Metrics** - Loss curves, accuracy, confusion matrices, ROC curves
- 🎨 **Decision Boundary Visualization** - See how models separate classes
- 📈 **Performance Analysis** - Learning curves, residual plots, feature importance
- 💾 **Export Capabilities** - Save models, plots, and generate code
- 🎲 **Built-in Data Generation** - Various patterns for testing (linear, spiral, moons, etc.)

## 🚀 Quick Start

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/VisiML.git
cd VisiML
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running VisiML

#### Streamlit Web App (Recommended)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

#### Standalone Matplotlib Version

```bash
python visiml_standalone.py
```

## 📖 Usage Guide

### Basic Workflow

1. **Select Task Type**: Choose between Regression or Classification
2. **Choose Algorithm**: Pick from available algorithms for your task
3. **Configure Data**:
   - Use built-in data generators with various patterns
   - Or upload your own CSV file
4. **Adjust Hyperparameters**: Use intuitive sliders and controls
5. **Train & Visualize**: Watch the model learn in real-time
6. **Analyze Results**: Explore metrics, visualizations, and insights

### Example: Linear Regression

```python
from visiml.models import LinearRegression
from visiml.data_generator import DataGenerator
from visiml.visualization import plot_regression_predictions

# Generate sample data
X, y = DataGenerator.generate_regression_data(
    n_samples=100, 
    function_type='polynomial',
    noise=0.1
)

# Create and train model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Visualize results
plot_regression_predictions(X, y, model)
```

## 📁 Project Structure

```
VisiML/
├── app.py                    # Streamlit web application
├── visiml_standalone.py      # Matplotlib standalone version
├── visiml/
│   ├── __init__.py
│   ├── models.py            # ML model implementations
│   ├── data_generator.py    # Data generation utilities
│   ├── visualization.py     # Visualization functions
│   └── utils.py            # Helper functions
├── examples/
│   ├── linear_regression.ipynb
│   ├── classification_demo.ipynb
│   └── custom_data_example.ipynb
├── docs/
│   ├── user_guide.md
│   ├── api_reference.md
│   └── images/
├── tests/
│   ├── test_models.py
│   ├── test_visualization.py
│   └── test_data_generator.py
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🎓 Educational Use Cases

VisiML is perfect for:

- **Machine Learning Courses** - Interactive demonstrations in lectures
- **Self-Study** - Hands-on learning of ML concepts
- **Algorithm Comparison** - Side-by-side algorithm performance
- **Hyperparameter Understanding** - See effects of different parameters
- **Debugging Models** - Visualize what's happening inside algorithms

## 🛠️ Advanced Features

### Custom Data Upload

Support for CSV files with automatic feature and target detection.

### Model Export

- Save trained models as pickle files
- Export training history and metrics
- Generate Python code for reproduction

### Visualization Options

- 2D and 3D plotting capabilities
- Animation of training progress
- Interactive plots with hover information

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the need for better ML education tools
- Built with Streamlit, Matplotlib, and scikit-learn
- Thanks to all contributors and users


## 🚧 Roadmap

- [ ] Add deep learning models (Neural Networks)
- [ ] Support for clustering algorithms
- [ ] Time series visualization
- [ ] Model comparison dashboard
- [ ] Export to TensorBoard
- [ ] Mobile-responsive design

---

**Made with ❤️ for ML Education**
