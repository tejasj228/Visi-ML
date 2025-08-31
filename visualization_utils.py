"""
VisiML - Visualization Utilities Module
This module provides visualization functions for different ML algorithms and their training processes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle, Rectangle, Ellipse
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
        
sns.set_palette("husl")

def create_figure_layout(task_type: str, model_type: str) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Create a figure layout based on task and model type.
    
    Returns a figure and dictionary of axes for different visualizations.
    """
    if task_type == "Regression":
        if model_type in ["Linear Regression", "Polynomial Regression"]:
            fig = plt.figure(figsize=(15, 10))
            axes = {
                'main': plt.subplot(2, 3, (1, 4)),
                'loss': plt.subplot(2, 3, 3),
                'weights': plt.subplot(2, 3, 6),
                'residuals': plt.subplot(2, 3, 2),
                'qq': plt.subplot(2, 3, 5)
            }
        else:
            fig = plt.figure(figsize=(12, 8))
            axes = {
                'main': plt.subplot(2, 2, (1, 3)),
                'metrics': plt.subplot(2, 2, 2),
                'feature_importance': plt.subplot(2, 2, 4)
            }
    else:  # Classification
        fig = plt.figure(figsize=(16, 10))
        
        if model_type == "Decision Tree":
            axes = {
                'main': plt.subplot(2, 3, (1, 4)),
                'tree': plt.subplot(2, 3, 3),
                'feature_importance': plt.subplot(2, 3, 6),
                'confusion': plt.subplot(2, 3, 2),
                'metrics': plt.subplot(2, 3, 5)
            }
        elif model_type == "Support Vector Machine (SVM)":
            axes = {
                'main': plt.subplot(2, 3, (1, 4)),
                'margin': plt.subplot(2, 3, 3),
                'support_vectors': plt.subplot(2, 3, 6),
                'confusion': plt.subplot(2, 3, 2),
                'metrics': plt.subplot(2, 3, 5)
            }
        else:
            axes = {
                'main': plt.subplot(2, 2, 1),
                'confusion': plt.subplot(2, 2, 2),
                'roc': plt.subplot(2, 2, 3),
                'metrics': plt.subplot(2, 2, 4)
            }
    
    fig.suptitle(f'{model_type} Visualization', fontsize=16)
    return fig, axes

def plot_regression_predictions(ax: plt.Axes, X: np.ndarray, y: np.ndarray, 
                              model: Any, title: str = "Regression Predictions"):
    """Plot regression data and model predictions."""
    ax.clear()
    
    # Plot training data
    ax.scatter(X[:, 0], y, alpha=0.6, label='Data', color='blue', edgecolors='black')
    
    # Plot predictions
    if hasattr(model, 'predict'):
        X_range = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300).reshape(-1, 1)
        
        # Handle multi-dimensional input
        if X.shape[1] > 1:
            X_pred = np.zeros((300, X.shape[1]))
            X_pred[:, 0] = X_range.ravel()
            # Set other features to their mean values
            for i in range(1, X.shape[1]):
                X_pred[:, i] = X[:, i].mean()
        else:
            X_pred = X_range
            
        try:
            y_pred = model.predict(X_pred)
            ax.plot(X_range, y_pred, 'r-', linewidth=2.5, label='Prediction')
            
            # Add confidence interval if available
            if hasattr(model, 'predict_confidence'):
                lower, upper = model.predict_confidence(X_pred)
                ax.fill_between(X_range.ravel(), lower, upper, alpha=0.2, color='red')
        except:
            pass
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Target', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_classification_decision_boundary(ax: plt.Axes, X: np.ndarray, y: np.ndarray, 
                                        model: Any, title: str = "Decision Boundary",
                                        show_probability: bool = False):
    """Plot classification data and decision boundary."""
    ax.clear()
    
    # Create mesh
    h = 0.02  # step size in mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Plot decision boundary if model is fitted
    if hasattr(model, 'predict'):
        try:
            if show_probability and hasattr(model, 'predict_proba'):
                Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                Z = Z.reshape(xx.shape)
                contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
                ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
                plt.colorbar(contour, ax=ax, label='Probability')
            else:
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        except:
            pass
    
    # Plot data points
    unique_classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
    
    for idx, class_val in enumerate(unique_classes):
        mask = y == class_val
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[idx]], 
                  label=f'Class {class_val}', s=50, edgecolors='black', alpha=0.8)
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_loss_curve(ax: plt.Axes, history: Dict[str, List], 
                   metric: str = 'costs', title: str = "Training Loss"):
    """Plot training loss or metric curve."""
    ax.clear()
    
    if metric in history and len(history[metric]) > 0:
        iterations = history.get('iterations', range(len(history[metric])))
        ax.plot(iterations, history[metric], 'b-', linewidth=2, label=metric.capitalize())
        
        # Add smoothed curve
        if len(history[metric]) > 10:
            window = min(len(history[metric]) // 10, 50)
            smoothed = np.convolve(history[metric], np.ones(window)/window, mode='valid')
            smooth_iterations = iterations[window//2:len(smoothed)+window//2]
            ax.plot(smooth_iterations, smoothed, 'r--', linewidth=2, alpha=0.7, label='Smoothed')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        if len(history[metric]) > 0:
            final_value = history[metric][-1]
            ax.annotate(f'Final: {final_value:.4f}', 
                       xy=(iterations[-1], final_value),
                       xytext=(iterations[-1] - len(iterations)*0.1, final_value + (max(history[metric]) - min(history[metric]))*0.1),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10)

def plot_weights_evolution(ax: plt.Axes, history: Dict[str, List], 
                         max_features: int = 10, title: str = "Weights Evolution"):
    """Plot how model weights change during training."""
    ax.clear()
    
    if 'weights' in history and len(history['weights']) > 0:
        weights_array = np.array(history['weights'])
        n_features = min(weights_array.shape[1], max_features)
        
        for i in range(n_features):
            ax.plot(weights_array[:, i], label=f'Weight {i}', linewidth=2)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Weight Value', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

def plot_residuals(ax: plt.Axes, y_true: np.ndarray, y_pred: np.ndarray, 
                  title: str = "Residual Plot"):
    """Plot residuals for regression analysis."""
    ax.clear()
    
    residuals = y_true - y_pred
    
    # Scatter plot of residuals
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='black')
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    # Add confidence bands
    std_residuals = np.std(residuals)
    ax.fill_between([y_pred.min(), y_pred.max()], 
                   [-2*std_residuals, -2*std_residuals],
                   [2*std_residuals, 2*std_residuals],
                   alpha=0.2, color='gray', label='±2σ')
    
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_confusion_matrix(ax: plt.Axes, y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: Optional[List[str]] = None,
                         title: str = "Confusion Matrix"):
    """Plot confusion matrix for classification."""
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError:
        ax.text(0.5, 0.5, 'sklearn not available\nfor confusion matrix', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=14)
        return
    
    ax.clear()
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add labels
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                         ha="center", va="center", color="white" if cm_normalized[i, j] > 0.5 else "black")
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)

def plot_roc_curve(ax: plt.Axes, y_true: np.ndarray, y_proba: np.ndarray, 
                  title: str = "ROC Curve"):
    """Plot ROC curve for binary classification."""
    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        ax.text(0.5, 0.5, 'sklearn not available\nfor ROC curve', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=14)
        return
    
    ax.clear()
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

def plot_feature_importance(ax: plt.Axes, feature_names: List[str], 
                          importances: np.ndarray, title: str = "Feature Importance"):
    """Plot feature importance for tree-based models."""
    ax.clear()
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot bars
    ax.bar(range(len(importances)), importances[indices], color='skyblue', edgecolor='black')
    
    # Add labels
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

def plot_tree_structure(ax: plt.Axes, tree_structure: List[Dict], 
                       title: str = "Decision Tree Structure"):
    """Visualize decision tree structure."""
    ax.clear()
    
    # This is a simplified tree visualization
    # In practice, you might want to use graphviz or similar
    
    ax.text(0.5, 0.5, "Tree Structure\n(Simplified View)", 
           ha='center', va='center', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=14)

def plot_svm_margins(ax: plt.Axes, X: np.ndarray, y: np.ndarray, 
                    model: Any, title: str = "SVM Decision Boundary and Margins"):
    """Plot SVM decision boundary with margins."""
    ax.clear()
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Get decision function values
    if hasattr(model, 'decision_function'):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'red'],
                  linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
        
        # Fill regions
        ax.contourf(xx, yy, Z, levels=[-np.inf, -1, 1, np.inf],
                   colors=['lightcoral', 'lightgray', 'lightblue'], alpha=0.3)
    
    # Plot data points
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='red', s=50, edgecolors='black', label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=50, edgecolors='black', label='Class 1')
    
    # Highlight support vectors
    if hasattr(model, 'support_vectors_'):
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                  s=200, facecolors='none', edgecolors='green', linewidths=3,
                  label='Support Vectors')
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_learning_curves(ax: plt.Axes, train_scores: List[float], 
                        val_scores: List[float], title: str = "Learning Curves"):
    """Plot learning curves to diagnose overfitting/underfitting."""
    ax.clear()
    
    iterations = range(len(train_scores))
    
    ax.plot(iterations, train_scores, 'b-', label='Training Score', linewidth=2)
    ax.plot(iterations, val_scores, 'r-', label='Validation Score', linewidth=2)
    
    # Fill area between curves
    ax.fill_between(iterations, train_scores, val_scores, alpha=0.2, color='gray')
    
    ax.set_xlabel('Training Iterations', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations for overfitting/underfitting
    if len(train_scores) > 0 and len(val_scores) > 0:
        gap = train_scores[-1] - val_scores[-1]
        if gap > 0.1:
            ax.annotate('Possible Overfitting', xy=(len(train_scores)*0.7, val_scores[-1]),
                       fontsize=10, color='red', style='italic')
        elif val_scores[-1] < 0.5:
            ax.annotate('Possible Underfitting', xy=(len(train_scores)*0.7, val_scores[-1]),
                       fontsize=10, color='orange', style='italic')

def create_performance_report(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                            task_type: str) -> Dict[str, Any]:
    """Generate a comprehensive performance report."""
    report = {}
    
    predictions = model.predict(X_test)
    
    if task_type == "Regression":
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        except ImportError:
            # Fallback to manual calculations
            report['mse'] = np.mean((y_test - predictions) ** 2)
            report['rmse'] = np.sqrt(report['mse'])
            report['mae'] = np.mean(np.abs(y_test - predictions))
            ss_res = np.sum((y_test - predictions) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            report['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            report['mean_residual'] = np.mean(y_test - predictions)
            report['std_residual'] = np.std(y_test - predictions)
            return report
        
        report['mse'] = mean_squared_error(y_test, predictions)
        report['rmse'] = np.sqrt(report['mse'])
        report['mae'] = mean_absolute_error(y_test, predictions)
        report['r2'] = r2_score(y_test, predictions)
        
        # Additional regression metrics
        report['mean_residual'] = np.mean(y_test - predictions)
        report['std_residual'] = np.std(y_test - predictions)
        
    else:  # Classification
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        except ImportError:
            # Fallback to manual calculations
            report['accuracy'] = np.mean(predictions == y_test)
            return report
        
        report['accuracy'] = accuracy_score(y_test, predictions)
        
        # Handle multi-class
        average = 'binary' if len(np.unique(y_test)) == 2 else 'macro'
        report['precision'] = precision_score(y_test, predictions, average=average, zero_division=0)
        report['recall'] = recall_score(y_test, predictions, average=average, zero_division=0)
        report['f1'] = f1_score(y_test, predictions, average=average, zero_division=0)
        
        if hasattr(model, 'predict_proba'):
            try:
                from sklearn.metrics import log_loss
                proba = model.predict_proba(X_test)
                report['log_loss'] = log_loss(y_test, proba)
            except ImportError:
                pass
    
    return report

def create_interactive_widget_config(model_type: str) -> Dict[str, Dict]:
    """
    Create configuration for interactive widgets based on model type.
    
    Returns a dictionary with widget configurations for Streamlit.
    """
    configs = {
        "Linear Regression": {
            "learning_rate": {
                "widget": "slider",
                "label": "Learning Rate",
                "min": 0.001,
                "max": 1.0,
                "default": 0.01,
                "step": 0.001,
                "help": "Controls step size in gradient descent. Lower = more stable but slower"
            },
            "regularization": {
                "widget": "selectbox",
                "label": "Regularization Type",
                "options": ["None", "L1 (Lasso)", "L2 (Ridge)", "Elastic Net"],
                "default": "None",
                "help": "Prevents overfitting by penalizing large weights"
            }
        },
        "Decision Tree": {
            "max_depth": {
                "widget": "slider",
                "label": "Maximum Depth",
                "min": 1,
                "max": 20,
                "default": 5,
                "help": "Limits tree depth to prevent overfitting"
            },
            "min_samples_split": {
                "widget": "slider",
                "label": "Min Samples for Split",
                "min": 2,
                "max": 50,
                "default": 2,
                "help": "Minimum samples required to split a node"
            }
        }
        # Add more model configurations...
    }
    
    return configs.get(model_type, {})