"""
VisiML - Data Generation Module
This module provides synthetic data generation utilities for testing and visualizing ML algorithms.
"""

import numpy as np
from typing import Tuple, Optional

class DataGenerator:
    """
    Enhanced data generator with multiple patterns for regression and classification tasks.
    
    This class provides methods to generate synthetic datasets that showcase
    different characteristics and challenges for ML algorithms.
    """
    
    @staticmethod
    def generate_regression_data(n_samples: int = 100, 
                               n_features: int = 1, 
                               noise: float = 0.1, 
                               function_type: str = 'linear', 
                               random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic regression data with various patterns.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        n_features : int
            Number of input features
        noise : float
            Standard deviation of Gaussian noise added to targets
        function_type : str
            Type of function to generate:
            - 'linear': Linear relationship
            - 'polynomial': Quadratic relationship
            - 'sinusoidal': Sine wave pattern
            - 'exponential': Exponential growth
            - 'logarithmic': Logarithmic relationship
            - 'step': Step function
        random_state : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray of shape (n_samples,)
            Target values
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        # Generate base features
        if function_type in ['sinusoidal', 'exponential', 'logarithmic', 'step']:
            # For these functions, we'll use a structured range for better visualization
            X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
            if n_features > 1:
                # Add additional random features
                X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
        else:
            X = np.random.randn(n_samples, n_features)
        
        # Generate targets based on function type
        if function_type == 'linear':
            # y = w^T * X + b + noise
            coefficients = np.random.randn(n_features) * 2
            bias = np.random.randn()
            y = X.dot(coefficients) + bias
            
            # Add description for user understanding
            description = f"Linear: y = {coefficients[0]:.2f}*x1"
            if n_features > 1:
                for i in range(1, n_features):
                    description += f" + {coefficients[i]:.2f}*x{i+1}"
            description += f" + {bias:.2f}"
            
        elif function_type == 'polynomial':
            # y = sum(x_i^2) + interactions + noise
            y = np.sum(X**2, axis=1)
            if n_features > 1:
                # Add interaction terms
                y += 0.5 * X[:, 0] * X[:, 1]
            description = "Polynomial: y = x1² + x2² + 0.5*x1*x2"
            
        elif function_type == 'sinusoidal':
            # y = sin(2π*x1) + 0.5*cos(4π*x2) + noise
            y = np.sin(2 * np.pi * X[:, 0])
            if n_features > 1:
                y += 0.5 * np.cos(4 * np.pi * X[:, 1])
            description = "Sinusoidal: y = sin(2π*x1) + 0.5*cos(4π*x2)"
            
        elif function_type == 'exponential':
            # y = exp(x/2) + noise
            y = np.exp(X[:, 0] / 2)
            description = "Exponential: y = exp(x/2)"
            
        elif function_type == 'logarithmic':
            # y = log(|x| + 1) + noise
            X = np.abs(X) + 1  # Ensure positive values
            y = np.log(X[:, 0])
            description = "Logarithmic: y = log(|x| + 1)"
            
        elif function_type == 'step':
            # Step function with multiple levels
            y = np.zeros(n_samples)
            thresholds = np.linspace(X[:, 0].min(), X[:, 0].max(), 5)
            for i, t in enumerate(thresholds[:-1]):
                mask = (X[:, 0] >= t) & (X[:, 0] < thresholds[i+1])
                y[mask] = i
            description = "Step function with 4 levels"
            
        else:
            raise ValueError(f"Unknown function type: {function_type}")
        
        # Add noise
        y += noise * np.random.randn(n_samples)
        
        # Print description for user understanding
        print(f"Generated {function_type} data: {description}")
        print(f"Added Gaussian noise with σ = {noise}")
        
        return X, y
    
    @staticmethod
    def generate_classification_data(n_samples: int = 200, 
                                   n_features: int = 2, 
                                   n_classes: int = 2, 
                                   pattern: str = 'blobs', 
                                   class_sep: float = 1.0, 
                                   random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic classification data with various patterns.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        n_features : int
            Number of features (only 2D supported for most patterns)
        n_classes : int
            Number of classes
        pattern : str
            Type of pattern to generate:
            - 'blobs': Gaussian blobs (linearly separable)
            - 'moons': Two interleaving half-circles
            - 'circles': Concentric circles
            - 'spiral': Spiral pattern
            - 'xor': XOR pattern (checkerboard)
        class_sep : float
            Separation between classes (for applicable patterns)
        random_state : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray of shape (n_samples,)
            Class labels
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        if pattern == 'blobs':
            # Generate Gaussian blobs
            X = []
            y = []
            samples_per_class = n_samples // n_classes
            
            # Create cluster centers
            if n_classes == 2:
                centers = np.array([[-class_sep, -class_sep], [class_sep, class_sep]])
            else:
                # Arrange centers in a circle
                angles = np.linspace(0, 2*np.pi, n_classes, endpoint=False)
                centers = np.column_stack([
                    class_sep * 2 * np.cos(angles),
                    class_sep * 2 * np.sin(angles)
                ])
            
            for i in range(n_classes):
                # Generate samples around each center
                cluster = np.random.randn(samples_per_class, n_features) * 0.5 + centers[i]
                X.append(cluster)
                y.append(np.full(samples_per_class, i))
                
            X = np.vstack(X)
            y = np.hstack(y)
            
            description = f"Gaussian blobs with {n_classes} classes"
            
        elif pattern == 'moons':
            # Generate two interleaving half circles
            n_samples_per_class = n_samples // 2
            
            # First moon (upper)
            theta1 = np.linspace(0, np.pi, n_samples_per_class)
            X1 = np.column_stack([
                np.cos(theta1),
                np.sin(theta1)
            ])
            
            # Second moon (lower)
            theta2 = np.linspace(0, np.pi, n_samples_per_class)
            X2 = np.column_stack([
                1 - np.cos(theta2),
                1 - np.sin(theta2) - 0.5
            ])
            
            X = np.vstack([X1, X2])
            y = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
            
            # Add noise
            X += np.random.randn(n_samples, 2) * 0.1
            
            description = "Two interleaving half-moons (non-linearly separable)"
            
        elif pattern == 'circles':
            # Generate concentric circles
            n_samples_per_class = n_samples // 2
            
            # Inner circle
            theta1 = np.linspace(0, 2*np.pi, n_samples_per_class)
            r1 = 1
            X1 = np.column_stack([
                r1 * np.cos(theta1),
                r1 * np.sin(theta1)
            ])
            
            # Outer circle
            theta2 = np.linspace(0, 2*np.pi, n_samples_per_class)
            r2 = 2.5
            X2 = np.column_stack([
                r2 * np.cos(theta2),
                r2 * np.sin(theta2)
            ])
            
            X = np.vstack([X1, X2])
            y = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
            
            # Add noise
            X += np.random.randn(n_samples, 2) * 0.1
            
            description = "Concentric circles (non-linearly separable)"
            
        elif pattern == 'spiral':
            # Generate spiral pattern
            n_samples_per_class = n_samples // n_classes
            X = []
            y = []
            
            for i in range(n_classes):
                # Create spiral for each class
                theta = np.linspace(0, 4*np.pi, n_samples_per_class)
                r = np.linspace(0.5, 2, n_samples_per_class)
                angle_offset = i * 2 * np.pi / n_classes
                
                x = r * np.cos(theta + angle_offset)
                y_coord = r * np.sin(theta + angle_offset)
                
                X.append(np.column_stack([x, y_coord]))
                y.append(np.full(n_samples_per_class, i))
                
            X = np.vstack(X)
            y = np.hstack(y)
            
            # Add small noise
            X += np.random.randn(X.shape[0], 2) * 0.05
            
            description = f"Spiral pattern with {n_classes} interleaving spirals"
            
        elif pattern == 'xor':
            # XOR pattern (checkerboard)
            X = np.random.uniform(-3, 3, (n_samples, 2))
            y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
            
            description = "XOR pattern (checkerboard, non-linearly separable)"
            
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        # Shuffle data
        indices = np.random.permutation(len(y))
        X = X[indices]
        y = y[indices].astype(int)
        
        # Print description
        print(f"Generated {pattern} pattern: {description}")
        print(f"Samples per class: {np.bincount(y)}")
        
        return X, y
    
    @staticmethod
    def generate_regression_animation_data(n_frames: int = 50, 
                                         n_samples: int = 100,
                                         function_type: str = 'linear') -> list:
        """
        Generate data for animating the learning process.
        
        Creates a sequence of datasets that gradually reveal the pattern,
        useful for demonstrating how models learn over time.
        """
        frames = []
        
        # Generate the final dataset
        X_final, y_final = DataGenerator.generate_regression_data(
            n_samples=n_samples,
            function_type=function_type,
            noise=0.1
        )
        
        # Create frames with increasing amounts of data
        for i in range(1, n_frames + 1):
            n_current = int(n_samples * (i / n_frames))
            indices = np.random.choice(n_samples, n_current, replace=False)
            
            frames.append({
                'X': X_final[indices],
                'y': y_final[indices],
                'frame': i,
                'total_frames': n_frames
            })
            
        return frames
    
    @staticmethod
    def describe_data(X: np.ndarray, y: np.ndarray, task_type: str = 'classification') -> dict:
        """
        Generate descriptive statistics for the dataset.
        
        Returns a dictionary with various statistics about the data.
        """
        stats = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_stats': {}
        }
        
        # Feature statistics
        for i in range(X.shape[1]):
            stats['feature_stats'][f'feature_{i}'] = {
                'mean': float(np.mean(X[:, i])),
                'std': float(np.std(X[:, i])),
                'min': float(np.min(X[:, i])),
                'max': float(np.max(X[:, i]))
            }
        
        if task_type == 'classification':
            # Class distribution
            unique, counts = np.unique(y, return_counts=True)
            stats['n_classes'] = len(unique)
            stats['class_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
            stats['class_balance'] = float(np.min(counts) / np.max(counts))
        else:
            # Target statistics for regression
            stats['target_stats'] = {
                'mean': float(np.mean(y)),
                'std': float(np.std(y)),
                'min': float(np.min(y)),
                'max': float(np.max(y))
            }
        
        return stats
    
    @staticmethod
    def add_outliers(X: np.ndarray, y: np.ndarray, 
                    outlier_fraction: float = 0.1,
                    outlier_factor: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add outliers to the dataset for robustness testing.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        outlier_fraction : float
            Fraction of samples to make outliers
        outlier_factor : float
            How many standard deviations away to place outliers
            
        Returns:
        --------
        X_with_outliers : np.ndarray
            Feature matrix with outliers
        y_with_outliers : np.ndarray
            Target values with outliers
        """
        n_outliers = int(len(y) * outlier_fraction)
        outlier_indices = np.random.choice(len(y), n_outliers, replace=False)
        
        X_outliers = X.copy()
        y_outliers = y.copy()
        
        # Add outliers to features
        for idx in outlier_indices:
            # Randomly choose to make it an outlier in X or y or both
            if np.random.rand() > 0.5:
                # Feature outlier
                feature_idx = np.random.randint(X.shape[1])
                X_outliers[idx, feature_idx] += outlier_factor * np.std(X[:, feature_idx]) * np.random.choice([-1, 1])
            
            if np.random.rand() > 0.5:
                # Target outlier
                y_outliers[idx] += outlier_factor * np.std(y) * np.random.choice([-1, 1])
        
        print(f"Added {n_outliers} outliers ({outlier_fraction*100:.1f}% of data)")
        
        return X_outliers, y_outliers
    
    @staticmethod
    def create_imbalanced_classification(X: np.ndarray, y: np.ndarray, 
                                       imbalance_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create an imbalanced classification dataset.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Class labels
        imbalance_ratio : float
            Ratio of minority class to majority class
            
        Returns:
        --------
        X_imbalanced : np.ndarray
            Imbalanced feature matrix
        y_imbalanced : np.ndarray
            Imbalanced class labels
        """
        classes, counts = np.unique(y, return_counts=True)
        
        if len(classes) != 2:
            print("Imbalanced data creation only supports binary classification")
            return X, y
        
        # Determine minority and majority classes
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        
        # Get indices
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # Calculate how many minority samples to keep
        n_minority = int(len(majority_indices) * imbalance_ratio)
        n_minority = max(n_minority, 10)  # Keep at least 10 samples
        
        # Sample minority class
        selected_minority = np.random.choice(minority_indices, n_minority, replace=False)
        
        # Combine indices
        selected_indices = np.concatenate([majority_indices, selected_minority])
        np.random.shuffle(selected_indices)
        
        X_imbalanced = X[selected_indices]
        y_imbalanced = y[selected_indices]
        
        # Print imbalance info
        new_counts = np.bincount(y_imbalanced)
        print(f"Created imbalanced dataset:")
        print(f"Class 0: {new_counts[0]} samples")
        print(f"Class 1: {new_counts[1]} samples")
        print(f"Imbalance ratio: {min(new_counts)/max(new_counts):.2f}")
        
        return X_imbalanced, y_imbalanced