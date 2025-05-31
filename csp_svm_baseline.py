"""
csp_svm_baseline.py - CSP+SVM-based EEG classification baseline

This script implements a Common Spatial Pattern (CSP) preprocessing pipeline
followed by an SVM classifier for EEG classification. CSP is a well-established
technique for enhancing discriminative spatial information in EEG signals.

Features extracted:
- CSP spatial features from filtered EEG signals
- Time domain: mean, std, skew, kurtosis on CSP-transformed signals
- Frequency domain: band powers (delta, theta, alpha, beta, gamma) on CSP signals
"""

import os
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import time

# For feature extraction and CSP
from scipy import signal
from scipy.stats import skew, kurtosis
from scipy.linalg import eigh

# For machine learning
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# For reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Constants - adjust paths as needed
ROOT_DIR = 'C:/Code/Uvicorn-article/EEG_competition/Train_data'
TEST_DIR = 'C:/Code/Uvicorn-article/EEG_competition/Test_data'
EVENTS_TO_CLS = {'left': 0, 'right': 1, 'left-right': 2, 'nothing': 3}

# EEG frequency bands (Hz)
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

class CSPFilter:
    """
    Common Spatial Pattern (CSP) filter implementation.
    
    CSP is a spatial filtering technique that maximizes the variance of one class
    while minimizing the variance of another class.
    """
    
    def __init__(self, n_components: int = 6):
        """
        Args:
            n_components: Number of CSP components to keep (should be even)
        """
        self.n_components = n_components
        self.filters_ = None
        self.patterns_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CSPFilter':
        """
        Fit CSP filters on training data.
        
        Args:
            X: EEG data of shape (n_trials, n_channels, n_samples)
            y: Labels of shape (n_trials,)
            
        Returns:
            self
        """
        n_trials, n_channels, n_samples = X.shape
        
        # Calculate covariance matrices for each class
        classes = np.unique(y)
        if len(classes) != 2:
            # For multi-class, use one-vs-rest approach
            # Here we'll use the first two classes for CSP
            classes = classes[:2]
            
        cov_1 = np.zeros((n_channels, n_channels))
        cov_2 = np.zeros((n_channels, n_channels))
        
        # Class 1 covariance
        X_1 = X[y == classes[0]]
        for trial in X_1:
            cov_1 += np.cov(trial)
        cov_1 /= len(X_1)
        
        # Class 2 covariance
        X_2 = X[y == classes[1]]
        for trial in X_2:
            cov_2 += np.cov(trial)
        cov_2 /= len(X_2)
        
        # Solve generalized eigenvalue problem
        eigenvals, eigenvecs = eigh(cov_1, cov_1 + cov_2)
        
        # Sort by eigenvalues (descending)
        ix = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[ix]
        eigenvecs = eigenvecs[:, ix]
        
        # Select most discriminative components
        # Take n_components/2 from each end (highest and lowest eigenvalues)
        n_select = self.n_components // 2
        filters = np.concatenate([eigenvecs[:, :n_select], 
                                eigenvecs[:, -n_select:]], axis=1)
        
        self.filters_ = filters.T
        self.patterns_ = np.linalg.pinv(self.filters_)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply CSP transformation to data.
        
        Args:
            X: EEG data of shape (n_trials, n_channels, n_samples)
            
        Returns:
            X_csp: CSP-transformed data of shape (n_trials, n_components, n_samples)
        """
        if self.filters_ is None:
            raise ValueError("CSP filters not fitted. Call fit() first.")
            
        n_trials, n_channels, n_samples = X.shape
        X_csp = np.zeros((n_trials, self.n_components, n_samples))
        
        for i, trial in enumerate(X):
            X_csp[i] = np.dot(self.filters_, trial)
            
        return X_csp

class BasicEEGDataset:
    """
    Dataset class for EEG data loading.
    Matches the implementation in dataloader.ipynb.
    """
    def __init__(self, subject_h5: str, subject_meta: str, events_to_cls: Dict[str, int]):
        """
        Args:
            subject_h5: Path to subject's h5 file
            subject_meta: Path to subject's meta file
            events_to_cls: Dictionary mapping event names to class indices
        """
        with open(subject_meta, 'rb') as f:
            meta_data = pickle.load(f)

        self.data = h5py.File(subject_h5, "r")
        epochs = self.data["epochs_on_task"][:]
        on_task_events = self.data["on_task_events"][:][:, 2]
        task_event_ids = meta_data["task_event_ids"]

        relevant_epochs = np.logical_or.reduce([on_task_events == task_event_ids[ev]
                                        for ev in events_to_cls.keys()])
        self.epochs = epochs[relevant_epochs, ...]

        event_id_to_cls = {task_event_ids[ev]: cls for ev, cls in events_to_cls.items()}
        self.events_cls = list(map(lambda e: event_id_to_cls[e], on_task_events[relevant_epochs]))
        self.session_idx = self.data["on_task_session_idx"][relevant_epochs]
        assert len(self.epochs) == len(self.events_cls), "Epochs and event classes must match in size"
    
    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, index):
        return self.epochs[index], self.events_cls[index]

    def rnd_split_by_session(self, train_ratio=0.8, train_session_idx=None, valid_session_idx=None):
        """Split dataset by session indices"""
        if train_session_idx is None or valid_session_idx is None:
            uniq_session_idx = np.unique(self.session_idx)
            nvalid_session = int(len(uniq_session_idx) * (1 - train_ratio))
            nvalid_session = max(1, nvalid_session)
            ntrain_session = len(uniq_session_idx) - nvalid_session

            # Use fixed seed for reproducibility 
            rng = np.random.RandomState(RANDOM_SEED)
            rnd_session_idx = rng.permutation(uniq_session_idx)
            train_session_idx = rnd_session_idx[:ntrain_session]
            valid_session_idx = rnd_session_idx[ntrain_session:]

        train_epochs_idx = np.logical_or.reduce([self.session_idx == i for i in train_session_idx], axis=0)
        train_epochs_idx = np.arange(self.epochs.shape[0])[train_epochs_idx]
        valid_epochs_idx = np.logical_or.reduce([self.session_idx == i for i in valid_session_idx], axis=0)
        valid_epochs_idx = np.arange(self.epochs.shape[0])[valid_epochs_idx]

        return train_epochs_idx, valid_epochs_idx

class CSPFeatureDataset:
    """
    Dataset wrapper that computes CSP-based features from EEG epochs.
    """
    def __init__(self, dataset: BasicEEGDataset, indices=None):
        """
        Args:
            dataset: The original EEG dataset
            indices: Optional indices to select a subset of the dataset
        """
        self.dataset = dataset
        self.indices = indices if indices is not None else list(range(len(dataset)))
        self.csp_filter = None
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Get CSP-based features for a single epoch"""
        real_idx = self.indices[idx]
        epoch, label = self.dataset[real_idx]
        
        # If CSP filter is fitted, apply it
        if self.csp_filter is not None:
            epoch = self.csp_filter.transform(epoch.reshape(1, *epoch.shape))[0]
            
        features = self.extract_csp_features(epoch)
        return features, label

    def fit_csp(self, other_datasets=None):
        """
        Fit CSP filter on this dataset (and optionally others).
        
        Args:
            other_datasets: Other CSPFeatureDataset objects to include in CSP fitting
        """
        # Collect all epochs and labels for CSP fitting
        all_epochs = []
        all_labels = []
        
        # Add epochs from this dataset
        for idx in self.indices:
            epoch, label = self.dataset[idx]
            all_epochs.append(epoch)
            all_labels.append(label)
            
        # Add epochs from other datasets if provided
        if other_datasets:
            for ds in other_datasets:
                for idx in ds.indices:
                    epoch, label = ds.dataset[idx]
                    all_epochs.append(epoch)
                    all_labels.append(label)
        
        X = np.array(all_epochs)
        y = np.array(all_labels)
        
        # Fit CSP filter
        self.csp_filter = CSPFilter(n_components=6)
        self.csp_filter.fit(X, y)
        
        # Share the fitted filter with other datasets
        if other_datasets:
            for ds in other_datasets:
                ds.csp_filter = self.csp_filter

    def extract_csp_features(self, epoch):
        """
        Extract features from a CSP-transformed EEG epoch.
        
        Args:
            epoch: CSP-transformed EEG epoch of shape (csp_components, samples)
            
        Returns:
            features: 1D array of extracted features
        """
        features = []
        
        # Sampling frequency (assumed 250 Hz, adjust if needed)
        fs = 250
        
        # Time domain features for each CSP component
        for ch in range(epoch.shape[0]):
            ch_data = epoch[ch, :]
            
            # Basic statistics
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                skew(ch_data),
                kurtosis(ch_data)
            ])
            
            # Log variance (common CSP feature)
            features.append(np.log(np.var(ch_data) + 1e-8))
        
        # Frequency domain features for each CSP component
        for ch in range(epoch.shape[0]):
            ch_data = epoch[ch, :]
            
            # Compute power spectral density
            freqs, psd = signal.welch(ch_data, fs=fs, nperseg=min(len(ch_data), 256))
            
            # Extract band powers
            for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.mean(psd[band_mask])
                features.append(band_power)
        
        return np.array(features, dtype=np.float32)

class ConcatFeatureDataset:
    """Concatenates multiple feature datasets into one."""
    
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_lengths = np.cumsum([len(ds) for ds in datasets])
    
    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths.size > 0 else 0
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        
        # Calculate local index within that dataset
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        
        return self.datasets[dataset_idx][local_idx]

def split_multi_subject_by_session(datasets: List[BasicEEGDataset], train_ratio=0.8):
    """Split multiple datasets by session indices"""
    train_datasets = []
    valid_datasets = []
    for data in datasets:
        train_idx, valid_idx = data.rnd_split_by_session(train_ratio)
        train_datasets.append(CSPFeatureDataset(data, train_idx))
        valid_datasets.append(CSPFeatureDataset(data, valid_idx))
    
    return ConcatFeatureDataset(train_datasets), ConcatFeatureDataset(valid_datasets)

def convert_dataset_to_arrays(dataset):
    """Convert a dataset to X and y numpy arrays for sklearn"""
    X, y = [], []
    for i in range(len(dataset)):
        features, label = dataset[i]
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

def train_and_evaluate_csp_svm():
    """Main function to train and evaluate the CSP+SVM model"""
    
    print("Loading datasets...")
    datasets = []
    for subject in tqdm(os.listdir(ROOT_DIR), desc='Loading subjects'):
        subject_h5 = os.path.join(ROOT_DIR, subject, f"{subject}_streams.h5")
        subject_meta = os.path.join(ROOT_DIR, subject, f"{subject}_meta.pckl")
        data = BasicEEGDataset(subject_h5, subject_meta, EVENTS_TO_CLS)
        datasets.append(data)
    
    print("Splitting into train/validation sets...")
    train_ds, valid_ds = split_multi_subject_by_session(datasets)
    
    print("Fitting CSP filters on training data...")
    # Fit CSP on training data only
    if hasattr(train_ds.datasets[0], 'fit_csp'):
        train_ds.datasets[0].fit_csp(train_ds.datasets[1:])
        
        # Apply the same CSP filter to validation datasets
        for valid_dataset in valid_ds.datasets:
            valid_dataset.csp_filter = train_ds.datasets[0].csp_filter
    
    print("Converting datasets to numpy arrays...")
    X_train, y_train = convert_dataset_to_arrays(train_ds)
    X_valid, y_valid = convert_dataset_to_arrays(valid_ds)
    
    print(f"Training set: {X_train.shape}, Validation set: {X_valid.shape}")
    
    # Define hyperparameter grid for search
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }
    
    # Define model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))
    ])
    
    # Create cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Perform grid search
    print("Performing grid search for hyperparameter tuning...")
    start_time = time.time()
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Best model results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Evaluate on validation set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_valid)
    
    # Print evaluation metrics
    acc = accuracy_score(y_valid, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_valid, y_pred, 
                               target_names=list(EVENTS_TO_CLS.keys())))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("CSP+SVM Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(EVENTS_TO_CLS))
    plt.xticks(tick_marks, list(EVENTS_TO_CLS.keys()), rotation=45)
    plt.yticks(tick_marks, list(EVENTS_TO_CLS.keys()))
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('csp_svm_confusion_matrix.png', dpi=300)
    plt.show()
    
    # Return results for further analysis
    return {
        'accuracy': acc,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'training_time': training_time,
        'model': best_model
    }


if __name__ == "__main__":
    results = train_and_evaluate_csp_svm()
    
    # Save results to text file for inclusion in article
    with open('csp_svm_results.txt', 'w') as f:
        f.write("CSP+SVM Baseline Results for EEG Classification\n")
        f.write("===============================================\n\n")
        f.write(f"Validation Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Best Cross-Validation Score: {results['cv_score']:.4f}\n")
        f.write(f"Best Parameters: {results['best_params']}\n")
        f.write(f"Training Time: {results['training_time']:.2f} seconds\n")
        f.write("\nMethod: Common Spatial Pattern (CSP) + Support Vector Machine (SVM)\n")
        f.write("CSP Components: 6 (3 highest + 3 lowest eigenvalues)\n")
        f.write("Features: CSP log-variance, time-domain stats, and frequency band powers\n")
