"""
svm_baseline.py - SVM-based EEG classification baseline

This script implements a feature-engineered SVM model for EEG classification
to serve as a baseline comparison with deep learning approaches.
It follows the same train/validation splitting approach as the DL pipeline
to ensure fair comparison.

Features extracted:
- Time domain: mean, std, skew, kurtosis per channel
- Frequency domain: band powers (delta, theta, alpha, beta, gamma)
"""

import os
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import time

# For feature extraction
from scipy import signal
from scipy.stats import skew, kurtosis

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

        return FeatureDataset(self, train_epochs_idx), FeatureDataset(self, valid_epochs_idx)


def split_multi_subject_by_session(datasets: List[BasicEEGDataset], train_ratio=0.8):
    """Split multiple datasets by session indices"""
    train_ds, valid_ds = [], []
    for ds in datasets:
        tds, vds = ds.rnd_split_by_session(train_ratio)
        train_ds.append(tds)
        valid_ds.append(vds)
    
    return ConcatFeatureDataset(train_ds), ConcatFeatureDataset(valid_ds)


class FeatureDataset:
    """
    Dataset wrapper that computes engineered features from EEG epochs.
    """
    def __init__(self, dataset: BasicEEGDataset, indices=None):
        """
        Args:
            dataset: The original EEG dataset
            indices: Optional indices to select a subset of the dataset
        """
        self.dataset = dataset
        self.indices = indices if indices is not None else np.arange(len(dataset))
        self.features_cache = {}
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        
        # Return cached features if available
        if orig_idx in self.features_cache:
            return self.features_cache[orig_idx]
        
        # Get raw epoch and extract features
        epoch, label = self.dataset[orig_idx]
        features = self.extract_features(epoch)
        
        # Cache and return
        self.features_cache[orig_idx] = (features, label)
        return features, label
    
    def extract_features(self, epoch):
        """
        Extract time and frequency domain features from an EEG epoch.
        
        Args:
            epoch: EEG epoch of shape (channels, samples)
            
        Returns:
            features: 1D array of extracted features
        """
        n_channels, n_samples = epoch.shape
        sampling_rate = 128  # Assuming 128 Hz based on typical EEG
        
        # List to store all features
        all_features = []
        
        # 1. Time domain features for each channel
        for ch in range(n_channels):
            channel_data = epoch[ch, :]
            
            # Basic statistical features
            all_features.extend([
                np.mean(channel_data),  # Mean
                np.std(channel_data),   # Standard deviation
                skew(channel_data),     # Skewness
                kurtosis(channel_data)  # Kurtosis
            ])
            
        # 2. Frequency domain features for each channel
        for ch in range(n_channels):
            channel_data = epoch[ch, :]
            
            # Compute power spectral density using Welch's method
            freqs, psd = signal.welch(channel_data, fs=sampling_rate, nperseg=min(256, n_samples))
            
            # Extract band powers
            for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
                # Find frequencies in band
                idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                # Calculate band power
                band_power = np.sum(psd[idx_band])
                all_features.append(band_power)
        
        return np.array(all_features)


class ConcatFeatureDataset:
    """
    Concatenates multiple feature datasets into one.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_lengths = np.cumsum([len(ds) for ds in datasets])
        
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        # Find which dataset the index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        if dataset_idx > 0:
            idx_within_dataset = idx - self.cumulative_lengths[dataset_idx - 1]
        else:
            idx_within_dataset = idx
            
        return self.datasets[dataset_idx][idx_within_dataset]


def convert_dataset_to_arrays(dataset):
    """Convert a dataset to X and y numpy arrays for sklearn"""
    X, y = [], []
    for i in range(len(dataset)):
        features, label = dataset[i]
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)


def train_and_evaluate_svm():
    """Main function to train and evaluate the SVM model"""
    print("Loading datasets...")
    datasets = []
    for subject in tqdm(os.listdir(ROOT_DIR), desc='Loading subjects'):
        subject_h5 = os.path.join(ROOT_DIR, subject, f"{subject}_streams.h5")
        subject_meta = os.path.join(ROOT_DIR, subject, f"{subject}_meta.pckl")
        data = BasicEEGDataset(subject_h5, subject_meta, EVENTS_TO_CLS)
        datasets.append(data)
    
    print("Splitting into train/validation sets...")
    train_ds, valid_ds = split_multi_subject_by_session(datasets)
    
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
    plt.title("Confusion Matrix")
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
    plt.savefig('svm_confusion_matrix.png', dpi=300)
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
    results = train_and_evaluate_svm()
    
    # Save results to text file for inclusion in article
    with open('svm_results.txt', 'w') as f:
        f.write("SVM Baseline Results for EEG Classification\n")
        f.write("==========================================\n\n")
        f.write(f"Validation Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Best Cross-Validation Score: {results['cv_score']:.4f}\n")
        f.write(f"Best Parameters: {results['best_params']}\n")
        f.write(f"Training Time: {results['training_time']:.2f} seconds\n")