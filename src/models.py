import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from sklearn.svm import OneClassSVM, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class SignatureVerifier:
    """
    Class for signature verification.
    
    Supported models:
    - 'svm': Binary SVM (requires genuine + forgery for training)
    - 'one_class_svm': One-Class SVM (genuine only)
    - 'random_forest': Random Forest (requires genuine + forgery)
    - 'distance': Euclidean distance to references (genuine only)
    
    Usage:
        verifier = SignatureVerifier(model_type='svm')
        verifier.train(X_genuine, X_forgery)
        scores = verifier.predict(X_test)
    """
    
    SUPPORTED_MODELS = ['svm', 'one_class_svm', 'random_forest', 'distance']
    
    def __init__(self, model_type: str = 'svm', **kwargs):
        """
        Args:
            model_type: 'svm', 'one_class_svm', 'random_forest', or 'distance'
            **kwargs: Model parameters (kernel, C, nu, n_estimators, etc.)
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.reference_features = None
        
        # Default parameters
        self.params = {
            'svm': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
            'one_class_svm': {'kernel': 'rbf', 'nu': 0.1, 'gamma': 'scale'},
            'random_forest': {'n_estimators': 100, 'max_depth': None, 'random_state': 42},
            'distance': {'aggregation': 'mean'}
        }.get(model_type, {})
        self.params.update(kwargs)
    
    def train(self, X_genuine: np.ndarray, X_forgery: Optional[np.ndarray] = None) -> 'SignatureVerifier':
        """
        Trains the model.
        
        Args:
            X_genuine: Features of genuine signatures (n_samples, n_features)
                       Output from Task B: extract_global_features()
            X_forgery: Features of forgeries (required for 'svm' and 'random_forest')
        
        Returns:
            self
        """
        X_genuine = np.array(X_genuine)
        print(f"Training {self.model_type} model...")
        print(f"  Genuine samples: {len(X_genuine)}, Features: {X_genuine.shape[1]}")
        
        if self.model_type == 'distance':
            self.reference_features = X_genuine.copy()
            self.is_fitted = True
            print(f"  Stored {len(self.reference_features)} reference signatures")
            return self
        
        elif self.model_type == 'one_class_svm':
            X_scaled = self.scaler.fit_transform(X_genuine)
            self.model = OneClassSVM(
                kernel=self.params.get('kernel', 'rbf'),
                nu=self.params.get('nu', 0.1),
                gamma=self.params.get('gamma', 'scale')
            )
            self.model.fit(X_scaled)
            self.is_fitted = True
            print("  One-Class SVM trained")
            return self
        
        else:  # svm, random_forest
            if X_forgery is None:
                raise ValueError(f"X_forgery is required for '{self.model_type}'")
            
            X_forgery = np.array(X_forgery)
            print(f"  Forgery samples: {len(X_forgery)}")
            
            X = np.vstack([X_genuine, X_forgery])
            y = np.array([1] * len(X_genuine) + [0] * len(X_forgery))
            X_scaled = self.scaler.fit_transform(X)
            
            if self.model_type == 'svm':
                self.model = SVC(
                    kernel=self.params.get('kernel', 'rbf'),
                    C=self.params.get('C', 1.0),
                    gamma=self.params.get('gamma', 'scale'),
                    probability=True
                )
            else:  # random_forest
                self.model = RandomForestClassifier(
                    n_estimators=self.params.get('n_estimators', 100),
                    max_depth=self.params.get('max_depth', None),
                    random_state=self.params.get('random_state', 42)
                )
            
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            print(f"  Model trained on {len(X)} samples")
            return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts dissimilarity scores.
        
        Args:
            X_test: Features of signatures to verify (n_samples, n_features)
        
        Returns:
            scores: Low score = genuine, high score = forgery
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        if self.model_type == 'distance':
            scores = []
            for x in X_test:
                distances = [np.linalg.norm(x - ref) for ref in self.reference_features]
                if self.params.get('aggregation') == 'min':
                    scores.append(np.min(distances))
                else:
                    scores.append(np.mean(distances))
            return np.array(scores)
        
        elif self.model_type == 'one_class_svm':
            X_scaled = self.scaler.transform(X_test)
            decision = self.model.decision_function(X_scaled)
            return 1 / (1 + np.exp(decision))  # Sigmoid to normalize
        
        else:  # svm, random_forest
            X_scaled = self.scaler.transform(X_test)
            proba = self.model.predict_proba(X_scaled)
            return 1 - proba[:, 1]  # 1 - P(genuine) = dissimilarity
    
    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"SignatureVerifier('{self.model_type}', {status})"


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def compute_far_frr(genuine_scores: np.ndarray, forgery_scores: np.ndarray, threshold: float) -> Tuple[float, float]:
    """
    Computes FAR and FRR for a given threshold.
    
    Convention: score < threshold -> accepted (genuine)
                score >= threshold -> rejected (forgery)
    
    Returns:
        (FAR, FRR)
    """
    frr = np.mean(genuine_scores >= threshold) if len(genuine_scores) > 0 else 0.0
    far = np.mean(forgery_scores < threshold) if len(forgery_scores) > 0 else 0.0
    return far, frr


def compute_eer(genuine_scores: np.ndarray, forgery_scores: np.ndarray, num_thresholds: int = 1000) -> Tuple[float, float]:
    """
    Computes the Equal Error Rate (EER).
    
    EER is the point where FAR = FRR.
    
    Returns:
        (EER, threshold_at_EER)
    """
    all_scores = np.concatenate([genuine_scores, forgery_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)
    
    fars = np.array([compute_far_frr(genuine_scores, forgery_scores, t)[0] for t in thresholds])
    frrs = np.array([compute_far_frr(genuine_scores, forgery_scores, t)[1] for t in thresholds])
    
    eer_index = np.argmin(np.abs(fars - frrs))
    eer = (fars[eer_index] + frrs[eer_index]) / 2
    
    return eer, thresholds[eer_index]


def compute_metrics(genuine_scores: np.ndarray, forgery_scores: np.ndarray) -> Dict:
    """
    Computes all evaluation metrics.
    
    Returns:
        Dict with eer, threshold, far, frr, etc.
    """
    eer, threshold = compute_eer(genuine_scores, forgery_scores)
    far, frr = compute_far_frr(genuine_scores, forgery_scores, threshold)
    
    return {
        'eer': eer,
        'eer_threshold': threshold,
        'far_at_eer': far,
        'frr_at_eer': frr,
        'genuine_mean': genuine_scores.mean(),
        'forgery_mean': forgery_scores.mean(),
        'n_genuine': len(genuine_scores),
        'n_forgery': len(forgery_scores)
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_score_distributions(genuine_scores: np.ndarray, forgery_scores: np.ndarray, 
                            threshold: Optional[float] = None, title: str = "Score Distributions"):
    """Plots the distributions of genuine and forgery scores."""
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_scores, bins=50, alpha=0.6, color='green',
             label=f'Genuine (n={len(genuine_scores)})', density=True)
    plt.hist(forgery_scores, bins=50, alpha=0.6, color='red',
             label=f'Forgery (n={len(forgery_scores)})', density=True)
    if threshold is not None:
        plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.3f}')
    plt.xlabel('Dissimilarity Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_det_curve(genuine_scores: np.ndarray, forgery_scores: np.ndarray, title: str = "DET Curve"):
    """Plots the DET (Detection Error Tradeoff) curve."""
    all_scores = np.concatenate([genuine_scores, forgery_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 1000)
    
    fars = [compute_far_frr(genuine_scores, forgery_scores, t)[0] for t in thresholds]
    frrs = [compute_far_frr(genuine_scores, forgery_scores, t)[1] for t in thresholds]
    eer, _ = compute_eer(genuine_scores, forgery_scores)
    
    plt.figure(figsize=(8, 8))
    plt.plot(np.array(fars)*100, np.array(frrs)*100, 'b-', linewidth=2, label='DET Curve')
    plt.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='FAR = FRR')
    plt.plot(eer*100, eer*100, 'ro', markersize=10, label=f'EER = {eer*100:.2f}%')
    plt.xlabel('False Acceptance Rate (%)')
    plt.ylabel('False Rejection Rate (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 50])
    plt.ylim([0, 50])
    plt.show()


def plot_far_frr_vs_threshold(genuine_scores: np.ndarray, forgery_scores: np.ndarray, 
                              title: str = "FAR/FRR vs Threshold"):
    """Plots FAR and FRR as a function of threshold."""
    all_scores = np.concatenate([genuine_scores, forgery_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 1000)
    
    fars = [compute_far_frr(genuine_scores, forgery_scores, t)[0] for t in thresholds]
    frrs = [compute_far_frr(genuine_scores, forgery_scores, t)[1] for t in thresholds]
    eer, eer_thresh = compute_eer(genuine_scores, forgery_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, np.array(fars)*100, 'r-', linewidth=2, label='FAR')
    plt.plot(thresholds, np.array(frrs)*100, 'b-', linewidth=2, label='FRR')
    plt.axvline(x=eer_thresh, color='green', linestyle='--', label=f'EER threshold = {eer_thresh:.3f}')
    plt.plot(eer_thresh, eer*100, 'go', markersize=10, label=f'EER = {eer*100:.2f}%')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# =============================================================================
# MAIN PIPELINE FUNCTIONS
# =============================================================================

def run_verification(
    X_genuine_train: np.ndarray,
    X_forgery_train: np.ndarray,
    X_genuine_test: np.ndarray,
    X_forgery_test: np.ndarray,
    model_type: str = 'svm',
    show_plots: bool = True
) -> Dict:
    """
    Main function to run signature verification.
    Returns:
        Dict with metrics (EER, threshold, etc.)
    """
    print(f"\n{'='*60}")
    print(f"SIGNATURE VERIFICATION: {model_type.upper()}")
    print(f"{'='*60}")
    print(f"  Train: {len(X_genuine_train)} genuine, {len(X_forgery_train)} forgery")
    print(f"  Test: {len(X_genuine_test)} genuine, {len(X_forgery_test)} forgery")
    print(f"  Features: {X_genuine_train.shape[1]}")
    
    # Create and train model
    verifier = SignatureVerifier(model_type=model_type)
    
    if model_type in ['svm', 'random_forest']:
        verifier.train(X_genuine_train, X_forgery_train)
    else:  # one_class_svm, distance
        verifier.train(X_genuine_train)
    
    # Predict
    genuine_scores = verifier.predict(X_genuine_test)
    forgery_scores = verifier.predict(X_forgery_test)
    
    # Compute metrics
    metrics = compute_metrics(genuine_scores, forgery_scores)
    
    print(f"\nResults:")
    print(f"  EER: {metrics['eer']*100:.2f}%")
    print(f"  Threshold: {metrics['eer_threshold']:.4f}")
    print(f"  Genuine mean: {metrics['genuine_mean']:.4f}")
    print(f"  Forgery mean: {metrics['forgery_mean']:.4f}")
    
    # Visualizations
    if show_plots:
        plot_score_distributions(genuine_scores, forgery_scores,
                                threshold=metrics['eer_threshold'],
                                title=f"Score Distributions ({model_type})")
        plot_det_curve(genuine_scores, forgery_scores,
                      title=f"DET Curve ({model_type}) - EER: {metrics['eer']*100:.2f}%")
    
    return metrics


def compare_models(
    X_genuine_train: np.ndarray,
    X_forgery_train: np.ndarray,
    X_genuine_test: np.ndarray,
    X_forgery_test: np.ndarray,
    show_plots: bool = True
) -> Dict[str, Dict]:
    """
    Compares all available models.
    
    Returns:
        Dict {model_name: metrics}
    """
    results = {}
    
    for model_type in ['distance', 'svm', 'one_class_svm', 'random_forest']:
        results[model_type] = run_verification(
            X_genuine_train, X_forgery_train,
            X_genuine_test, X_forgery_test,
            model_type=model_type,
            show_plots=show_plots
        )
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'EER':>10} {'Threshold':>15}")
    print("-" * 45)
    for model, m in results.items():
        print(f"{model:<20} {m['eer']*100:>9.2f}% {m['eer_threshold']:>15.4f}")
    
    best = min(results, key=lambda x: results[x]['eer'])
    print(f"\nBest model: {best} (EER: {results[best]['eer']*100:.2f}%)")
    
    return results
