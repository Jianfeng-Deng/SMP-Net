import numpy as np
import warnings
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler

# ==========================================================
# 特征名称（与提取顺序严格一致）
# ==========================================================
FEATURE_NAMES = [
    "Time_Mean", "Time_Std", "Time_RMS",
    "Time_Max", "Time_Min", "Time_PeakToPeak",
    "Time_Skewness", "Time_Kurtosis",

    "Spec_Mean", "Spec_Std", "Spec_RMS",
    "Spec_Max", "Spec_Min",
    "Spec_Variance", "Spec_Skewness", "Spec_Kurtosis",
    "Spec_Centroid", "Spec_Spread",
    "Spec_Entropy", "Spec_Rolloff",

    "BandEnergy_0_10Hz",
    "BandEnergy_10_30Hz",
    "BandEnergy_30_60Hz",
    "BandEnergy_60_100Hz"
]

def extract_features(x, fs):
    # [ATT95] Silence RuntimeWarnings (Overflow/Precision Loss) 
    # These are expected for high-energy synthetic signals and are handled by the nan_to_num output patch.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return _extract_features_unsafe(x, fs)

def _extract_features_unsafe(x, fs):
    # ---------- 时域 ----------
    time_feats = [
        np.mean(x), np.std(x), np.sqrt(np.mean(x**2)),
        np.max(x), np.min(x), np.ptp(x),
        skew(x), kurtosis(x)
    ]

    # ---------- 频域 ----------
    spec = np.abs(rfft(x))
    freqs = rfftfreq(len(x), 1/fs)
    spec_norm = spec / (np.sum(spec) + 1e-12)

    centroid = np.sum(freqs * spec_norm)
    spread   = np.sqrt(np.sum((freqs-centroid)**2 * spec_norm))
    entropy  = -np.sum(spec_norm*np.log(spec_norm+1e-12))
    rolloff  = freqs[np.searchsorted(np.cumsum(spec_norm), 0.85)]

    freq_feats = [
        np.mean(spec), np.std(spec), np.sqrt(np.mean(spec**2)),
        np.max(spec), np.min(spec),
        np.var(spec), skew(spec), kurtosis(spec),
        centroid, spread, entropy, rolloff
    ]

    def band_energy(f1, f2):
        return np.sum(spec[(freqs>=f1)&(freqs<f2)]**2)

    band_feats = [
        band_energy(0,10),
        band_energy(10,30),
        band_energy(30,60),
        band_energy(60,100)
    ]

    # [Robustness] Clean NaNs/Infs that might arise from constant signals (div by zero in skew/kurtosis)
    min_val = np.finfo(np.float32).min
    max_val = np.finfo(np.float32).max
    
    feats = np.hstack([time_feats, freq_feats, band_feats])
    feats = np.nan_to_num(feats, nan=0.0, posinf=max_val, neginf=min_val)
    
    return feats

# ==========================================================
# LDA 损失重构
# ==========================================================
def compute_between_class_scatter(X, y):
    classes = np.unique(y)
    mean_total = X.mean(axis=0)
    S_B = np.zeros((X.shape[1], X.shape[1]))

    for c in classes:
        Xc = X[y == c]
        mean_c = Xc.mean(axis=0)
        n_c = Xc.shape[0]
        diff = (mean_c - mean_total).reshape(-1,1)
        S_B += n_c * diff @ diff.T

    return S_B

def compute_within_class_scatter(X, y):
    classes = np.unique(y)
    S_W = np.zeros((X.shape[1], X.shape[1]))

    for c in classes:
        Xc = X[y == c]
        mean_c = Xc.mean(axis=0)
        diff = Xc - mean_c
        S_W += diff.T @ diff

    return S_W

def compute_domain_scatter(X, y, domains):
    classes = np.unique(y)
    domain_ids = np.unique(domains)

    D = X.shape[1]
    S_domain = np.zeros((D, D))

    for c in classes:
        Xc = X[y == c]
        dc = domains[y == c]

        # 类跨域均值
        mu_c = Xc.mean(axis=0)

        for d in domain_ids:
            X_cd = Xc[dc == d]
            if len(X_cd) == 0:
                continue
            mu_cd = X_cd.mean(axis=0)
            diff = (mu_cd - mu_c).reshape(-1,1)
            S_domain += diff @ diff.T

    return S_domain

def domain_regularized_lda(X, y, domains, n_components, lambda_domain=0.3):
    S_B = compute_between_class_scatter(X, y)
    S_W = compute_within_class_scatter(X, y)
    S_D = compute_domain_scatter(X, y, domains)

    S_total = S_W + lambda_domain * S_D

    # 解广义特征值问题
    eigvals, eigvecs = np.linalg.eig(
        np.linalg.pinv(S_total) @ S_B
    )

    # 按特征值排序
    idx = np.argsort(eigvals.real)[::-1]
    W = eigvecs[:, idx[:n_components]].real

    return X @ W, W

# ==========================================================
# 训练流程
# ==========================================================
def run_pipeline(X, y, domain, n_components, top_k=5, lambda_domain=0.3):
    """
    训练LDA pipeline（仅在训练集上fit scaler）
    
    Args:
        X: (N, D) 特征矩阵（应该是已经提取好的统计特征，不是原始波形）
        y: (N,) 标签
        domain: (N,) 域ID
        n_components: LDA降维后的维度
        top_k: 打印top-k特征（用于可视化，可选）
        lambda_domain: 域正则化系数
    
    Returns:
        X_lda: (N, n_components) LDA降维后的特征
        W: (D, n_components) LDA投影矩阵
        scaler: 训练好的StandardScaler（必须保存并在推理时复用）
    """
    # 注意：X 应该是已经提取好的特征（由调用方通过 extract_features 提取）
    # 如果传入的是原始波形，需要先提取特征（但当前调用链已经在外部做了）
    
    # 只在训练集上 fit scaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 使用域正则化LDA
    X_lda, W = domain_regularized_lda(
        Xs, y, domain,
        n_components=n_components,
        lambda_domain=lambda_domain
    )

    return X_lda, W, scaler

def run_pipeline_tran(X, W, scaler):
    """
    使用训练好的scaler和LDA投影矩阵转换新数据（推理阶段）
    
    Args:
        X: (N, D) 特征矩阵（应该已经通过 extract_features 提取）
        W: (D, n_components) 训练好的LDA投影矩阵
        scaler: 训练好的StandardScaler（必须用训练时的scaler，不能重新fit）
    
    Returns:
        X_lda: (N, n_components) LDA降维后的特征
    """
    # 关键：只用 transform，不能 fit_transform（避免数据泄漏）
    Xs = scaler.transform(X)
    
    return Xs @ W