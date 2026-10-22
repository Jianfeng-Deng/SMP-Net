import random
from typing import List, Optional

import numpy as np

from signalAnalysis.signalExtract import extract_features, run_pipeline, run_pipeline_tran


class dataProcess():
    def __init__(self, cwru_data, cwru_labels, pu_data, pu_labels, hust_data, hust_labels):
        self.cwru_data = cwru_data
        self.cwru_labels = cwru_labels
        self.pu_data = pu_data
        self.pu_labels = pu_labels
        self.hust_data = hust_data
        self.hust_labels = hust_labels
        self.cwru_original = None
        self.pu_original = None
        self.hust_original = None
        
        # 保存 LDA 相关的权重和缩放器
        self.W = None
        self.scaler = None

    def _copy_origin_data_(self):
        return self.cwru_data.copy(), self.pu_data.copy(), self.hust_data.copy()

    def _signal_feature_extract_(self, data_list, fs):
        """通用的特征提取函数"""
        from signalAnalysis.signalExtract import extract_features
        return np.array([extract_features(sig, fs) for sig in data_list], dtype=np.float32)

    def _signal_feature_extract_all_(self):
        cwru_feat = self._signal_feature_extract_(self.cwru_data, 12000)
        pu_feat = self._signal_feature_extract_(self.pu_data, 64000)
        hust_feat = self._signal_feature_extract_(self.hust_data, 25600)
        return cwru_feat, pu_feat, hust_feat

    def _lda_train_(self, X_feat_all, y_all, domain, n_component, lambda_domain):
        X_lda_all, W, scaler = run_pipeline(
            X=X_feat_all,
            y=y_all,
            domain=domain,
            n_components=n_component,
            lambda_domain=lambda_domain)
        
        # 记录到实例中
        self.W = W
        self.scaler = scaler

        return X_lda_all, W, scaler

    def _lda_trans_(self, cwru_feat, pu_feat, hust_feat, W, scaler):
        # 关键：所有数据集（包括训练集和测试集）都用同一个训练时fit的scaler
        cwru_data = run_pipeline_tran(cwru_feat, W, scaler)
        pu_data = run_pipeline_tran(pu_feat, W, scaler)
        hust_data = run_pipeline_tran(hust_feat, W, scaler)

        return cwru_data, pu_data, hust_data

    def fit(self, n_component=2, lambda_domain=0.15):
        cwru_origin, pu_origin, hust_origin = self._copy_origin_data_()
        cwru_feat, pu_feat, hust_feat = self._signal_feature_extract_all_()

        # 只在训练集（CWRU + PU）上训练LDA和scaler
        X_feat_all = np.concatenate([cwru_feat, pu_feat], axis=0)
        y_all = np.concatenate([self.cwru_labels, self.pu_labels], axis=0)

        domains_all = np.concatenate([
            np.zeros(len(self.cwru_data)),  # CWRU = 0
            np.ones(len(self.pu_data))  # PU   = 1
        ])

        # 训练LDA（返回scaler）
        X_lda_all, W, scaler = self._lda_train_(X_feat_all, y_all, domains_all, n_component, lambda_domain)

        # 使用训练好的scaler和W转换所有数据（包括HUST测试集）
        cwru_data, pu_data, hust_data = self._lda_trans_(cwru_feat, pu_feat, hust_feat, W, scaler)

        return cwru_origin, pu_origin, hust_origin, cwru_data, pu_data, hust_data

    def transform_new_data(self, raw_signals, fs):
        """
        对新的测试集（如 HIT）应用同样的预处理流水线
        """
        if self.W is None or self.scaler is None:
            raise RuntimeError("LDA weights not initialized. Call fit() first.")
        
        # 检查输入数据
        if raw_signals is None or len(raw_signals) == 0:
            raise ValueError("raw_signals is empty or None")
        
        # 确保 raw_signals 是列表或数组
        if isinstance(raw_signals, np.ndarray):
            if raw_signals.ndim == 2:
                # 如果是一维数组的数组，转换为列表
                raw_signals = [raw_signals[i] for i in range(len(raw_signals))]
            elif raw_signals.ndim == 1:
                raw_signals = [raw_signals]
        
        # 1. 提取特征
        print(f"[HIT] Extracting features from {len(raw_signals)} signals...")
        feats = self._signal_feature_extract_(raw_signals, fs)
        
        # 检查特征提取结果
        if feats is None or len(feats) == 0:
            raise ValueError(f"Feature extraction failed: extracted {len(feats) if feats is not None else 0} features from {len(raw_signals)} signals")
        
        if feats.ndim == 1:
            raise ValueError(f"Expected 2D feature array, got 1D array with shape {feats.shape}. Check extract_features function.")
        
        print(f"[HIT] Extracted features shape: {feats.shape}")
        
        # 2. 应用 LDA 转换
        transformed = run_pipeline_tran(feats, self.W, self.scaler)
        return transformed

    """==============================================prompt转换======================================================="""

    def _append_operating_condition_(self, text: str, operating_condition: Optional[str]) -> str:
        if operating_condition is None:
            return text
        return f"{text} under {operating_condition}."

    def _label_to_text_(self, label: int, FAULT_PROMPT_TEMPLATES, operating_condition: Optional[str] = None) -> str:
        if label not in FAULT_PROMPT_TEMPLATES:
            raise ValueError(f"Unknown fault label: {label}")

        base_text = random.choice(FAULT_PROMPT_TEMPLATES[label])
        return self._append_operating_condition_(base_text, operating_condition)

    def _labels_to_texts_(self, labels: np.ndarray, FAULT_PROMPT_TEMPLATES, operating_conditions: Optional[List[Optional[str]]] = None) -> List[str]:

        if operating_conditions is None:
            operating_conditions = [None] * len(labels)

        if len(labels) != len(operating_conditions):
            raise ValueError("labels and operating_conditions must have the same length")

        return [
            self._label_to_text_(label=int(lbl), FAULT_PROMPT_TEMPLATES=FAULT_PROMPT_TEMPLATES, operating_condition=cond)
            for lbl, cond in zip(labels, operating_conditions)
        ]

    def data_label_to_prompt(self, FAULT_PROMPT_TEMPLATES, hit_labels=None):
        cwru_text = self._labels_to_texts_(labels=self.cwru_labels, FAULT_PROMPT_TEMPLATES=FAULT_PROMPT_TEMPLATES)
        pu_text = self._labels_to_texts_(labels=self.pu_labels, FAULT_PROMPT_TEMPLATES=FAULT_PROMPT_TEMPLATES)
        hust_text = self._labels_to_texts_(labels=self.hust_labels, FAULT_PROMPT_TEMPLATES=FAULT_PROMPT_TEMPLATES)
        
        hit_text = None
        if hit_labels is not None:
            hit_text = self._labels_to_texts_(labels=hit_labels, FAULT_PROMPT_TEMPLATES=FAULT_PROMPT_TEMPLATES)

        if hit_text is not None:
            return cwru_text, pu_text, hust_text, hit_text
        else:
            return cwru_text, pu_text, hust_text