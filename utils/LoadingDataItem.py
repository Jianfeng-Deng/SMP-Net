import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat


class LoadData:
    def __init__(self, filepath: list[str]):
        """
        初始化LoadData类
        
        Parameters:
        -----------
        filepath : list[str]
            包含三个数据集路径的列表：[cwru_path, pu_path, hust_path]
        """
        self.cwruPath = filepath[0]
        self.puPath = filepath[1]
        self.hustPath = filepath[2]
        self.seuPath = filepath[3] if len(filepath) > 3 else None
        self.hitPath = filepath[4] if len(filepath) > 4 else None
        
        # 统一的标签映射
        self.label_map = {
            "健康": 0,
            "normal": 0,
            "healthy": 0,
            "inner": 1,
            "outer": 2,
            "ball": 3,
            "其他": 4,
            "combination": 4,
            "combo": 4
        }
        
        # 采样率（Hz）
        self.sampling_rates = {
            "CWRU": 12000,
            "PU": 64000,
            "HUST": 25600,
            "SEU": 2000,
            "HIT": 25000
        }
        
        # 窗口大小（秒）
        self.window_size_sec = 1.0

    def mat_struct_to_dict(self, mat_obj):
        """
        递归将 MATLAB struct 转换为 Python 字典
        """
        if isinstance(mat_obj, np.ndarray) and mat_obj.dtype.names is not None:
            out = {}
            for name in mat_obj.dtype.names:
                # 递归处理
                out[name] = self.mat_struct_to_dict(mat_obj[name])
            return out
        elif isinstance(mat_obj, np.ndarray) and mat_obj.size == 1:
            # 如果是单元素数组，直接取出元素
            return self.mat_struct_to_dict(mat_obj[0])
        else:
            return mat_obj
        
    def _get_window_length(self, dataset_name):
        """根据数据集名称获取窗口长度（采样点数）"""
        return int(self.window_size_sec * self.sampling_rates[dataset_name])

    def _get_label_from_filename(self, filename: str, dataset: str) -> int:
        """
        根据文件名和数据集类型获取标签
        
        Parameters:
        -----------
        filename : str
            文件名
        dataset : str
            数据集类型：'CWRU', 'PU', 或 'HUST'
            
        Returns:
        --------
        int : 标签值 (0-4)
        """
        filename_lower = filename.lower()
        
        if dataset == 'CWRU':
            if 'normal' in filename_lower:
                return self.label_map["健康"]
            elif 'inner' in filename_lower:
                return self.label_map["inner"]
            elif 'outer' in filename_lower:
                return self.label_map["outer"]
            elif 'ball' in filename_lower:
                return self.label_map["ball"]
            else:
                return self.label_map["其他"]
                
        elif dataset == 'PU':
            # PU: K001-K006健康，KI内圈，KA外圈
            if 'k001' in filename_lower or 'k002' in filename_lower or 'k003' in filename_lower or \
               'k004' in filename_lower or 'k005' in filename_lower or 'k006' in filename_lower:
                return self.label_map["健康"]
            elif 'ki' in filename_lower:
                return self.label_map["inner"]
            elif 'ka' in filename_lower:
                return self.label_map["outer"]
            else:
                return self.label_map["其他"]
                
        elif dataset == 'HUST':
            # HUST: H健康，I内圈，O外圈，B滚珠，C组合
            if 'h_' in filename_lower or '_h.' in filename_lower:
                return self.label_map["健康"]
            elif 'i_' in filename_lower or '_i.' in filename_lower:
                return self.label_map["inner"]
            elif 'o_' in filename_lower or '_o.' in filename_lower:
                return self.label_map["outer"]
            elif 'b_' in filename_lower or '_b.' in filename_lower:
                return self.label_map["ball"]
            elif 'c_' in filename_lower or '_c.' in filename_lower:
                return self.label_map["其他"]
            else:
                return self.label_map["其他"]
        
        return self.label_map["其他"]

    def _extract_cwru_condition_key(self, file_path: str) -> str:
        """
        Extract condition key for CWRU samples.
        Priority:
          1) motor speed from parent folder token: gkX-{speed}
          2) motor speed from filename: {speed}-{diameter}-{fault}.mat
          3) HP token (e.g., 0HP/1HP/2HP/3HP)
          4) fallback gk token / "unknown"
        """
        _path_text = str(file_path).replace("\\", "/").lower()

        # Priority-1: canonical folder structure ".../gk2-1750/xxx.mat" -> speed_1750.
        # This avoids mixed condition keys like "gk2" vs "speed_1750" for the same condition.
        m = re.search(r"(?:^|/)gk[_-]?\d+\s*-\s*(\d+)(?:/|$)", _path_text)
        if m:
            return f"speed_{m.group(1)}"

        basename = os.path.basename(str(file_path))
        m = re.match(r"(\d+)\s*-\s*", basename)
        if m:
            return f"speed_{m.group(1)}"

        m = re.search(r"(\d+)\s*hp", _path_text)
        if m:
            return f"hp{m.group(1)}"

        # Fallback: bare gk token → map to known RPM so keys stay unified as speed_*.
        # CWRU: 0HP=1797RPM, 1HP=1772, 2HP=1750, 3HP=1730.
        _GK_TO_RPM = {"0": 1797, "1": 1772, "2": 1750, "3": 1730,
                       "4": 1797, "5": 1772, "6": 1750, "7": 1730}
        m = re.search(r"gk[_-]?(\d+)", _path_text)
        if m:
            _gk_idx = m.group(1)
            _rpm = _GK_TO_RPM.get(_gk_idx)
            if _rpm is not None:
                return f"speed_{_rpm}"
            return f"gk{_gk_idx}"
        return "unknown"

    def _extract_pu_condition_key(self, file_path: str) -> str:
        """
        Extract condition key for PU samples.
        Priority:
          1) explicit gk token in path/file
          2) Nxx_Mxx_Fxx tuple from filename/path
          3) partial Nxx/Fxx if available
          4) fallback "unknown"
        """
        text = str(file_path).lower().replace("\\", "/")
        m = re.search(r"gk[_-]?\d+", text)
        if m:
            return m.group(0).replace("-", "_")

        n_token = re.search(r"n\d{2}", text)
        m_token = re.search(r"m\d{2}", text)
        f_token = re.search(r"f\d{2}", text)
        if n_token and m_token and f_token:
            return f"{n_token.group(0)}_{m_token.group(0)}_{f_token.group(0)}"
        if n_token and f_token:
            return f"{n_token.group(0)}_{f_token.group(0)}"
        if n_token:
            return n_token.group(0)
        return "unknown"

    def _extract_hust_condition_key(self, file_path: str) -> str:
        """
        Extract HUST condition key from speed token in filename.
        Examples:
          0.5X_B_65Hz.xls -> speed_65Hz
          X_I_0-40-0Hz.xls -> speed_0-40-0Hz
        """
        text = str(file_path).replace("\\", "/")
        m = re.search(r"(\d+(?:-\d+)*)\s*[Hh][Zz]", text)
        if m:
            return f"speed_{m.group(1)}Hz"
        # Fallback: tokenized folder names such as ".../65hz/...".
        m = re.search(r"(?:^|/)(\d{2,4}(?:-\d{2,4})*)hz(?:/|$)", text, flags=re.IGNORECASE)
        if m:
            return f"speed_{m.group(1)}Hz"
        return "unknown"

    def _extract_hit_condition_key(self, file_path: str) -> str:
        """
        Extract HIT LP speed key from file path.
        Supports common tokens:
          - LP1000 / lp_1000 / lp-speed=1000
          - n15_m07_f10 (fallback as lp_n15)
          - 600rpm_1200rpm (fallback as lp_600rpm)
        """
        text = str(file_path).replace("\\", "/")
        m = re.search(
            r"(?:^|[^a-zA-Z])lp(?:speed|rpm)?(?:\s*[:=_\-]?\s*)(\d{2,5})(?:rpm)?",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            return f"lp_{m.group(1)}"
        # Common fallback in naming where LP may be encoded as Nxx (e.g., N15_M07_F10).
        m = re.search(r"(?:^|[^a-zA-Z])n(\d{2,4})(?:[^0-9]|$)", text, flags=re.IGNORECASE)
        if m:
            return f"lp_n{m.group(1)}"
        # Fallback for datasets that only expose bearing file ids (data1..data5).
        # This is not LP-speed LOO, but still prevents silent collapse to single unknown group.
        m = re.search(r"(?:^|/)(data\d+)\.npy$", text, flags=re.IGNORECASE)
        if m:
            return f"bearing_{m.group(1).lower()}"
        # Fallback: extract RPM token list and use first rpm as LP speed key.
        _rpm_tokens = re.findall(r"(\d{2,5})\s*rpm", text, flags=re.IGNORECASE)
        if len(_rpm_tokens) > 0:
            return f"lp_{_rpm_tokens[0]}rpm"
        # Fallback: numeric speed pair in path segment, e.g. ".../600_1200/..."
        m = re.search(r"(?:^|/)(\d{2,5})[_\-](\d{2,5})(?:/|$)", text)
        if m:
            return f"lp_{m.group(1)}"
        return "unknown"

    def _extract_seu_condition_key(self, file_path: str) -> str:
        """
        Extract SEU condition key from filename.
        Expected pattern: faultType_speed_load.csv, e.g. inner_20_0.csv
        """
        base = os.path.basename(str(file_path)).lower()
        m = re.match(r"([a-z]+)_(\d+)[_\-](\d+)\.csv$", base)
        if m:
            return f"speed_{m.group(2)}_load_{m.group(3)}"
        m = re.search(r"(\d+)[_\-](\d+)", base)
        if m:
            return f"speed_{m.group(1)}_load_{m.group(2)}"
        return "unknown"

    def _get_seu_label_from_filename(self, file_path: str) -> int:
        base = os.path.basename(str(file_path)).lower()
        if base.startswith("health"):
            return self.label_map["健康"]
        if base.startswith("inner"):
            return self.label_map["inner"]
        if base.startswith("outer"):
            return self.label_map["outer"]
        if base.startswith("ball"):
            return self.label_map["ball"]
        return self.label_map["其他"]

    def _extract_hit_condition_key_from_sample(self, sample_2d: np.ndarray) -> str:
        """
        Infer HIT condition key from per-sample raw channels.
        Priority:
          1) lp/hp pair from channel-0 and channel-1 (median-based, quantized)
          2) lp only from channel-0
          3) unknown
        """
        if not isinstance(sample_2d, np.ndarray) or sample_2d.ndim != 2:
            return "unknown"
        if sample_2d.shape[0] < 2:
            return "unknown"

        def _candidate_speed(chan: np.ndarray, low: float, high: float):
            vals = chan[np.isfinite(chan)]
            if vals.size < 32:
                return None
            stats = [
                float(np.median(vals)),
                float(np.median(np.abs(vals))),
                float(np.mean(np.abs(vals))),
            ]
            valid = []
            for s in stats:
                for c in (s, abs(s)):
                    if low <= c <= high:
                        valid.append(c)
            if not valid:
                return None
            mid = 0.5 * (low + high)
            return min(valid, key=lambda x: abs(x - mid))

        # User-provided operating ranges.
        lp = _candidate_speed(sample_2d[0], low=1000.0, high=5000.0)
        hp = _candidate_speed(sample_2d[1], low=1200.0, high=6000.0)

        def _q(v: float) -> int:
            return int(round(v / 50.0) * 50)

        if lp is not None and hp is not None:
            return f"lp_{_q(lp)}_hp_{_q(hp)}"
        if lp is not None:
            return f"lp_{_q(lp)}"
        return "unknown"

    def cwruLoad(self, return_conditions: bool = False):
        """
        加载CWRU数据集（优化版本：使用numpy预分配和向量化操作）
        
        Returns:
        --------
        tuple : (data, labels)
            data : numpy array of shape (n_samples, window_len)
            labels : numpy array of shape (n_samples,)
        """
        # 获取所有.mat文件
        file_list = glob.glob(os.path.join(self.cwruPath, "**", "*.mat"), recursive=True)
        
        if not file_list:
            raise ValueError(f"No .mat files found in {self.cwruPath}")
        
        window_len = self._get_window_length("CWRU")
        # 预分配列表（使用列表存储，最后一次性转换）
        data_segments = []
        label_list = []
        condition_keys = []
        unknown_condition_examples = []
        
        for file in file_list:
            try:
                mat = loadmat(file)
                # 找 DE 通道
                key = [k for k in mat.keys() if "DE_time" in k and not k.startswith('__')]
                if not key:
                    continue
                key = key[0]
                sig = mat[key].flatten()
                
                # 获取标签
                label = self._get_label_from_filename(os.path.basename(file), 'CWRU')
                condition_key = self._extract_cwru_condition_key(file)
                
                # 切片为1秒窗口（使用numpy向量化切片）
                total = len(sig)
                if total < window_len:
                    continue

                # 计算可以切分的窗口数量（不重叠，步长为window_len）
                # 与原始代码 range(0, total - window_len, window_len) 一致
                num_windows = (total - window_len) // window_len
                if num_windows > 0:
                    # 截取可以完整切分的部分
                    sig_truncated = sig[:num_windows * window_len]
                    # 使用reshape将1D数组重塑为2D数组，每行是一个窗口
                    segments = sig_truncated.reshape(num_windows, window_len)
                    data_segments.append(segments)
                    label_list.extend([label] * num_windows)
                    condition_keys.extend([condition_key] * num_windows)
                    
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        # 一次性转换为numpy数组（比逐个append快得多）
        if data_segments:
            data = np.vstack(data_segments)
            labels = np.array(label_list, dtype=np.int32)
        else:
            data = np.empty((0, window_len), dtype=np.float32)
            labels = np.empty(0, dtype=np.int32)
        
        if return_conditions:
            return data, labels, np.asarray(condition_keys, dtype=object)
        return data, labels

    def puLoad(self, return_conditions: bool = False):
        """
        加载PU数据集，支持 MATLAB v7/v7.3 文件（优化版本）

        Returns:
        --------
        tuple : (data, labels)
            data : numpy array of shape (n_samples, window_len)
            labels : numpy array of shape (n_samples,)
        """
        file_list = glob.glob(os.path.join(self.puPath, "**", "*.mat"), recursive=True)
        if not file_list:
            raise ValueError(f"No .mat files found in {self.puPath}")

        window_len = self._get_window_length("PU")
        data_segments = []
        label_list = []
        condition_keys = []

        for file in file_list:
            try:
                mat = loadmat(file)
                user_keys = [k for k in mat.keys() if not k.startswith('__')]
                if not user_keys:
                    continue
                main_key = user_keys[0]
                main_data = mat[main_key]
                data_dict = self.mat_struct_to_dict(main_data)
                vibration_signal = data_dict['Y']['Data'][0][6].flatten()

                # 获取标签
                label = self._get_label_from_filename(os.path.basename(file), 'PU')
                condition_key = self._extract_pu_condition_key(file)

                # 切片为固定窗口（使用numpy向量化）
                total = len(vibration_signal)
                if total < window_len:
                    continue

                # 计算可以切分的窗口数量（不重叠，步长为window_len）
                # 与原始代码 range(0, total - window_len + 1, window_len) 一致
                # 注意：原始代码是 total - window_len + 1，意味着最后一个窗口可能重叠
                # 但为了保持一致性，我们使用不重叠的方式
                num_windows = (total - window_len) // window_len + 1
                if num_windows > 0:
                    # 如果最后一个窗口不完整，截取到最后一个完整窗口
                    if total >= num_windows * window_len:
                        sig_truncated = vibration_signal[:num_windows * window_len]
                        segments = sig_truncated.reshape(num_windows, window_len)
                    else:
                        # 处理最后一个不完整的窗口
                        num_complete = num_windows - 1
                        if num_complete > 0:
                            sig_complete = vibration_signal[:num_complete * window_len]
                            segments_complete = sig_complete.reshape(num_complete, window_len)
                            # 最后一个窗口（可能不完整，但至少window_len长度）
                            last_segment = vibration_signal[-window_len:]
                            segments = np.vstack([segments_complete, last_segment.reshape(1, -1)])
                        else:
                            segments = vibration_signal[-window_len:].reshape(1, -1)
                    data_segments.append(segments)
                    label_list.extend([label] * segments.shape[0])
                    condition_keys.extend([condition_key] * segments.shape[0])

            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

        # 一次性转换为numpy数组
        if data_segments:
            data = np.vstack(data_segments)
            labels = np.array(label_list, dtype=np.int32)
        else:
            data = np.empty((0, window_len), dtype=np.float32)
            labels = np.empty(0, dtype=np.int32)

        if return_conditions:
            return data, labels, np.asarray(condition_keys, dtype=object)
        return data, labels

    def hustLoad(self, return_conditions: bool = False):
        """
        加载HUST数据集（实际为TSV格式文件，虽然后缀是 .xls）（优化版本：使用numpy直接读取）

        Returns:
        --------
        tuple : (data, labels)
            data : numpy array of shape (n_samples, window_len)
            labels : numpy array of shape (n_samples,)
        """
        # 获取所有.xls文件
        file_list = glob.glob(os.path.join(self.hustPath, "**", "*.xls"), recursive=True)

        if not file_list:
            raise ValueError(f"No .xls files found in {self.hustPath}")

        window_len = self._get_window_length("HUST")
        data_segments = []
        label_list = []
        condition_keys = []

        for file in file_list:
            try:
                # HUST 文件实际为TSV，前22行为meta
                # 使用numpy的loadtxt替代pandas，速度更快
                try:
                    # 尝试使用numpy直接读取（跳过前22行，只读取第2列，即索引为2）
                    acc_x = np.loadtxt(
                        file,
                        delimiter='\t',
                        skiprows=22,
                        usecols=2,  # 只取X方向（第2列，索引为2）
                        encoding='utf-8'
                    )
                except (UnicodeDecodeError, ValueError):
                    # 如果numpy读取失败，回退到pandas
                    df = pd.read_csv(
                        file,
                        sep='\t',
                        header=None,
                        skiprows=22,
                        usecols=[2, 3, 4],
                        encoding='utf-8'
                    )
                    if df.shape[1] < 1:
                        continue
                    acc_x = df.iloc[:, 0].values

                # 获取标签（组合故障在本轮实验协议中丢弃）
                label = self._get_label_from_filename(os.path.basename(file), 'HUST')
                if int(label) == int(self.label_map["其他"]):
                    continue
                condition_key = self._extract_hust_condition_key(file)

                # 切片为固定窗口（使用numpy向量化）
                total = len(acc_x)
                if total < window_len:
                    continue

                # 计算可以切分的窗口数量（不重叠，步长为window_len）
                # 与原始代码 range(0, total - window_len + 1, window_len) 一致
                num_windows = (total - window_len) // window_len + 1
                if num_windows > 0:
                    # 如果最后一个窗口不完整，截取到最后一个完整窗口
                    if total >= num_windows * window_len:
                        acc_x_truncated = acc_x[:num_windows * window_len]
                        segments = acc_x_truncated.reshape(num_windows, window_len)
                    else:
                        # 处理最后一个不完整的窗口
                        num_complete = num_windows - 1
                        if num_complete > 0:
                            acc_x_complete = acc_x[:num_complete * window_len]
                            segments_complete = acc_x_complete.reshape(num_complete, window_len)
                            # 最后一个窗口（可能不完整，但至少window_len长度）
                            last_segment = acc_x[-window_len:]
                            segments = np.vstack([segments_complete, last_segment.reshape(1, -1)])
                        else:
                            segments = acc_x[-window_len:].reshape(1, -1)
                    data_segments.append(segments)
                    label_list.extend([label] * segments.shape[0])
                    condition_keys.extend([condition_key] * segments.shape[0])

            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

        # 一次性转换为numpy数组
        if data_segments:
            data = np.vstack(data_segments)
            labels = np.array(label_list, dtype=np.int32)
        else:
            data = np.empty((0, window_len), dtype=np.float32)
            labels = np.empty(0, dtype=np.int32)

        if return_conditions:
            return data, labels, np.asarray(condition_keys, dtype=object)
        return data, labels

    def hitLoad(self, hit_path, return_conditions: bool = False):
        """
        加载HIT数据集（.npy格式，已分段为20480点）
        
        Parameters:
        -----------
        hit_path : str
            HIT数据集根目录路径，包含 healthy/, inner/, outer/ 子目录
        
        Returns:
        --------
        tuple : (data, labels)
            data : numpy array of shape (n_samples, window_len)
            labels : numpy array of shape (n_samples,)
        """
        # HIT数据存储在子目录中：healthy/, inner/, outer/
        subdirs = {
            "healthy": 0,
            "inner": 1,
            "outer": 2
        }
        
        # HIT数据集已经预分段为20480点，不使用基于时间的窗口长度计算
        window_len = 20480
        data_segments = []
        label_list = []
        condition_keys = []
        unknown_condition_examples = []
        condition_source_counts = {"speed": 0, "path": 0, "unknown": 0}
        
        for subdir, label_val in subdirs.items():
            search_path = os.path.join(hit_path, subdir, "**", "*.npy")
            npy_files = glob.glob(search_path, recursive=True)
            
            for npy_file in npy_files:
                try:
                    # Load data: shape should be (N, 8, 20480) or (8, 20480)
                    data = np.load(npy_file)
                    
                    # Handle different shapes
                    if len(data.shape) == 2 and data.shape[1] == 20480 and data.shape[0] >= 6:
                        # Single sample: (C, 20480) -> reshape to (1, C, 20480), C>=6
                        data = data[np.newaxis, :, :]
                    elif len(data.shape) == 3:
                        # Multiple samples: (N, C, 20480), C>=6
                        if data.shape[1] < 6 or data.shape[2] != 20480:
                            print(f"[HIT] Warning: Skipping {npy_file} due to unexpected shape {data.shape}, expected (N, C>=6, 20480)")
                            continue
                    else:
                        print(f"[HIT] Warning: Skipping {npy_file} due to unexpected shape {data.shape}")
                        continue
                    
                    N = data.shape[0]
                    
                    # Process each sample
                    for n in range(N):
                        # Extract columns 3-6 (0-indexed: channels 2, 3, 4, 5)
                        # Take mean of these 4 columns to get single-channel signal
                        acc_channels = data[n, 2:6, :]  # Shape: (4, 20480)
                        signal = np.mean(acc_channels, axis=0).astype(np.float32)  # Shape: (20480,)
                        
                        # Verify length matches window_len (should always be 20480 for HIT)
                        if len(signal) != window_len:
                            print(f"[HIT] Warning: Signal length {len(signal)} != window_len {window_len} in {npy_file}, skipping")
                            continue
                        
                        data_segments.append(signal)
                        label_list.append(label_val)
                        _cond_key = self._extract_hit_condition_key_from_sample(data[n, :, :])
                        if _cond_key != "unknown":
                            condition_source_counts["speed"] += 1
                        else:
                            _cond_key = self._extract_hit_condition_key(npy_file)
                            if _cond_key != "unknown":
                                condition_source_counts["path"] += 1
                            else:
                                condition_source_counts["unknown"] += 1
                        condition_keys.append(_cond_key)
                        if _cond_key == "unknown" and len(unknown_condition_examples) < 10:
                            unknown_condition_examples.append(str(npy_file))
                        
                except Exception as e:
                    print(f"[HIT] Error loading {npy_file}: {e}")
                    continue
        
        # Convert to numpy arrays
        if data_segments:
            data = np.vstack(data_segments)
            labels = np.array(label_list, dtype=np.int32)
        else:
            data = np.empty((0, window_len), dtype=np.float32)
            labels = np.empty(0, dtype=np.int32)

        # Diagnostic summary for condition extraction robustness (helps strict LOO debugging).
        try:
            if len(condition_keys) > 0:
                _uniq = sorted(set([str(x) for x in condition_keys]))
                _unk = int(sum(1 for x in condition_keys if str(x) == "unknown"))
                print(f"[HIT] condition_groups={len(_uniq)} unknown={_unk}/{len(condition_keys)}")
                _bearing_groups = [k for k in _uniq if str(k).startswith("bearing_data")]
                if len(_bearing_groups) > 0:
                    print(
                        "[HIT] Warning: LP speed token not found in file/path; "
                        f"using bearing-id fallback groups={_bearing_groups}"
                    )
                print(
                    "[HIT] condition key source: "
                    f"speed={condition_source_counts['speed']} "
                    f"path={condition_source_counts['path']} "
                    f"unknown={condition_source_counts['unknown']}"
                )
                if _unk > 0 and len(unknown_condition_examples) > 0:
                    print(f"[HIT] unknown condition examples (first {len(unknown_condition_examples)}):")
                    for _ex in unknown_condition_examples:
                        print(f"[HIT]   {_ex}")
        except Exception:
            pass
        
        if return_conditions:
            return data, labels, np.asarray(condition_keys, dtype=object)
        return data, labels

    def seuLoad(self, seu_path, return_conditions: bool = False):
        """
        加载SEU数据集（CSV格式，前16行为元信息）
        文件命名: 故障类型_工况.csv，如 inner_20_0.csv
        有效信号列: 第2/3/4列（1-indexed）
        """
        file_list = glob.glob(os.path.join(seu_path, "*.csv"))
        if not file_list:
            raise ValueError(f"No .csv files found in {seu_path}")

        window_len = self._get_window_length("SEU")
        data_segments = []
        label_list = []
        condition_keys = []
        unknown_condition_examples = []

        for file in file_list:
            try:
                label = int(self._get_seu_label_from_filename(file))
                if label == int(self.label_map["其他"]):
                    continue
                condition_key = self._extract_seu_condition_key(file)
                if condition_key == "unknown" and len(unknown_condition_examples) < 10:
                    unknown_condition_examples.append(str(file))

                # 前16行为元信息，从第17行开始是数值；读取第2/3/4列（0-indexed: 1,2,3）
                try:
                    arr = np.loadtxt(
                        file,
                        delimiter="\t",
                        skiprows=16,
                        usecols=(1, 2, 3),
                        dtype=np.float32,
                        encoding="utf-8",
                    )
                except Exception:
                    try:
                        arr = np.loadtxt(
                            file,
                            delimiter=",",
                            skiprows=16,
                            usecols=(1, 2, 3),
                            dtype=np.float32,
                            encoding="utf-8",
                        )
                    except Exception:
                        df = pd.read_csv(
                            file,
                            sep=r"[\t,]+",
                            engine="python",
                            header=None,
                            skiprows=16,
                            usecols=[1, 2, 3],
                            encoding="utf-8",
                        )
                        arr = df.values.astype(np.float32)

                if arr.ndim != 2 or arr.shape[1] != 3:
                    continue

                # 与现有单通道输入保持一致：对有效三轴取均值后再做1秒窗口切片
                signal = np.mean(arr, axis=1).astype(np.float32)
                total = len(signal)
                if total < window_len:
                    continue

                num_windows = (total - window_len) // window_len
                if num_windows <= 0:
                    continue
                sig_truncated = signal[:num_windows * window_len]
                segments = sig_truncated.reshape(num_windows, window_len)
                data_segments.append(segments)
                label_list.extend([label] * num_windows)
                condition_keys.extend([condition_key] * num_windows)
            except Exception as e:
                print(f"[SEU] Error loading {file}: {e}")
                continue

        if data_segments:
            data = np.vstack(data_segments)
            labels = np.array(label_list, dtype=np.int32)
        else:
            data = np.empty((0, window_len), dtype=np.float32)
            labels = np.empty(0, dtype=np.int32)

        try:
            if len(condition_keys) > 0:
                _uniq = sorted(set([str(x) for x in condition_keys]))
                _unk = int(sum(1 for x in condition_keys if str(x) == "unknown"))
                print(f"[SEU] condition_groups={len(_uniq)} unknown={_unk}/{len(condition_keys)}")
                if _unk > 0 and len(unknown_condition_examples) > 0:
                    print(f"[SEU] unknown condition examples (first {len(unknown_condition_examples)}):")
                    for _ex in unknown_condition_examples:
                        print(f"[SEU]   {_ex}")
        except Exception:
            pass

        if return_conditions:
            return data, labels, np.asarray(condition_keys, dtype=object)
        return data, labels