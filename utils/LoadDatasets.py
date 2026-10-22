from utils.LoadingDataItem import LoadData

def load_datasets(cwru_path, pu_path, hust_path, seu_path=None, hit_path=None, include_conditions=False):
    """加载所有数据集"""
    loader = LoadData([cwru_path, pu_path, hust_path, seu_path, hit_path])
    # 加载CWRU数据集
    if include_conditions:
        cwru_data, cwru_labels, cwru_conditions = loader.cwruLoad(return_conditions=True)
    else:
        cwru_data, cwru_labels = loader.cwruLoad()
        cwru_conditions = None
    print(f"   CWRU数据形状: {cwru_data.shape if hasattr(cwru_data, 'shape') else '无数据'}")
    print(f"   CWRU标签形状: {cwru_labels.shape if hasattr(cwru_labels, 'shape') else '无数据'}")
    if hasattr(cwru_labels, 'shape'):
        unique_labels = set(cwru_labels)
        print(f"   CWRU唯一标签: {sorted(unique_labels)}")
        for label in sorted(unique_labels):
            count = sum(cwru_labels == label)
            print(f"     标签 {label}: {count} 个样本")

    # 加载PU数据集
    if include_conditions:
        pu_data, pu_labels, pu_conditions = loader.puLoad(return_conditions=True)
    else:
        pu_data, pu_labels = loader.puLoad()
        pu_conditions = None
    print(f"   PU数据形状: {pu_data.shape if hasattr(pu_data, 'shape') else '无数据'}")
    print(f"   PU标签形状: {pu_labels.shape if hasattr(pu_labels, 'shape') else '无数据'}")
    if hasattr(pu_labels, 'shape'):
        unique_labels = set(pu_labels)
        print(f"   PU唯一标签: {sorted(unique_labels)}")
        for label in sorted(unique_labels):
            count = sum(pu_labels == label)
            print(f"     标签 {label}: {count} 个样本")

    # 加载HUST数据集
    if include_conditions:
        hust_data, hust_labels, hust_conditions = loader.hustLoad(return_conditions=True)
    else:
        hust_data, hust_labels = loader.hustLoad()
        hust_conditions = None
    print(f"   HUST数据形状: {hust_data.shape if hasattr(hust_data, 'shape') else '无数据'}")
    print(f"   HUST标签形状: {hust_labels.shape if hasattr(hust_labels, 'shape') else '无数据'}")
    if hasattr(hust_labels, 'shape'):
        unique_labels = set(hust_labels)
        print(f"   HUST唯一标签: {sorted(unique_labels)}")
        for label in sorted(unique_labels):
            count = sum(hust_labels == label)
            print(f"     标签 {label}: {count} 个样本")

    # 加载SEU数据集（如果路径提供）
    seu_data = None
    seu_labels = None
    seu_conditions = None
    if seu_path is not None:
        try:
            if include_conditions:
                seu_data, seu_labels, seu_conditions = loader.seuLoad(seu_path, return_conditions=True)
            else:
                seu_data, seu_labels = loader.seuLoad(seu_path)
            
            print(f"   SEU数据形状: {seu_data.shape if hasattr(seu_data, 'shape') else '无数据'}")
            print(f"   SEU标签形状: {seu_labels.shape if hasattr(seu_labels, 'shape') else '无数据'}")
            if hasattr(seu_labels, 'shape'):
                unique_labels = set(seu_labels)
                print(f"   SEU唯一标签: {sorted(unique_labels)}")
                for label in sorted(unique_labels):
                    count = sum(seu_labels == label)
                    print(f"     标签 {label}: {count} 个样本")
        except Exception as e:
            print(f"   [SEU] Warning: Failed to load SEU dataset: {e}")
            seu_data = None
            seu_labels = None

    # 加载HIT数据集（如果路径提供）
    hit_data = None
    hit_labels = None
    hit_conditions = None
    if hit_path is not None:
        try:
            if include_conditions:
                hit_data, hit_labels, hit_conditions = loader.hitLoad(hit_path, return_conditions=True)
            else:
                hit_data, hit_labels = loader.hitLoad(hit_path)

            print(f"   HIT数据形状: {hit_data.shape if hasattr(hit_data, 'shape') else '无数据'}")
            print(f"   HIT标签形状: {hit_labels.shape if hasattr(hit_labels, 'shape') else '无数据'}")
            if hasattr(hit_labels, 'shape'):
                unique_labels = set(hit_labels)
                print(f"   HIT唯一标签: {sorted(unique_labels)}")
                for label in sorted(unique_labels):
                    count = sum(hit_labels == label)
                    print(f"     标签 {label}: {count} 个样本")
        except Exception as e:
            print(f"   [HIT] Warning: Failed to load HIT dataset: {e}")
            hit_data = None
            hit_labels = None
            hit_conditions = None

    if include_conditions:
        return (
            cwru_data, cwru_labels, pu_data, pu_labels, hust_data, hust_labels,
            seu_data, seu_labels, hit_data, hit_labels,
            cwru_conditions, pu_conditions, hust_conditions, seu_conditions, hit_conditions,
        )
    return cwru_data, cwru_labels, pu_data, pu_labels, hust_data, hust_labels, seu_data, seu_labels, hit_data, hit_labels
