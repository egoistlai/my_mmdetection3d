import pickle
import numpy as np

def print_recursive(data, indent=0):
    """
    以缩进格式递归打印数据结构（字典、列表等）的详细内容。
    """
    prefix = '  ' * indent
    
    if isinstance(data, dict):
        # 如果是字典，遍历键值对
        for key, value in data.items():
            print(f"{prefix}Key: '{key}'")
            print_recursive(value, indent + 1)
            
    elif isinstance(data, list):
        # 如果是列表，打印前几个元素
        num_to_show = 3
        print(f"{prefix}List (共 {len(data)} 个元素, 显示前 {num_to_show} 个):")
        for i, item in enumerate(data[:num_to_show]):
            print(f"{prefix}  - Item {i}:")
            print_recursive(item, indent + 2)
        if len(data) > num_to_show:
            print(f"{prefix}  ...")
            
    elif isinstance(data, np.ndarray):
        # 如果是Numpy数组，设置打印格式并完整打印
        with np.printoptions(precision=4, suppress=True):
            print(f"{prefix}numpy array (shape: {data.shape}, dtype: {data.dtype}):")
            # 将数组的每一行都加上缩进，方便阅读
            array_str = str(data)
            indented_array_str = '\n'.join([f"{prefix}  {line}" for line in array_str.split('\n')])
            print(indented_array_str)
            
    else:
        # 其他基本类型直接打印
        print(f"{prefix}Value: {data} (Type: {type(data).__name__})")


def main():
    # ====================================================================
    # --- 用户需要修改的部分 ---
    # 1. 设置你想要检查的 .pkl 文件路径
    #    示例: 'data/comp_data/drad_infos_train.pkl'
    #    示例: 'data/comp_data/drad_dbinfos_train.pkl'
    pkl_file_path = 'work_dirs/my_test_results/pred_instances.pkl'
    
    # 2. 设置你想查看列表中的前几个样本的详细信息 (建议先设为1，避免信息过多)
    num_samples_to_inspect = 1
    # ====================================================================

    print(f"--- 正在检查文件: {pkl_file_path} ---")

    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"\n错误：文件未找到！请确认路径 '{pkl_file_path}' 是否正确。")
        return
    except Exception as e:
        print(f"\n加载文件时发生错误: {e}")
        return

    print("\n文件加载成功！")

    # 对已知的数据结构进行特殊处理，使其更有条理
    if pkl_file_path.endswith('_infos_train.pkl') or pkl_file_path.endswith('_infos_val.pkl'):
        print(f"文件结构: 这是一个包含 'metainfo' 和 'data_list' 的字典。")
        print("\n--- Metainfo 内容 ---")
        print_recursive(data.get('metainfo', {}))
        
        print(f"\n--- 'data_list' 内容 (显示前 {num_samples_to_inspect} 个样本) ---")
        data_list = data.get('data_list', [])
        for i in range(min(num_samples_to_inspect, len(data_list))):
            print(f"\n--- 样本 {i} ---")
            print_recursive(data_list[i])
    else:
        # 对于其他类型的pkl文件，直接打印
        print("\n--- 文件完整内容 ---")
        print_recursive(data)

    print(f"\n--- 检查完成 ---")


if __name__ == '__main__':
    main()