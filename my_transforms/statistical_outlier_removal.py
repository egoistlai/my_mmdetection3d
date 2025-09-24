import numpy as np
import open3d as o3d
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import BasePoints
import torch

@TRANSFORMS.register_module()
class StatisticalOutlierRemoval:
    """
    使用 Open3D 实现的统计离群点移除 (Statistical Outlier Removal) 模块。
    这是一种有效的点云去噪方法，可以去除稀疏的背景噪声点。

    Args:
        nb_neighbors (int): 用于计算平均距离的邻近点数量。
        std_ratio (float): 标准差的倍数阈值。点与其邻居的平均距离如果大于
                         `全局平均距离 + std_ratio * 全局标准差`，则被视为离群点。
    """
    def __init__(self, nb_neighbors: int, std_ratio: float):
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def __call__(self, results: dict) -> dict:
        """
        处理点云数据，移除离群点。

        Args:
            results (dict): 包含点云数据的字典。

        Returns:
            dict: 处理后的结果字典。
        """
        # 从results字典中获取点云对象
        points = results['points']
        
        # 将MMDetection3D的点云对象转换为Open3D的PCD对象
        pcd = o3d.geometry.PointCloud()
        # open3d需要的是 (N, 3) 的坐标
        pcd.points = o3d.utility.Vector3dVector(points.tensor[:, :3].cpu().numpy())
        
        # 执行统计离群点移除
        # remove_statistical_outlier 返回一个元组 (filtered_pcd, inlier_indices)
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors,
            std_ratio=self.std_ratio
        )
        
        # 使用计算出的内点索引来过滤原始的点云对象
        # 这样可以保留原始的所有维度信息 (x, y, z, doppler, power, etc.)
        # 1. 将索引列表转换为 PyTorch Tensor
        keep_indices = torch.tensor(ind, dtype=torch.long, device=points.device)

        # 2. 使用 Tensor 来对 points 对象进行索引
        results['points'] = points[keep_indices]
        
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nb_neighbors={self.nb_neighbors}, std_ratio={self.std_ratio})'