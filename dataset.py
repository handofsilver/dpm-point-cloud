"""
ShapeNet 点云数据集
Paper: Luo & Hu, "Diffusion Probabilistic Models for 3D Point Cloud Generation", CVPR 2021

数据格式：ShapeNet 点云打包为 .h5 文件，每个文件包含多朵点云。
预处理：每朵点云独立归一化到零均值 + 单位方差，与噪声尺度匹配。
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNetDataset(Dataset):
    """
    ShapeNet 点云数据集。

    从一个目录下的所有 .h5 文件中读取点云，拼成一个大数组，
    __getitem__ 返回归一化后的单朵点云。

    目录结构示例:
        data/shapenet/
            train_0.h5    # 内含 key "data": (M, 2048, 3)
            train_1.h5
            ...

    用法:
        dataset = ShapeNetDataset(root="data/shapenet", split="train")
        loader  = DataLoader(dataset, batch_size=128, shuffle=True)
        for x0 in loader:   # x0: (128, 2048, 3)
            loss = model.get_loss(x0)
    """

    def __init__(self, root: str, split: str = "train", num_points: int = 2048):
        """
        Args:
            root      : 存放 .h5 文件的目录路径
            split     : "train" 或 "test"，用于匹配文件名
            num_points: 每朵点云采样的点数（默认 2048）
        """
        super().__init__()

        self.num_points = num_points

        # 找到所有匹配的 .h5 文件，sorted() 保证顺序稳定（文件系统返回顺序不确定）
        pattern = os.path.join(root, f"*{split}*.h5")
        h5_files = sorted(glob.glob(pattern))
        assert len(h5_files) > 0, f"在 '{root}' 下未找到包含 '{split}' 的 .h5 文件，请检查路径。"

        # 读入所有文件，沿第 0 维拼成一个大数组
        # 每个文件：f["data"] shape = (M, N, 3)，M 为该文件包含的点云数
        all_points = []
        for path in h5_files:
            with h5py.File(path, "r") as f:
                all_points.append(np.array(f["data"]))  # np.array() 触发实际读取并明确返回 ndarray
        self.points = np.concatenate(all_points, axis=0)  # (total, N, 3)

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        返回第 idx 朵点云，归一化后转为 Tensor。

        归一化（逐朵独立）：
            1. 减去重心（零均值）：pc -= pc.mean(axis=0)   # 平移到原点，3 轴各自去均值
            2. 除以全局标准差：   pc /= pc.std()           # 均匀缩放，保持 xyz 比例

        Returns:
            pc: (num_points, 3)  归一化后的点云，dtype=torch.float32
        """
        pc = self.points[idx]  # (N, 3)，numpy array

        # 随机采样 num_points 个点（不放回），增加训练多样性
        # replace=False 保证不重复选同一个点
        choice = np.random.choice(pc.shape[0], self.num_points, replace=False)
        pc = pc[choice]  # (num_points, 3)

        # 零均值：减去重心（每轴独立），平移点云到原点
        pc -= pc.mean(axis=0)

        # 单位方差：除全局 std（一个标量），均匀缩放保持形状比例
        # 用全局而非分轴 std，避免 xyz 三轴被独立拉伸而变形
        pc /= pc.std()

        # from_numpy 零拷贝转 Tensor（共享内存），.float() 转 float32 匹配模型权重类型
        return torch.from_numpy(pc).float()
