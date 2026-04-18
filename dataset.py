"""
ShapeNet 点云数据集
Paper: Luo & Hu, "Diffusion Probabilistic Models for 3D Point Cloud Generation", CVPR 2021

数据格式：单个 HDF5 文件（shapenet.hdf5），内部结构：
    {synsetid} / {split} → ndarray (N, 2048, 3)

归一化：每朵点云独立 shape_unit — 减重心，除全局 std。
"""

import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# synsetid → 类别名称映射（完整 ShapeNet 55 类）
SYNSET_TO_CATE = {
    "02691156": "airplane",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02747177": "can",
    "02942699": "camera",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03046257": "clock",
    "03207941": "dishwasher",
    "03211117": "monitor",
    "04379243": "table",
    "04401088": "telephone",
    "02946921": "tin_can",
    "04460130": "tower",
    "04468005": "train_obj",
    "03085013": "keyboard",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "speaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote_control",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04530566": "vessel",
    "04554684": "washer",
    "02992529": "cellphone",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
}
CATE_TO_SYNSET = {v: k for k, v in SYNSET_TO_CATE.items()}


class ShapeNetDataset(Dataset):
    """
    ShapeNet 点云数据集，从单个 .hdf5 文件读取。

    HDF5 内部结构：
        f[synsetid][split] → (N, 2048, 3)

    用法:
        # 训练，所有类别
        dataset = ShapeNetDataset(path="data/shapenet/shapenet.hdf5", split="train")

        # 只加载 airplane + chair（用于 per-category 评估）
        dataset = ShapeNetDataset(path=..., split="test", cates=["airplane", "chair"])

        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        for batch in loader:
            x0 = batch["pointcloud"]   # (B, 2048, 3)
            cate = batch["cate"]       # list of str，长度 B
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        cates: list = None,  # type: ignore
        num_points: int = 2048,
        random_seed: int = 2020,
    ):
        """
        Args:
            path       : shapenet.hdf5 文件路径
            split      : "train" | "val" | "test"
            cates      : 类别名称列表，如 ["airplane", "chair"]；None 表示加载全部类别
            num_points : 每朵点云采样点数（默认 2048）
            random_seed: shuffle 用的随机种子（保证可复现）
        """
        super().__init__()
        assert split in ("train", "val", "test"), f"split 必须是 train/val/test，got: {split}"
        self.num_points = num_points

        # 确定要加载的 synsetid 列表
        if cates is None:
            target_synsets = None  # 加载所有
        else:
            target_synsets = [CATE_TO_SYNSET[c] for c in cates]

        # 从 HDF5 读取点云，保留 cate 信息供 per-category 评估使用
        self.samples = []  # list of {"pointcloud": Tensor(N,3), "cate": str}

        with h5py.File(path, "r") as f:
            available_synsets = list(f.keys())
            synsets_to_load = (
                [s for s in available_synsets if s in target_synsets]
                if target_synsets is not None
                else available_synsets
            )
            assert len(synsets_to_load) > 0, (
                f"HDF5 中未找到指定类别的 synsetid。"
                f"可用: {available_synsets}, 请求: {target_synsets}"
            )

            for synsetid in synsets_to_load:
                if split not in f[synsetid]:  # type: ignore
                    continue
                # f[synsetid][split]: (N, 2048, 3)
                pcs = f[synsetid][split][...]  # type: ignore # 触发实际读取，返回 ndarray
                cate_name = SYNSET_TO_CATE.get(synsetid, synsetid)
                for pc in pcs:  # type: ignore
                    self.samples.append({"pointcloud": pc, "cate": cate_name})

        assert (
            len(self.samples) > 0
        ), f"从 {path} 读取到 0 条数据，请检查 split={split} 和 cates={cates}"

        # 确定性 shuffle（与原仓库保持一致，保证训练/测试集划分可复现）
        random.Random(random_seed).shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        返回第 idx 条样本，包含归一化点云和类别名。

        归一化（shape_unit，逐朵独立）：
            pc -= pc.mean(axis=0)   # 平移到原点
            pc /= pc.std()          # 全局 std 均匀缩放，保持形状比例

        Returns:
            dict:
                "pointcloud": Tensor (num_points, 3) float32
                "cate":       str，类别名称
        """
        sample = self.samples[idx]
        pc = sample["pointcloud"].copy()  # ndarray (N, 3)，copy 避免修改原数组

        # 随机采样 num_points 个点（不放回），增加训练多样性
        if pc.shape[0] > self.num_points:
            choice = np.random.choice(pc.shape[0], self.num_points, replace=False)
            pc = pc[choice]

        # shape_unit 归一化，记录 shift/scale 供反归一化
        shift = pc.mean(axis=0)  # (3,)  各轴均值
        scale = pc.std()  # scalar  全局 std
        pc = (pc - shift) / scale

        return {
            "pointcloud": torch.from_numpy(pc).float(),  # (num_points, 3)
            "shift": torch.from_numpy(shift).float(),  # (3,)
            "scale": torch.tensor(scale, dtype=torch.float32),  # scalar
            "cate": sample["cate"],
        }
