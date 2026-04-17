"""
点云可视化工具
提供两个函数：
    - plot_point_cloud   : 单朵点云 3D 散点图
    - plot_reconstruction: 输入 vs 重建并排对比图（附 CD 数值）
"""

import torch
import matplotlib.pyplot as plt

# mpl_toolkits 是 matplotlib 的 3D 扩展包，import 后才能使用 projection='3d'
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_point_cloud(
    pc: torch.Tensor,  # (N, 3) 单朵点云，不含 batch 维
    title: str = "",
    ax: "Axes3D | None" = None,
) -> "Axes3D":
    """
    把一朵点云画成 3D 散点图。

    Args:
        pc   : (N, 3) Tensor
        title: 子图标题
        ax   : 传入已有的 Axes3D 对象可复用；None 则自动新建

    Returns:
        ax: 画好的 Axes3D 对象
    """
    # .detach().cpu().numpy()：断开计算图 → 移回 CPU → 转 numpy
    # 凡是把 Tensor 交给非 PyTorch 库（matplotlib、numpy）都需要这三步
    pc_np = pc.detach().cpu().numpy()

    x, y, z = pc_np[:, 0], pc_np[:, 1], pc_np[:, 2]

    # projection='3d' 告诉 matplotlib 这个子图使用三维坐标系
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")

    # s=0.5 是点的大小（单位：像素²），alpha=0.6 是透明度
    ax.scatter(x, y, z, s=0.5, alpha=0.6)  # type: ignore[union-attr]

    # view_init(elev, azim)：elev 仰角（0=水平视角，90=正俯视），azim 水平方位角
    # elev=30, azim=45 是经验值，大多数形状从这个角度看得比较清楚
    ax.view_init(elev=30, azim=45)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore[union-attr]

    return ax  # type: ignore[return-value]


def plot_reconstruction(
    input_pc: torch.Tensor,  # (N, 3) 输入点云（真值）
    recon_pc: torch.Tensor,  # (N, 3) 重建点云
    cd_value: float | None = None,
) -> plt.Figure:  # type: ignore
    """
    左右并排展示输入点云与重建点云，顶部标题附 CD 数值。

    Args:
        input_pc : (N, 3) 输入点云
        recon_pc : (N, 3) 重建点云
        cd_value : Chamfer Distance 数值（可选，None 则不显示）

    Returns:
        fig: matplotlib Figure 对象，调用方可 fig.savefig() 保存
    """
    # figsize=(宽, 高)，单位英寸；10×5 刚好放两个并排子图
    fig = plt.figure(figsize=(10, 5))
    # add_subplot(行数, 列数, 第几个)：1行2列网格，分别取第1、第2个子图
    ax1 = fig.add_subplot(nrows=1, ncols=2, index=1, projection="3d")
    ax2 = fig.add_subplot(nrows=1, ncols=2, index=2, projection="3d")

    plot_point_cloud(input_pc, title="Input", ax=ax1)
    plot_point_cloud(recon_pc, title="Reconstruction", ax=ax2)

    # fig.suptitle 是图级标题，显示在整张 figure 顶部
    # 区别于 ax.set_title（子图级），有多个子图时全局信息用 suptitle
    if cd_value is not None:
        fig.suptitle(f"Chamfer Distance: {cd_value:.6f}")

    # tight_layout 自动调整子图间距，防止标题/标签重叠
    plt.tight_layout()
    return fig
