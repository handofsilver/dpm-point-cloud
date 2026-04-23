# 实验与评测记录索引

本目录记录本仓库在 ShapeNet 数据集上训练 / 评测 DPM-3D（Luo & Hu 2021）的完整过程与归因分析。文档按**事件发生先后**组织，每篇职责分明、避免重复排查过程。

---

## 文档一览（按时间线）

| 阶段 | 文档 | 主题 | 一句话摘要 |
|:-:|---|---|---|
| Round 1 实验记录 | [`1st_round_training_and_eval.md`](1st_round_training_and_eval.md) | 首轮完整训练 + 评测 | 在服务器上一次性跑完 AE / GaussianVAE / FlowVAE 全 55 类合训 + Table 1 / Table 2 评测，记录环境、命令、原始数字 |
| Round 1 归因 | [`ae_eval_gap_analysis.md`](ae_eval_gap_analysis.md) | Table 2（AE）CD 约 10× gap 归因 | 用官方 `AE_all.pt` 在本地 `shapenet.hdf5` 上复测同样落后论文 Table 2 —— gap 来自数据集（PC15k vs shapenet.hdf5）而非训练质量 |
| Round 1 归因 | [`gen_eval_gap_analysis.md`](gen_eval_gap_analysis.md) | Table 1（Gen）COV/1-NNA/JSD gap 排查 | 定位 4 个差距来源：全 55 类合训（主因）、Flow 容量、LR schedule、数据集差异；给出 Round 2 的消融方向 |
| 工程决策 | [`emd_sinkhorn_decision.md`](emd_sinkhorn_decision.md) | EMD 用 Sinkhorn 代替 approxmatch | 论文 `approxmatch.cu` 原码已失传 + PyTorch 2.x / CUDA 12 不兼容，改用 `geomloss` Sinkhorn；MMD-EMD 绝对值与论文不可比，COV-EMD / 1-NNA-EMD 方向性可比 |
| 工程备份 | [`checkpoint_loading_restore.md`](checkpoint_loading_restore.md) | 加载官方 `AE_all.pt` 的兼容补丁 | 原仓库 Encoder 带 BN、`var_sched` 长度 201 与本仓库简化实现不一致；补丁仅用于官方 ckpt 复现实验，不入主线 |
| Round 2 规划 | [`next_experiments.md`](next_experiments.md) | Round 2 消融计划 | 列出 P0（per-category 训练）/ P1（LR schedule 两段式）/ P2（Flow 扩容）/ P3（Encoder BN）四项，定优先级与预期信号 |
| Round 2 实验记录 | [`2nd_round_training_and_eval.md`](2nd_round_training_and_eval.md) | Round 2 per-category + LR schedule + ablation | 合并 P0 + P1 执行 airplane / chair FlowVAE 训练，CD 基本对齐论文；同协议 GaussianVAE 对照验证 FlowVAE > GaussianVAE 方向性成立 |
| Round 2 归因 | [`jsd_gap_analysis.md`](jsd_gap_analysis.md) | Table 1 JSD 列 gap 拆分 | 纯数据 oracle 实验证明 `shapenet.hdf5` airplane train/test JSD 下限 ~0.007（论文 PC15k 下限 0.000809，差 8.6×）；实测 0.011 = 0.007 数据下限 + 0.004 模型残差 |

---

## 关系图

```
         Round 1 实验
  1st_round_training_and_eval.md
           │
   ┌───────┼────────────┐
   ▼       ▼            ▼
 AE gap  Gen gap    工程决策
ae_..    gen_..     emd_sinkhorn_decision.md
analysis analysis   checkpoint_loading_restore.md
           │
           ▼
      Round 2 规划
    next_experiments.md
           │
           ▼
        Round 2 实验
 2nd_round_training_and_eval.md
           │
           ▼
        Round 2 归因
    jsd_gap_analysis.md
```

---

## 阅读顺序建议

- **只想看当前最好结果**：直接看 [`2nd_round_training_and_eval.md`](2nd_round_training_and_eval.md) §5
- **想理解归因链条**：`1st_round_training_and_eval.md` → `gen_eval_gap_analysis.md` → `next_experiments.md` → `2nd_round_training_and_eval.md` → `jsd_gap_analysis.md`
- **想理解 AE 为什么量级对不上论文**：[`ae_eval_gap_analysis.md`](ae_eval_gap_analysis.md)
- **想理解为什么 EMD 绝对值不可比**：[`emd_sinkhorn_decision.md`](emd_sinkhorn_decision.md)
- **想复现官方 ckpt 的评测**：[`checkpoint_loading_restore.md`](checkpoint_loading_restore.md)

---

## 文档职责边界

为避免内容重复与陈述冲突，每份文档的职责范围严格划分：

- **`*_training_and_eval.md`** 只记录**做了什么 / 跑了什么命令 / 得到什么原始数字**，不做归因
- **`*_gap_analysis.md`** 只做**某一列指标与论文差距的归因**，不重复训练命令细节
- **`*_decision.md`** 只记录**工程选型的理由与影响**，不涉及具体训练 / 评测结果
- **`next_experiments.md`** 只面向未来，记录**规划**，实验一旦落地由对应 `*_training_and_eval.md` 接手
