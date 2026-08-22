# HPID-Split 跨类别融合消融（2026-08-21）

## 实验范围

- 使用与主实验相同的 65 个 PACO-LVIS oracle-crop holdout 案例。
- 覆盖设备 15 例、家具 4 例、容器 18 例、交通工具 2 例、日用品 15 例、工具/道具 11 例。
- 四个条件共享原自动推理生成的冻结候选掩码，只切换 HPID-Split 融合与所有权开关。
- 每个案例的四个预测全部完成后，评价器才读取部件真值；推理不使用真值。
- 完整融合条件在 65/65 个案例上复现了原发布包的 Part 数量和前景像素。

## 消融条件

| 条件 | 含义 |
| --- | --- |
| A0 independent max | 不使用跨源 noisy-OR 共识、层级约束、根覆盖守恒、直接门控、层级重复抑制、余量归并和 specificity ownership。 |
| A1 cross-source consensus | 在 A0 上加入按来源族去相关后的跨源 noisy-OR 共识。 |
| A2 consensus + hierarchy | 在 A1 上加入父级支持、父级残差、根覆盖守恒、直接门控、层级重复抑制和余量归并。 |
| A3 full fusion | 在 A2 上加入 specificity-aware pixel ownership，为完整 HPID-Split 融合。 |

## 主要结果

| 条件 | Object IoU | Part P@0.25 | Part R@0.25 | Part F1@0.25 | Boundary F1 | Semantic recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 0.6825 | 0.3300 | 0.1366 | 0.1684 | 0.2217 | 0.0763 |
| A1 | 0.6851 | 0.4118 | 0.1946 | 0.2285 | 0.3069 | 0.1006 |
| A2 | 0.7369 | 0.3190 | 0.3332 | 0.2870 | 0.4872 | 0.1631 |
| A3 | 0.7368 | 0.3525 | 0.4099 | 0.3319 | 0.4939 | 0.1782 |

相对 A0，完整融合的 Part F1@0.25 配对增量为 0.1635，case-bootstrap 95% CI 为 [0.1024, 0.2242]；Part recall 增量为 0.2733 [0.2098, 0.3371]。相对 A2，specificity ownership 将 Part F1 提高 0.0449 [0.0217, 0.0706]，将 recall 提高 0.0766 [0.0424, 0.1134]。其 Object IoU 变化为 -0.0001 [-0.0003, 0.0000]，说明该阶段主要重新分配根对象内部像素，而不是扩大目标轮廓。

六个领域的平均 Part F1 都随 A0 至 A3 上升。由于家具仅 4 例、交通工具仅 2 例，分领域结果仅作描述，不作为独立显著性结论。

## 证据边界

该实验是冻结候选之后的端到端融合与 Part-ID 分配消融，不是候选生成器消融，也不是整图开放世界目标检测实验。三重语义-结构-外观候选门控的独立消融仍由 `candidate_gate_ablation_*.csv` 报告。角色与游戏场景结果只作为额外结构回归，不再作为跨类别主消融的替代证据。

## 文件

- `fusion_ablation_cases.csv`：65 x 4 条逐案例结果。
- `fusion_ablation_summary.csv`：总体均值与 95% case-bootstrap CI。
- `fusion_ablation_by_domain.csv`：六领域描述性均值。
- `fusion_ablation_paired_deltas.csv`：完整融合相对各删减条件的配对差值与区间。
- `full_fusion_reproduction.csv`：完整条件对原发布结果的复现核对。
- `fusion_ablation_report.json`：输入哈希、开关、范围和无真值推理声明。
- `figures/`：跨类别消融图的 PNG、SVG 和 TIFF 文件。
