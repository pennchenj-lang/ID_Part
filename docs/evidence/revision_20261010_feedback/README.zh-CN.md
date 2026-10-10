# 反馈问题与修订数据索引 — 2026年10月10日

[English index](README.md) · [机器可读索引](feedback_index.json)

本索引将全部14条反馈主题对应到修订所依据的源表、冻结协议与分析记录，仅列主题，不复制审稿通信原文。数据既包含新增实验，也包含历史实验的重新分析；仅涉及符号的修改单独标明。

| 编号 | 主题与关键数据 | 解释范围 |
|---|---|---|
| R1.1 | 身份协议与返回选项数量<br>[paired_option_contrasts_all_four.csv](R1C6_Handoff253/statistics/paired_option_contrasts_all_four.csv) · [paired_correctness_guardrails.csv](R1C6_Handoff253/statistics/paired_correctness_guardrails.csv) · [summary.json](R2C2/evidence_audit/summary.json) | 253次请求支持共同展示规则下返回更少选项；不等于分割质量领先，也未测量人工用时。 |
| R1.2 | 门控与像素归属敏感性<br>[sensitivity_protocol.json](R2C4/existing_source_tables/sensitivity_protocol.json) · [sensitivity_summary.csv](R2C4/existing_source_tables/sensitivity_summary.csv) · [group_gate_summary.csv](../revision_20261008/evidence/gate_ablation/group_gate_summary.csv) · [gate_report.txt](../revision_20261008/evidence/gate_ablation/gate_report.txt) | 226例50种细粒度ID设置，加上旧归档226例及42例的八子集Group门控实验；局部敏感性分析不证明最优或排除过拟合。 |
| R1.3 | 受控请求掩模交接的范围<br>[protocol_frozen.json](R1C6_Handoff253/protocol_frozen.json) · [completion_frontend_report.json](R2C5/existing_source_tables/completion_frontend_report.json) · [evidence_audit.json](R2C5/evidence_audit.json) | 身份包在缺损前冻结。253请求扩展评估路由，不新增从遮挡观测恢复身份的证据，也不扩大原37例LaMa实验。 |
| R1.4 | 完整图像自动发现<br>[protocol.json](R1C4_FullImage42/protocol.json) · [full_image_cases.csv](R1C4_FullImage42/full_image_cases.csv) · [full_image_summary.json](R1C4_FullImage42/full_image_summary.json) | 42目标完整图像与标注导出裁剪的配对比较，计入主对象选择错误；不是全对象检测AP评测。 |
| R1.5 | 已执行的学习型部件基线<br>[protocol_frozen.json](R1C5_PartCATSeg/protocol_frozen.json) · [fairness_input_audit.json](R1C5_PartCATSeg/fairness_input_audit.json) · [summary.json](R1C5_PartCATSeg/scoring/summary.json) · [cases.json](R1C5_PartCATSeg/scoring/cases.json) | 42个共同裁剪输入，使用官方PartCATSeg VOC权重及固定连通分量转换；结论限于VOC到PACO迁移协议，HOPS与KPS仍未执行。 |
| R1.6 | 扩展路由、置信区间与精简失败<br>[routing_cases.csv](R1C6_Handoff253/routing_run/routing_cases.csv) · [routing_states_with_wilson.csv](R1C6_Handoff253/statistics/routing_states_with_wilson.csv) · [domain_category_failure_summary.csv](R1C6_Handoff253/statistics/domain_category_failure_summary.csv) · [summary.json](R1C6_Handoff253/mechanism_analysis/summary.json) | 原37例加新增216个独立源图像，五种方法共1265次运行；保留分队列结果、分母、正确选项损失及条件复核率。 |
| R2.1 | 未知分类体系与层级行为<br>[taxonomy_behavior.json](R2C1_Taxonomy/taxonomy_behavior.json) | 43个确定性API测试说明回退与校验行为，不代表未知分类体系图像上的识别准确率。 |
| R2.2 | 相同候选下DBSCAN与HPID差异<br>[summary.json](R2C2/evidence_audit/summary.json) · [all_990_reference_pairing.csv](R2C2/evidence_audit/all_990_reference_pairing.csv) · [dbscan_only_hpid_no_geometry_61_actual_stage_transitions.csv](R2C2/evidence_audit/dbscan_only_hpid_no_geometry_61_actual_stage_transitions.csv) | 针对1951个共同候选和990个参考部件的阶段与匹配诊断；阶段关联本身不是门控因果消融。 |
| R2.3 | 来源可靠性与noisy-OR<br>[family_empirical_reliability.csv](R2C3/reliability_audit/family_empirical_reliability.csv) · [summary.csv](R2C3/clean_ablation/summary.csv) · [paired_deltas.csv](R2C3/clean_ablation/paired_deltas.csv) · [report.json](R2C3/consensus_stress/report.json) | 冻结启发式权重、经验诊断、226例融合消融及150个受控压力案例；重复抑制不等于概率校准，也不保证排除高置信假阳性。 |
| R2.4 | 手工像素归属常数<br>[sensitivity_protocol.json](R2C4/existing_source_tables/sensitivity_protocol.json) · [sensitivity_cases.csv](R2C4/existing_source_tables/sensitivity_cases.csv) · [summary.json](R2C4/evidence_audit/summary.json) | 50种冻结候选设置包含单参数扰动、16个联合角点与模块移除；置信区间为探索性逐点区间。 |
| R2.5 | 请求覆盖与PSNR变化<br>[completion_frontend_paired.csv](R2C5/existing_source_tables/completion_frontend_paired.csv) · [synthesis_diagnostics.csv](R2C5/existing_source_tables/synthesis_diagnostics.csv) · [evidence_audit.json](R2C5/evidence_audit.json) | 原37例配对LaMa数据诊断固定后端下的覆盖变化；不能证明生成先验受损，253请求扩展未新增PSNR。 |
| R2.6 | 错误唯一返回的解析机制<br>[resolver_causal_audit.json](R2C6/resolver_causal_audit.json) · [resolver_independent_audit.json](R2C6/resolver_independent_audit.json) · [wrong_unique_diagnostics.csv](R1C6_Handoff253/statistics/wrong_unique_diagnostics.csv) · [summary.json](R1C6_Handoff253/mechanism_analysis/summary.json) | 含原样对照的60个解析器单元场景验证单一名称匹配且缺乏独立质量否决的决策机制；不是上游分割干预，也不是新增总体性能估计。 |
| R2.7 | 门控符号 | 仅正文符号修改：式(1)使用v_sem、v_str、v_app，式(4)保留g*表示解析后的Group ID；没有新增实验数据。 |
| R2.8 | 根掩模瓶颈<br>[root_robustness_summary.csv](R2C8_Root226/root_robustness_summary.csv) · [root_robustness_paired.csv](R2C8_Root226/root_robustness_paired.csv) · [automatic_root_iou_strata.csv](R2C8_Root226/automatic_root_iou_strata.csv) | 226例根掩模替换是诊断对照，不是数学上界；与42目标完整图像评测分开解释。 |

## 先前已公开的门控实验

八子集Group门控数据位于[旧证据归档](../revision_20261008/evidence/gate_ablation/)，此处直接链接，不重复复制。可依次查看[协议](../revision_20261008/evidence/gate_ablation/group_gate_protocol.json)、[2144条案例×条件记录](../revision_20261008/evidence/gate_ablation/group_gate_cases.csv)、[汇总](../revision_20261008/evidence/gate_ablation/group_gate_summary.csv)及[解释与复现报告](../revision_20261008/evidence/gate_ablation/gate_report.txt)。226例与42例分别使用各自历史版本的Group实现，不应合并为同一版本结果。

## 阅读与复用

- CSV保存逐例或汇总数值，JSON保存协议、核验及来源记录；[共享代码](shared_code/)提供分析所用源码快照。
- 本目录不含原始照片、掩模/图像池、模型权重或运行环境。记录用于数值审核与统计重算；逐像素重跑仍需另行取得具有授权的输入、冻结实现及依赖环境。
- 公开副本替换本机路径前缀，保留时间戳与历史证据哈希；PUBLICATION_MANIFEST.json记录原始与公开文件的双SHA256。脚本需重新绑定输入路径；早期完成记录反映写入时的状态，不代表最终论文状态。
- 正确性、选项数量、根掩模质量、分割质量和图像重建是不同终点；跨队列比较前须读取各自协议与分母。

## 发布完整性与获取方式

本目录含260个科学证据源文件及发布元数据。[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)列出原始文件与公开派生文件的双SHA256；[FILE_MANIFEST.json](FILE_MANIFEST.json)用于校验实际公开文件；[original_source_manifest.json](original_source_manifest.json)保留原始源文件校验值。

公开副本仅替换本机路径前缀：__WORKSPACE__表示分析工作区，__RUNTIME_ROOT__表示早期推理环境，__USER_HOME__表示原用户环境。实验数值、案例标识、时间戳和历史证据哈希保持不变；历史哈希仍指向原始证据，不指向去标识后的公开字节。归档脚本需重新绑定路径并另行取得依赖及具备授权的输入，不能视为自包含推理环境。

仓库现有[Apache-2.0许可](../../../LICENSE)保持不变；原始数据集、图像及模型权重的权利仍属于相应提供方，本次不再分发这些内容，也不将仓库许可扩展到它们。引用时可使用本目录的固定提交链接、仓库名称、作者Jianpeng Chen和Yu Huang及修订日期2026年10月10日；未声称具有DOI。
