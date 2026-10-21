# 本次论文数据

论文：**Object-Conditioned HPID-Split: Deterministic Part-Identity Fusion for Editable Assets**  
作者：Jianpeng Chen、Yu Huang  
整理日期：**2026年10月21日**

本目录按当前英文正文及补充材料整理，涵盖原始实验、本轮补充实验和失败诊断。各实验与论文位置的对应关系见[数据索引](DATA_INDEX.md)。

[补充材料 PDF](Supplementary_Material_for_Review.pdf) 包含本稿的补充方法、表格和图。

- `original/`：开发集与测试集清单、226例基础实验、身份审计、参数分析、Group回归、42例holdout、37例修复实验及相关诊断。
- `revision/`：全图评估、PartCATSeg、253请求路由、分类体系、可靠性、权重、根掩码与解析器实验。
- `review_burden37/`：原37请求的待复核选项分析。
- `stability/`：输入扰动后的目标保留分析。
- `auxiliary/`：补充材料仍使用的角色、Gaussian/PAGA及v0.3.8工程审计。

不属于当前论文的旧开发对照、重复发布和编辑记录已从仓库当前目录移除。本稿仍使用的历史实验保留其真实日期、版本和结果，实验数值未改动。1,951候选的fine-ID实验与2,188候选的Group回归是同226幅图像上的不同预测池；253请求由原37请求及另216请求组成，不能视为253次重新运行的修复实验。

[MANIFEST.json](MANIFEST.json)提供文件校验信息，[PROVENANCE.json](PROVENANCE.json)记录原始与公开副本的校验值。原始照片、第三方标注、完整候选掩码及模型权重按各自来源和许可获取。本目录提供数值结果与分析记录；冻结脚本的完整运行仍需要对应输入、依赖及版本。
