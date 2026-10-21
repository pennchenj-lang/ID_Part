# HPID-Split

**Object-Conditioned HPID-Split: Deterministic Part-Identity Fusion for Editable Assets**  
Jianpeng Chen、Yu Huang

## 本次论文数据与补充材料

本仓库提供当前论文的数据、补充材料与实现代码。下列数据对应[2026年10月21日投稿定稿版本](https://github.com/pennchenj-lang/ID_Part/tree/96416d8a5c42497b33540f0b620eaebb7eeab821/paper_data)。

**[补充材料 PDF](paper_data/Supplementary_Material_for_Review.pdf)** · **[实验与论文对应索引](paper_data/DATA_INDEX.md)** · [数据说明](paper_data/README.zh-CN.md) · [English](README.md)

| 数据目录 | 内容 |
| --- | --- |
| [原始实验](paper_data/original) | 数据集清单、226例基础实验、身份审计、参数分析、Group回归、42例holdout、原37请求修复实验及相关诊断 |
| [修回补充实验](paper_data/revision) | 全图评估、PartCATSeg对照、253请求路由、分类体系、可靠性、权重、根掩码与解析器分析 |
| [原37请求的待复核选项](paper_data/review_burden37) | 选项数量、配对比较及正确性核查 |
| [输入扰动稳定性](paper_data/stability) | 受控输入变化后的目标保留和分割结果 |
| [辅助实验](paper_data/auxiliary) | 补充材料中的角色、Gaussian/PAGA实验及v0.3.8工程审计 |

[数据索引](paper_data/DATA_INDEX.md)列出25类实验与论文章节、表格及源文件的对应关系。各实验保留真实的数据集范围和实现版本。

1,951候选的fine-ID实验与2,188候选的后期Group回归使用同226幅图像上的不同预测池。253请求由原37请求与另216请求组成。

## 文件校验与获取

- [文件清单](paper_data/MANIFEST.json)：大小和SHA-256校验值。
- [来源记录](paper_data/PROVENANCE.json)：原始文件与公开副本的校验值。
- [数据集信息](paper_data/DATASET.json)：对应稿件版本及实验映射。

下载仓库后，可运行以下命令核验文件：

```bash
python paper_data/verify_data.py
```

原始照片、第三方标注、完整候选掩码及模型权重按各自来源和许可获取。冻结分析脚本需要相应协议列明的输入和依赖。

## 实现代码

代码见[src](src)，配置见[configs](configs)，脚本见[scripts](scripts)，测试见[tests](tests)。另见[算法说明](docs/ALGORITHM.md)和[输出格式](docs/OUTPUT_FORMAT.md)。

代码许可及第三方材料条款分别见[LICENSE](LICENSE)和[THIRD_PARTY](THIRD_PARTY.md)。
