# SkillSight 复现交付说明

本交付包对应 SkillSight 在单张 NVIDIA RTX 4090 上的复现实验，日期为 2026-11-08。

读者可以先看本文件，再按需查看结果明细和完整英文报告。

## 交付范围

本轮完成了离线检索和 Qwen3-4B 端到端实验。

已完成的离线内容：

- SRA-Bench 主检索结果；
- Dense、SBC、LEC 和融合方法消融；
- Qwen3、SkillRouter、SkillRet 三种 embedding；
- SkillBench Easy、Hard 配置；
- 背景能量、token 证据和检索深度分析；
- 普通文本对照；
- 候选池、beta 和背景秩敏感性。

已完成的端到端内容：

- Qwen3-4B-Instruct-2507；
- SRA-Bench 六项任务；
- SkillSight top-3 技能注入；
- 单卡检索和推理分阶段运行。

本轮未复现：

- Llama-3.1-8B 端到端结果；
- GPT-5.4-mini 端到端结果；
- 同场 LLM Direct、LLM Select 和 Progressive Disclosure 基线。

Qwen3-4B 六任务总体准确率为 60.02%，论文为 61.13%。

## 主要结果

### SRA-Bench 离线主结果

| Embedding | Dense R@10 | SkillSight R@10 | 论文 SkillSight R@10 |
| --- | ---: | ---: | ---: |
| Qwen3 | 66.09% | 86.13% | 86.23% |
| SkillRouter | 58.25% | 79.87% | 79.81% |
| SkillRet | 78.37% | 85.68% | 86.42% |

Qwen3 主方法的 Recall@10 差 0.10 个百分点。

SkillRouter 主方法差 0.06 个百分点。

SkillRet 主方法差 0.74 个百分点。

### SRA-Bench 消融

| 方法 | 复现 Recall@10 | 论文 Recall@10 |
| --- | ---: | ---: |
| Dense | 66.09% | 66.02% |
| Dense + LEC | 79.85% | 79.76% |
| SBC only | 78.37% | 78.23% |
| LEC only | 72.95% | 71.46% |
| SkillSight | 86.13% | 86.23% |

`LEC only` 的候选池定义没有在正文中完整说明。这个差异保留为待核事项，没有通过调参掩盖。

### SkillBench-Supp

| 配置 | Dense R@10 | SkillSight R@10 |
| --- | ---: | ---: |
| Hard，75 任务 | 55.48% | 61.29% |
| Easy，75 任务 | 58.33% | 63.68% |
| 论文 | 56.56% | 64.24% |

论文描述约 77k 候选技能。当前公开数据包含 78,361 条 Easy 技能和 79,141 条 Hard 技能。

本轮没有找到与论文完全一致的候选池快照，因此 SkillBench 结果标记为当前公开数据下的近似复现。

### Qwen3-4B 端到端

| 任务 | 正确数 | 准确率 | 论文准确率 |
| --- | ---: | ---: | ---: |
| TheoremQA | 489 / 747 | 65.46% | 68.40% |
| LogicBench | 598 / 760 | 78.68% | 80.53% |
| MedCalc-Bench | 899 / 1,100 | 81.73% | 80.91% |
| CHAMP | 169 / 223 | 75.78% | 73.99% |
| BigCodeBench | 523 / 1,140 | 45.88% | 47.02% |
| ToolQA | 563 / 1,430 | 39.37% | 41.05% |
| 总体 | 3,241 / 5,400 | 60.02% | 61.13% |

ToolQA 使用 ReAct 和外部语料。其余五项任务使用 direct engine。

## 分析结论

SRA-Bench 的 Dense 平均有 1.240% 的文档排在最佳 gold 技能之前。论文报告值为 1.24%。

SBC 将同一指标降到 0.220%。

Qwen3 的文档平均背景能量为 0.3758，查询平均背景能量为 0.0813。文档能量约为查询能量的 4.62 倍。

ArguAna、FiQA 和 SciFact 的 gold-above 比例分别为 0.39%、0.14% 和 0.41%。这些值都落在论文报告的 0.05%～0.42% 区间。

普通文本的文档与查询能量比为 1.20～1.39，明显低于 SRA-Bench 的 4.62。这支持论文关于技能文档共享背景的判断。

## 环境

关键环境如下：

| 项目 | 版本 |
| --- | --- |
| GPU | NVIDIA RTX 4090，24,564 MiB |
| 显卡驱动 | 595.80 |
| SkillSight Python | 3.11.18 |
| PyTorch | 2.6.0+cu124 |
| sentence-transformers | 5.5.1 |
| NumPy | 2.4.6 |
| vLLM Python | 3.10.22 |
| vLLM | 0.31.0 |
| vLLM PyTorch | 2.13.0+cu130 |

完整依赖清单位于 `environment/`。

## 数据与模型版本

| 资产 | Revision |
| --- | --- |
| SRA-Bench | `6143f2634eb284955ce312213bac24b582d039f3` |
| SkillRouter-Eval-Core | `20a03920e7f08d76b67af367350d25ef4468198e` |
| Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| SkillRouter-Embedding-0.6B | `c03c9bcee9fce92ab0262bb6dcf54d174a8ba558` |
| SkillRet-Embedding-0.6B | `67f1594c46c40e13a5eb2f4092cc85c98c9ed32d` |
| Qwen3-4B-Instruct-2507 | 由 Hugging Face 公共仓库下载 |

模型权重、原始数据集和 embedding 缓存不放入交付包。

这些文件体积大，可按 `README.md` 中的下载命令重新获取。

## 交付目录

交付包包含以下内容：

```text
SkillSight_Reproduction_2026-11-08/
├── README.zh-CN.md
├── MANIFEST.json
├── environment/
├── reports/
├── source/
├── results/
├── analysis/
└── logs/
```

各目录用途：

- `environment/`：依赖快照、系统信息、Git 提交和 requirements；
- `reports/`：中文交付说明和英文详细报告；
- `source/`：对应 Git 提交的源码快照；
- `results/`：离线与端到端评测的 JSON、CSV 和 summary；
- `analysis/`：能量、token、深度、普通文本和超参数分析；
- `logs/`：关键运行日志。

## 已知差异

正文 Eq. 15 与仓库默认融合方式不一致。

仓库默认 `skillsight` 使用 `lam=None`。正文公式对应 `lam=1.0`。

当前默认实现更接近论文主表数值，因此报告中保留默认实现，并单列 `skillsight_eq15` 作为诊断。

SR-Agents 当前推理输出没有逐题 latency 和 token 元数据。

端到端汇总保留检索延迟，空出推理延迟和 token 数。

没有根据总量估算或补写缺少的字段。

## 使用方式

查看源码提交和自动测试：

```bash
cd source/SkillSight
git status
python -m unittest discover -s tests -v
```

重新执行主要实验时，先按仓库根目录的 `README.md` 下载模型和数据，再使用报告中列出的命令。

文件完整性可用以下命令检查：

```bash
sha256sum -c SHA256SUMS.txt
```

GitHub 地址：

```text
https://github.com/QMJZR/SkillSight
```
