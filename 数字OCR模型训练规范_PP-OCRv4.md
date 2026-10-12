# PP-OCRv4 数字识别模型训练与部署工程规范

> **适用域**：PP-OCRv4_mobile_rec / PP-OCRv5 识别分支 + 字符集 `{0,1,2,3,4,5,6,7,8,9,.,-}`（cardinality = 12）+ PaddleX 训练/推理框架 + 预训练权重迁移学习 + 数据集规模 1k ~ 50k samples + 单 GPU 训练
> **版本**：v2.0（严谨工程版）
> **更新**：2026-08-20

---

## 目录

**Part 1 · 核心理论层**（形式化定义与算法原理）
- 1.1 端到端文字识别架构定义
- 1.2 PP-OCRv4 MultiHead 双分支结构：CTCHead + NRTRHead
- 1.3 CTC（Connectionist Temporal Classification）形式化表述
- 1.4 NRTR（NRTR/GTC 自回归）分支原理
- 1.5 字符字典（Vocabulary / Label Mapping）机制与一致性约束
- 1.6 优化器与学习率调度：Cosine Annealing + Warmup

**Part 2 · 配置规范层**（参数分区、合理区间、偏离 → 根因映射）
- 2.1 配置生效预校验：INFO 日志 8 锚点一致性检查
- 2.2 Group A · 全局静态参数（Global 块，单次配置）
- 2.3 Group B · 优化器与训练超参数（Optimizer / Train 块，调参主区）
- 2.4 Group C · 数据集与采样器参数（Dataset / Sampler 块）
- 2.5 Group D · 多目标 Loss 权重
- 2.6 Group E · 评估与 Checkpoint 持久化参数
- 2.7 Group F · 运行时可观测指标（日志只读，诊断上游配置）

**Part 3 · 运行时诊断层**（监控指标定义 + 决策树）
- 3.1 日志指标定义与量纲
- 3.2 CTCLoss / NRTRLoss 分段健康区间表
- 3.3 学习率调度曲线对照（Cosine + Linear Warmup）
- 3.4 训练异常诊断决策树（Diagnostic Decision Tree）
- 3.5 典型异常栈映射表

**Part 4 · 工程流程层**（Pipeline SOP）
- 4.1 项目目录结构与模块职责
- 4.2 数据集校验算法（7 项形式化约束）
- 4.3 数据集划分协议（顺序约束 + 分层采样 + 泄露预防）
- 4.4 训练启动 SOP
- 4.5 训练终止判据（Early Stopping）
- 4.6 Inference 模型导出与归档规范
- 4.7 推理质量评估协议（三桶分箱法：incorrect / incorrect_high_conf / correct_low_conf）
- 4.8 推荐数据集方案与合成数据生成算法
- 4.9 端到端交付 CheckList

---

---

# Part 1 · 核心理论层

## 1.1 端到端文字识别架构定义

### 1.1.1 任务形式化表述

给定输入图像 `I ∈ R^(H×W×3)` 与目标文本序列 `y = (c₁, c₂, …, c_L)`，其中 `c_i ∈ V`，`V` 为有限字符表（Vocabulary），文字识别模型需学习条件概率分布 `P(y | I)` 并在推理阶段求解：

```
ŷ = argmax_y P(y | I)
```

### 1.1.2 PP-OCRv4_mobile_rec 架构组件

| 组件 | 实现类 | 输入维度 | 输出维度 | 作用 |
|---|---|---|---|---|
| Backbone | PPLCNetV3 (scale=0.95) | `[B, 3, 48, 320]` | `[B, 120, T]`，T = W/4 ≈ 80 | 视觉特征提取：将像素空间映射为序列特征图，沿宽度维度展开为时间步序列 |
| CTC Neck | SVTR Tiny (depth=2, dims=120, hidden=120) | Backbone 输出 | `[B, 120, T]` | 对特征图做轻量序列建模，输出送入 CTC Head |
| CTC Head | Linear + Softmax | `[B, 120, T]` | `[B, T, |V|+1]` | 逐时间步分类：每个时间步输出 `|V|+1` 类概率（+1 为 CTC blank token） |
| NRTR Head | Transformer Decoder (nrtr_dim=384, max_text_length=M) | Backbone 输出 + 前缀 token | `[B, M, |V|+3]` | 自回归建模：`P(c_t | c₁…c_{t-1}, I)`；+3 含 BOS / EOS / PAD |
| PostProcess | CTCLabelDecode | CTC 概率矩阵 `[T, |V|+1]` | `(rec_text, rec_score)` | 贪心解码：合并连续重复 token → 移除 blank → 查表映射为字符串 |

> **关键约束**：推理阶段仅使用 CTC Head + CTCLabelDecode 输出最终结果；NRTR Head 仅参与训练阶段的多任务学习，不参与推理。

---

## 1.2 PP-OCRv4 MultiHead 双分支结构

PP-OCRv4 采用 **MultiLoss** 加权训练范式：

```
L_total = w_ctc · L_ctc + w_nrtr · L_nrtr
```

| 分支 | Loss 函数 | 梯度路径 | 收敛速度 | 推理使用 |
|---|---|---|---|---|
| CTC | CTCLoss（基于 Forward-Backward 算法的最大似然） | 各时间步独立 → 路径短 | 快（ep3 可收敛至 0.1 以下） | ✅ 是（CTCLabelDecode） |
| NRTR / GTC | NRTRLoss（每步 teacher forcing 交叉熵均值） | 自回归 t=1..M 链式依赖 → 路径长 | 慢（CTC 的 3~10 倍迭代数） | ❌ 否（仅作训练时正则项，间接提升 Backbone 表征质量） |

**设计动机**：
- CTC 提供快速收敛的强监督信号，保证推理阶段的序列解码精度。
- NRTR 通过自回归引入长程依赖约束，作为辅助训练目标（Auxiliary Objective）改善特征空间的判别性，对最终推理精度提供 0.5%~2% 的边际增益。

---

## 1.3 CTC（Connectionist Temporal Classification）形式化表述

### 1.3.1 标注对齐问题

训练时输入长度为 T 的特征序列与长度为 L 的标签序列无显式对齐（`T >> L`，典型 T≈80，L≤8）。CTC 引入 **blank token（ε）** 构造中间路径 π ∈ (V ∪ {ε})^T，并定义多对一映射 B：折叠重复字符后移除 ε。

### 1.3.2 目标函数

```
L_ctc = -ln Σ_{π: B(π)=y}  Π_{t=1..T}  P(π_t | I)
```

即：所有经 B 映射后等于标签 y 的合法路径的概率乘积之和，取负对数。数值实现采用动态规划 Forward-Backward 算法，复杂度 O(T·L)。

### 1.3.3 解码（推理）

**贪心解码（Greedy Decode）**：逐时间步取 `argmax P(c_t)` 得到 `π*`，执行 B(π*) 得到最终序列。PP-OCRv4 默认使用贪心解码（不使用 Beam Search）。

### 1.3.4 置信度（rec_score）

```
rec_score = mean( P(π*_t) ) ，t ∈ π* 中非 blank 且非重复合并的位置
```

即：构成最终解码字符的各时间步最大概率值的均值，范围 (0, 1]，数值越大表示模型越确信。

---

## 1.4 NRTR（NRTR / GTC）分支原理

### 1.4.1 任务定义

NRTR（基于 Transformer Decoder 的自回归识别）在训练阶段使用 **Teacher Forcing**：

```
P(c_t | I, c₁, …, c_{t-1}) = Softmax(W · Decoder_t(E(c₁…c_{t-1}), Enc(I)))
```

其中：
- `Enc(I)`：Backbone 输出的视觉编码（Encoder 侧）
- `E(·)`：字符嵌入层（Embedding，shape `[|V|+3, nrtr_dim]`）
- `Decoder_t`：Transformer Decoder 第 t 步输出
- `W`：分类头（nrtr_dim → |V|+3）

### 1.4.2 Loss 定义

```
L_nrtr = (1 / M_eff) Σ_{t=1..M} CrossEntropy(P(c_t), one_hot(y_t))
```

其中 `M_eff` = 有效 token 数（忽略 PAD 填充位）。由于每步依赖前序真实 token 的梯度回传，L_nrtr 梯度路径长度为 O(M)，收敛慢于 L_ctc 的 O(1) 逐时间步独立梯度，属正常现象。

### 1.4.3 特殊 token 定义（NRTR 内部）

| token | 索引 | 含义 |
|---|---|---|
| BOS（Begin of Sequence） | `|V|` | 序列起始，第 1 步输入 |
| EOS（End of Sequence） | `|V|+1` | 序列结束，输出此 token 即停止解码 |
| PAD | `|V|+2` | 填充位，`max_text_length` 对齐时使用，Loss 计算中 mask |

因此 NRTR 层 Embedding 维度为 `|V|+3`（12 类字符 → 15 类 Embedding；BOS/EOS/PAD 3 类 + 可选 blank → shape `[16, 384]`，与日志中 shape mismatch 吻合）。

---

## 1.5 字符字典（Vocabulary / Label Mapping）机制与一致性约束

### 1.5.1 字典定义

字符字典 `dict.txt` 是一一映射 `f: Index → Character` 的序列化形式。设 `|V| = C`，则字典共 C 行，第 i 行（0-indexed）存字符 `f(i)`。

**纯数字场景规范字典**（C = 12）：
```
行 0 → 字符 '0'   （索引 0）
行 1 → 字符 '1'   （索引 1）
…
行 9 → 字符 '9'   （索引 9）
行 10 → 字符 '.'  （索引 10）
行 11 → 字符 '-'  （索引 11）
```

**验证方式**：
- 行数：`wc -l dict.txt` == C
- 无 BOM / 无空行 / 无 Windows 换行（`\r\n`）
- CTC Head 输出维度：`head.ctc_head.fc.weight.shape[1]` == C（训练日志 shape mismatch 警告中应出现 `[120, C]`）

### 1.5.2 训练 - 推理一致性约束（Hard Constraint）

**定理**：若训练阶段使用字典 `f_train`，推理阶段使用字典 `f_infer`，且存在索引 i 使得 `f_train(i) ≠ f_infer(i)`，则模型输出正确字符序列的概率为 0（或退化为随机乱码，表现为高置信度错误）。

**证明思路**：CTC Head 分类层（Linear）在训练时学习的第 i 类决策边界，其语义由 `f_train(i)` 的监督信号锚定。若推理阶段将该决策边界解释为 `f_infer(i) ≠ f_train(i)`，则分类结果与监督目标错位，故输出不可用。

**三类字典故障的形式化后果**：

| 故障类型 | 形式化条件 | 训练期表现 | 推理期表现 |
|---|---|---|---|
| **字符集不足** | ∃ y 的字符 c ∉ dom(f) | L_ctc 无法构造合法路径，长期停留在 2~3；偶发 "unknown char" 告警 | — |
| **行序错位** | |dom(f_train)| = |dom(f_infer)| = C，∃i 使 f_train(i) ≠ f_infer(i) | L_ctc、train acc、val acc 全部正常（训练自洽） | incorrect_high_conf 显著增加，错误模式为固定字符间互换（如 7↔8） |
| **基数不一致** | C_train ≠ C_infer（典型：训练 C=12，推理默认 C=6625） | — | CTCLabelDecode 阶段 `IndexError: index out of range`；或静默输出非预期字符 |

> **工程强约束**：训练配置（`Global.character_dict_path`）、Inference 归档目录、推理脚本（`rec_char_dict_path`）三者必须指向字节级完全一致的同一文件（md5sum 校验）。

---

## 1.6 优化器与学习率调度：Cosine Annealing + Linear Warmup

### 1.6.1 优化器

默认使用 **Adam**（β₁ = 0.9，β₂ = 0.999，ε = 1e-8），L2 正则因子 `regularizer.factor = 3e-5`（仅作用于未显式设置 ParamAttr.regularizer 的参数）。

### 1.6.2 学习率调度函数

记总 epoch 数为 `E`，warmup epoch 数为 `W`，base_lr 为 `η_max`，当前训练进度（按 step 线性插值）为 `r ∈ [0, 1]`，warmup 进度为 `r_w = r · E / W`：

```
         ⎧  η_max · r_w                                   当 r ≤ W/E   (warmup 期)
η(r)  =  ⎨
         ⎩  0.5 · η_max · (1 + cos(π · (r - W/E) / (1 - W/E)))    当 r > W/E   (cosine 衰减期)
```

### 1.6.3 参数推荐比例

- `W/E ≈ 1/4`：典型配置 E=20 → W=5；E=50 → W=10
- 微调预训练：`η_max ∈ [5e-4, 2e-3]`，默认 1e-3
- 从头训练：`η_max ∈ [3e-3, 8e-3]`

---

---

# Part 2 · 配置规范层

> 变更策略：单轮实验仅允许改变 1 项参数，观察 1~2 epoch 验证效果后再继续；严禁多参数联动变更导致归因失败。

## 2.1 配置生效预校验：INFO 日志 8 锚点一致性检查

训练启动阶段 PaddleOCR 输出 `ppocr INFO` 结构化配置树。以下 8 项校验 **全部通过** 后，方可确认配置生效并等待训练：

| # | INFO 路径 | 预期值（数字识别场景） | 失败根因 |
|---|---|---|---|
| G1 | `Global.character_dict_path` | 非空绝对路径，指向 dict.txt（C=12） | 空值 → 回退默认中文 6625 类字典 |
| G2 | `Global.use_gpu` ∧ `device Place(...)` | `True` ∧ `Place(gpu:0)` | False / Place(cpu) → 硬件未利用 |
| G3 | `Global.epoch_num` | 与命令行 `-o Global.epoch_num` 数值一致 | 不一致 → `-o` 路径拼写错误（如误写 `Train.epochs_iters`） |
| G4 | `Global.pretrained_model` + 后续 `load pretrain successful` | 非空（URL 或本地 .pdparams） | 空值 → 随机初始化，收敛速度降低 5~10× |
| G5 | `Train.dataset.data_dir` + `label_file_list[0]` | 路径可 `os.path.exists()` | 路径错误 → 立即 `FileNotFoundError` |
| G6 | `Train.dataset.name` | ≤50k samples：`TextRecDataset` | 仍为 `MSTextRecDataset` → 与 MultiScaleSampler 组合触发递归采样失败 |
| G7 | `Train.loader.batch_size_per_card` | ∈ [16, 64]（显存占用 60%~85%） | = 1 → BN 统计量不稳定，吞吐量下降 4~10× |
| G8 | `Train.sampler.name` | ≤50k ∨ 极端宽高比：`BatchSampler` | 仍为 `MultiScaleSampler` → 宽高比落在 scales 外部的样本将被系统性跳过 |

---

## 2.2 Group A · 全局静态参数（Global 块，单次配置）

| # | YAML 完整路径 | 语义 | 合理区间 | 偏离诊断 | 修正操作 |
|---|---|---|---|---|---|
| A1 | `Global.character_dict_path` | 索引→字符双射的序列化文件路径 | 必须指向训练时使用的 dict.txt（C=12，0~9 + . + -），绝对路径，UTF-8 无 BOM | ① 空/默认 → 推理错位，incorrect_high_conf 占比异常升高<br>② 标签字符不在字典 → L_ctc 停滞于 2~3<br>③ 推理维度越界 → `IndexError` | ① YAML 写死绝对路径<br>② 校验 dict.txt 行数 = C，并随机抽 50 条 train.txt 字符 ∈ dom(f)<br>③ 训练、导出、推理三阶段字典 md5 一致 |
| A2 | `Global.device` / `use_gpu` | 计算硬件选择 | `gpu:0` + `use_gpu: True` | ① CPU → ips ≤ 10 samples/s<br>② CUDA 版本不匹配 → 启动失败 | ① `-o Global.device=gpu:0`<br>② `paddle.utils.run_check()` + `nvidia-smi` 验证 |
| A3 | `Global.model_name` | 模型架构标识符 | 本场景：`PP-OCRv4_mobile_rec`（移动端精度/速度 Pareto 最优） | 误选 `_server_` → 显存 ×2、latency ×2~3，acc 提升 ≤1% | 保持 mobile 版 |
| A4 | `Global.pretrained_model` | 迁移学习初始化权重源 | 官方 URL 或本地 .pdparams 有效路径 | ① 空 → ep5 L_ctc > 2（收敛缓慢）<br>② 文件损坏 → `shape mismatch` 无 `load pretrain successful` | ① 保留默认预训练<br>② 清除缓存 `~/.paddleocr/models/*.pdparams` 后重下 |
| A5 | `Global.max_text_length` (M) | NRTR 序列最大长度（BOS→EOS 的最大步数） | 数字场景：标签最大位数 + 2（如 "12345.67" → M=8） | 默认 25 → EOS token 学习延迟，L_nrtr 偏高 20%+ | `-o Global.max_text_length=8`，与 train.txt 最大 len(label) 匹配 |
| A6 | `Global.use_space_char` | 是否在字典中注入空格字符 | 纯数字：`False`；若标签确实包含空格 → True，且 dict.txt 含空格行 | `True` ∧ 字典无空格 → 类别数 C 与 head 维度不匹配 | 匹配 dict.txt 内容 |

---

## 2.3 Group B · 优化器与训练超参数

| # | YAML 完整路径 | 语义 | 合理区间 | 偏离诊断 | 修正操作 |
|---|---|---|---|---|---|
| B1 | `Global.epoch_num` (E) | 总迭代 epoch 数 | 有预训练 + 10k~50k samples：20~50；无预训练：100~200 | ① <10 → 欠拟合终止<br>② >100 → 过拟合 + 资源浪费 | 满足 3.5 节 Early Stopping 判据即可终止 |
| B2 | `Optimizer.lr.learning_rate` (η_max) | base_lr，调度峰值 | 微调预训练：[5e-4, 2e-3]；从头训练：[3e-3, 8e-3] | ① <3e-4 → ep10 L_ctc > 0.5（欠学习）<br>② >5e-3（微调场景）→ loss NaN / 剧烈震荡 | 默认 1e-3；偏离 1 数量级后 ×2 / ÷2 粗调 |
| B3 | `Optimizer.lr.name` | 调度函数族 | Cosine（默认） | Step / Fixed → 后期震荡不收敛 | 保持 Cosine |
| B4 | `Optimizer.lr.warmup_epoch` (W) | warmup epoch 数 | ≈ E/4（E=20→W=5；E=50→W=10） | ① = 0 → 首轮 loss 飙升（预训练权重被大梯度破坏）<br>② > E/3 → 有效学习期不足 | 固定比例 E/4 |
| B5 | `Optimizer.name` | 优化器算法 | Adam（β₁=0.9, β₂=0.999） | 无 momentum 类 → 收敛速度 ×0.5 | 保持默认 |
| B6 | `Optimizer.regularizer.factor` (λ) | L2 正则权重衰减系数 | [1e-5, 1e-4]，默认 3e-5 | ① >5e-4 → L_ctc 偏高 1 量级（过正则）<br>② = 0 → ep20 后 train acc ≫ val acc（过拟合） | 默认 3e-5；过拟合征象出现后调至 1e-4 |
| B7 | `Train.loader.batch_size_per_card` (B) | 每 GPU 单 batch 样本数 | ∈ [16, 64]，目标显存占用 70%~85% | ① ≤8 → BN 统计偏差，吞吐量下降 4~8×<br>② >128（单 GPU）→ CUDA OOM | 从 32 开始倍增；OOM 时折半重试 |
| B8 | `Train.loader.num_workers` (N_w) | DataLoader 预取线程数 | [4, 8]（CPU 物理核数的 1/2） | ① = 0 → reader_cost / batch_cost > 30%（GPU 等数据）<br>② >16 → 上下文切换开销 + filelock 递归栈溢出 | reader_cost 占比 < 10% 时停止增加 |

---

## 2.4 Group C · 数据集与采样器参数

| # | YAML 完整路径 | 语义 | 合理区间 | 偏离诊断 | 修正操作 |
|---|---|---|---|---|---|
| C1 | `Train.dataset.name` | Dataset 实现类 | ≤50k ∨ 特殊宽高比：`TextRecDataset`；>100k 均匀比例：`MSTextRecDataset` | 小数据 + MSTextRecDataset + MultiScaleSampler → 匹配失败引发无限递归 | `-o Train.dataset.name=TextRecDataset`（与 C2 联动） |
| C2 | `Train.sampler.name` | Batch 采样策略 | ≤50k ∨ 特殊宽高比：`BatchSampler`；>100k：`MultiScaleSampler` | MultiScaleSampler + 数据宽高比 ∉ scales → 被系统性跳过，有效数据量下降；极端时 `__getitem__` 递归 900+ 次栈溢出 | `-o Train.sampler.name=BatchSampler`（与 C1 联动） |
| C3 | `Train.sampler.scales` | MultiScaleSampler 候选尺寸集合 | 仪表数字 2~4:1 → `[[160,48],[200,64]]`；BatchSampler 场景无此项 | scales 仅含 10:1 长条 → 实际有效样本过滤率 >30% | 先统计数据集 W/H 分布，按百分位设置 scales；或直接关闭 MultiScaleSampler |
| C4 | `RecAug` / `RecConAug.prob` (p_aug) | 数据增强执行概率 | RecAug: None；RecConAug.prob ∈ [0.3, 0.5] | ① p_aug = 1.0 → 前 3 epoch L_ctc 停滞不降（增强过强）<br>② p_aug = 0 → train acc - val acc > 15%（过拟合） | 默认值；val acc 低于预期时在 [0.3, 0.5] 内微调 |

---

## 2.5 Group D · 多目标 Loss 权重

| # | YAML 完整路径 | 语义 | 合理区间 | 偏离诊断 | 修正操作 |
|---|---|---|---|---|---|
| D1 | `Loss.loss_config_list.CTCLoss.weight` (w_ctc) | CTC 分支权重 | **固定 1.0**（推理使用 CTC 解码） | <0.5 → CTC 梯度被稀释，推理质量下降 | 保持 1.0 |
| D2 | `Loss.loss_config_list.NRTRLoss.weight` (w_nrtr) | NRTR 分支权重 | 默认 1.0；时间受限 [0.2, 1.0]；极致加速可设 0.0 | ① >2.0 → L_total 由 L_nrtr 主导，CTC 收敛变慢<br>② = 0.0 → 训练耗时降低 20~30%，val acc 边际下降 0.3%~1% | ① 优先精度：1.0<br>② 精度达标后：0.2 或 0.0 关闭 NRTR |

---

## 2.6 Group E · 评估与 Checkpoint 持久化参数

| # | YAML 完整路径 | 语义 | 合理区间 | 偏离诊断 | 修正操作 |
|---|---|---|---|---|---|
| E1 | `Global.eval_batch_epoch` | 评估间隔（epoch 数） | **1**（每 epoch 必评估一次 val 集） | >1 → 错过峰值，Early Stopping 滞后 N epoch | 保持 1 |
| E2 | `Global.save_epoch_step` | 存储间隔（epoch 数） | **1** | >1 → 中断后回退 N 个 epoch 重跑 | 保持 1 |
| E3 | `Global.save_inference_dir` | 训练结束自动导出 inference 模型目录 | 指定为绝对路径 | None → 训练结束需手动 `Global.mode=export` 二次执行 | 训练命令显式指定 `-o Global.save_inference_dir=/abs/path/to/infer` |
| E4 | `Global.cal_metric_during_train` | 是否计算 train metric | **True** | False → 无法通过 train acc - val acc 做过拟合监测 | 保持 True |

---

## 2.7 Group F · 运行时可观测指标（日志只读）

> 以下指标为 stdout 日志输出项，仅用于诊断上游配置正确性，不可直接设置。

| # | 日志 Key | 形式化定义 | 健康区间（本场景） | 异常 → 上游参数 |
|---|---|---|---|---|
| F1 | `lr` | 当前 step 调度后学习率 η(r) | warmup 线性 0→η_max；cosine 期按 1.6.2 公式递减至 ~0 | 曲线偏离公式 → B2 / B3 / B4 |
| F2 | `CTCLoss` | 1.3.2 节 L_ctc | ep1 ∈ [0.5, 2.0]；ep3 ∈ [0.05, 0.5]；ep10 ∈ [1e-3, 5e-2]；ep20 < 1e-3 | ① ep3 > 1.0 → A1 字典错 ∨ B2 η 太小<br>② 停滞不降 → A4 预训练空 ∨ C4 p_aug 过强 ∨ A1 字符超集 |
| F3 | `NRTRLoss` | 1.4.2 节 L_nrtr | ep1 ∈ [1.5, 3.0]；ep5 ∈ [0.6, 1.8]；ep15 ∈ [0.15, 0.6]；ep20 ∈ [0.05, 0.15] | ① ep5 > 1.5 → D2 w_nrtr 调至 0.2<br>② 长期 > 3 → A5 M 过长（NRTR 梯度链过长） |
| F4 | `train acc` | 训练集序列精确匹配率（`#(ŷ=y)/B`） | 仅用于与 val acc 做差：差值 < 5% 健康；> 15% 判定过拟合 | 差值过大 → C4 ↑ p_aug ∨ B7 ↑ bs ∨ B6 ↑ λ ∨ 补充数据 |
| F5 | `val acc` (eval 行) | 验证集序列精确匹配率 | ep5 ≥ 0.95；ep10 ≥ 0.98；ep20 ≥ 0.99 | ① < 0.90 → 首查 A1 字典一致性，次查 C4 增强强度、标签错误率<br>② 连续 3 epoch 不增长 → Early Stopping（终止训练） |
| F6 | `norm_edit_dis` | 归一化编辑距离 = 1 - (Levenshtein(ŷ, y) / max(|ŷ|, |y|)) | 与 val acc 同趋势；val NED ≥ 0.99 视为鲁棒 | NED 比 acc 低 >5% → 单字符混淆（0/8、1/7 等），针对性补对应样本 |
| F7 | `reader_cost / batch_cost` | DataLoader I/O 耗时占比 | < 10%（IO 健康） | > 30% → B8 N_w ↑（4→8）∨ 存储介质迁移至 SSD |
| F8 | `ips` | 吞吐率（samples / sec） | mobile + 单卡 + bs=32：80~300 | < 30 → F7 IO / A2 硬件（CPU 模式）/ B7 bs 过小 |
| F9 | `max_mem_allocated / total` | GPU 显存占用比 | 0.60 ~ 0.85 | ① < 0.40 → B7 bs ×2 直至达 0.70~0.80<br>② > 0.95 ∧ CUDA OOM → B7 bs ÷2 |
| F10 | `eta` | 预计剩余训练时长 | 参考 | 异常飙升 → F7 IO 拥塞 / GPU 降频 / 其他进程抢占算力 |

---

---

# Part 3 · 运行时诊断层

## 3.1 日志指标量纲与采样周期

| 指标 | 量纲 | 采样频率 | 打印频率（`Global.log_smooth_window=20`） |
|---|---|---|---|
| lr | 标量（1e-4 ~ 1e-3 量级） | 每 step | 每 `print_batch_step`=20 step |
| L_ctc / L_nrtr / loss | 负对数似然（0 ~ +∞，理想→0） | 每 batch | 20 step 滑动均值 |
| reader_cost / batch_cost | 毫秒 / batch | 每 batch | 20 step 滑动均值 |
| ips | samples / sec | 每 batch | 20 step 滑动均值 |
| train acc / NED | 比例 [0, 1] | 每 batch | 20 step 滑动均值 |
| val acc / NED | 比例 [0, 1] | 每 eval 周期 | 每 `eval_batch_epoch`=1 epoch |
| max_mem_allocated | GB | 每 step（峰值记录） | 每行日志 |

## 3.2 CTCLoss / NRTRLoss 分段健康区间

### 3.2.1 L_ctc（推理相关，核心指标）

| epoch | 区间 | 对应路径概率把握 e^{-L_ctc} | 状态判定 | 异常处置 |
|---|---|---|---|---|
| ep0 初始化 | [3.0, 5.0] | [0.7%, 50%] | head 随机初始化，正常 | — |
| ep1 末 | [0.5, 2.0] | [13.5%, 60.7%] | 预训练权重生效，快速下降 | ✅ |
| ep2~3 | [0.05, 0.5] | [60.7%, 95.1%] | CTC 决策边界基本形成 | ✅ |
| ep5~10 | [1e-3, 5e-2] | [95.1%, 99.9%] | 主训练目标收敛，关注 val acc | 无需继续追踪 L_ctc |
| ep10+ | < 1e-3（平台期） | ≥ 99.9% | 饱和，继续训练对 CTC 无增益 | 满足 Early Stopping 即终止 |

### 3.2.2 L_nrtr（辅助目标，不可与 L_ctc 直接比绝对值）

| epoch | 区间 | 状态判定 | 异常处置 |
|---|---|---|---|
| ep1 初 | [2.5, 4.0] | 接近均匀分布（C=12 理论下界 ln(12) ≈ 2.48） | ✅ |
| ep1 末 | [1.5, 3.0] | 习得首字符先验 | ✅ |
| ep2~5 | [0.6, 1.8] | 习得短上下文（如数字后接小数点） | ✅ |
| ep6~15 | [0.15, 0.6] | 习得位置规律（负号在首部、小数点唯一） | 仍 > 1.5 → D2 w_nrtr 降至 0.2 |
| ep15+ | [0.05, 0.15]（饱和） | 数字域上下文信息有限，无法进一步下降 | 忽略 |

> **对比量纲差异原因**：L_ctc = -ln(总路径概率)，单样本单值；L_nrtr 为 M 步每步交叉熵均值。二者取值域、物理含义均不同，不可比较绝对值大小。

---

## 3.3 学习率调度曲线对照（E=20, W=5, η_max=0.001）

| 训练进度（epoch 中点） | 调度阶段 | η 理论值 | 诊断点 |
|---|---|---|---|
| 0.5 | warmup | ≈ 0.00010 | warmup 启动斜率 |
| 1.0 | warmup | ≈ 0.00020 | |
| 2.0 | warmup | ≈ 0.00040 | |
| 3.0 | warmup | ≈ 0.00060 | 日志 0.000522（浮动 6% 内正常）✅ |
| 4.0 | warmup | ≈ 0.00080 | |
| 5.0（end） | warmup 峰值 | = 0.00100（应精确匹配 η_max） | 关键校验点 |
| 10.0 | cosine | ≈ 0.000854 | cos(π·5/15) = 0.5 → 0.5·0.001·(1.5) ≈ 0.00075；修正后见下 |
| 15.0 | cosine | ≈ 0.000250 | |
| 18.0 | cosine | ≈ 0.000076 | |
| 20.0（end） | cosine | ≈ 0.000005（→ 0） | |

> cosine 期 η(r) = 0.5 · 0.001 · (1 + cos(π · (r-0.25)/0.75))：r=10/20=0.5 → (0.5-0.25)/0.75 = 1/3 → cos(π/3) = 0.5 → η = 0.0005 · (1.5) = 0.00075（上表约 0.00085 为中点近似误差，实际按精确公式计算）

---

## 3.4 训练异常诊断决策树

```
启动期异常
│
├─ [RecursionError + simple_dataset.py:__getitem__ x900+]
│    └─ C2 = MultiScaleSampler ∧ C1 = MSTextRecDataset  →  根因：无匹配 scale 样本
│       处置：C1→TextRecDataset, C2→BatchSampler；或 pip install filelock==3.13.1
│
├─ [copy.deepcopy → filelock/_audit_fork_safety + RecursionError]
│    └─ filelock ≥ 3.14 版本与 deepcopy 审计钩子冲突
│       处置：pip install filelock==3.13.1
│
├─ [unrecognized arguments: Global.xxx=xxx]
│    └─ Shell 参数解析：多 KV 串写，未在每项前加 `-o`
│       处置：`-o A=B -o C=D -o E=F`，每项独立前缀
│
├─ [shape mismatch WARNING: head.ctc_head.fc.weight]
│    └─ [INFO] 后紧跟 `load pretrain successful`  →  正常（自定义 C ≠ 官方 6625，自动跳过 Head）
│       处置：忽略
│
训练期异常
│
├─ L_ctc 连续 5 epoch ≥ 2.0
│    ├─ ∃ 标签字符 ∉ dom(f)？           ── 是 ──→  A1 字典缺失字符
│    ├─ pretrained_model 为空？         ── 是 ──→  A4 配置预训练
│    └─ RecConAug.prob = 1.0？          ── 是 ──→  C4 概率降至 0.3
│
├─ loss = NaN / Inf
│    ├─ η_max > 0.01？                 ── 是 ──→  B2 ×0.1 从 1e-4 起步
│    └─ ∃ 空标签？                     ── 是 ──→  数据校验 11.2 第 3 项
│
├─ train acc 0.9999 ∧ val acc < 0.80
│    └─ 过拟合
│       处置：C4 ↑ p_aug，B6 λ 调 1e-4，B7 bs ×2，补充数据，启用 Hard Example Mining
│
评估后异常
│
├─ val acc 连续 3 epoch 不增长
│    └─ Early Stopping：终止训练，取 best_accuracy.pdparams
│
├─ val acc < 0.90
│    └─ 首查 A1：train dict vs val dict 是否字节级一致；次查标签错误率；再查增强是否破坏可读性
│
推理期异常
│
├─ IndexError: index out of range（rec_postprocess.decode）
│    └─ A1 推理字典类别数 < 模型 head 维度
│       处置：训练 / 推理使用同一 dict.txt
│
└─ incorrect_high_conf / incorrect 比例 > 30%
     └─ ① 字典顺序错位（字符互换类错误）
        ② 对应类别样本不足
        处置：抽 10 张手工核对；若为系统性互换 → 修正 A1；否则 Hard Example Mining 补入训练集
```

---

## 3.5 典型异常栈映射表

| 异常关键信息（正则匹配） | 根因概率分布 | 标准处置 |
|---|---|---|
| `__getitem__` 被回溯 `>500` 次 + `RecursionError` | 95%：MultiScaleSampler 无匹配宽高比；5%：其他 | C1→TextRecDataset, C2→BatchSampler；filelock 降级 3.13.1 |
| `CUDA out of memory` | 100%：B7 bs 超显存 | B7 ÷2 直至 F9 ∈ [60%, 85%] |
| `IndexError` at `rec_postprocess.*decode` | 90%：推理字典 C 与训练不一致；10%：use_space_char 不匹配 | 三阶段字典 md5sum 对齐 |
| `label contains unknown character` | 100%：标签字符 ∉ dom(f) | 更新字典或清洗标签 |
| CTCLoss 平台期 ≥ 2 | 70%：字典字符超集；20%：预训练为空；10%：增强过强 | A1 → A4 → C4 顺序排查 |

---

---

# Part 4 · 工程流程层

## 4.1 项目目录结构与模块职责

```
paddlex_ppocr_rec_predict/
├── demo/
│   ├── demo.py                          # 单图推理最小验证：create_model / predict / save
│   ├── general_ocr_rec_001.png          # 样例输入
│   └── output/
│       ├── *_res.png                    # 标注叠加输出
│       └── res.json                     # 结构化结果（rec_text / rec_score / bbox）
│
├── test/
│   ├── test_single_ppocrv5.py           # 单数据集批量评估 CLI：--model_dir --input_dataset_dir --output_result_dir
│   ├── test_batch.py                    # 多数据集回归评估：读取 JSON 配置 → 分测试集独立输出
│   ├── test_batch_config_example.json   # 配置 Schema：{model{model_name, model_dir}, output_result_dir, test_sets[{input_dataset_dir}]}
│   └── result/<run_id>/
│       ├── <test_set_id>/
│       │   ├── incorrect/               # 桶 1：预测 ≠ 真值，全量（含高/低置信度）
│       │   │   ├── <img>_res.jpg        # 标注图
│       │   │   └── info.txt             # 格式：文件名 真值 预测 置信度（固定宽度对齐）
│       │   ├── incorrect_high_conf/     # 桶 2：预测≠真值 ∧ rec_score ≥ τ（默认 τ=0.8）
│       │   ├── correct_low_conf/        # 桶 3：预测=真值 ∧ rec_score < τ
│       │   └── summary.txt              # 单测试集统计：总数/准确率/三类计数
│       └── global_summary.txt           # 跨测试集总体汇总
│
└── script_verification/
    ├── test_img/                        # 冒烟测试样本集（21 张，1~2 位整数仪表图）
    └── test.txt                         # Tab 分隔标注：<相对路径>\t<标签>
```

### 4.1.1 核心模块职责矩阵

| 模块 | 输入 | 输出 | 调用方 |
|---|---|---|---|
| demo.py | 单图像路径 | 终端打印 + 标注图 + JSON | 开发自测、环境验证 |
| test_single_ppocrv5.py | `--model_dir --input_dataset_dir --output_result_dir` | 三桶分箱 + summary.txt | 单数据集精度评估 |
| test_batch.py | JSON 配置 + `--conf_threshold --batch_size` | 多测试集结果目录树 + global_summary.txt | 回归测试、跨数据集对比 |

---

## 4.2 数据集校验算法（7 项形式化约束）

**输入**：标注文件集合 `L = {train.txt, val.txt, test.txt}`，每行格式 `<rel_path>\t<label>`，基准目录 `D_root`，字典 `f: [0, C-1] → V`。

**输出**：布尔值 `VALID` + 违规样本集合 `B`。

**算法步骤**：

```python
def validate(L, D_root, f, M):
    V_set = set(f.values())
    B = []
    for label_file in L:
        for i, line in enumerate(open(label_file), 1):
            # Constraint 1: 格式 — 必须为 Tab 分隔 2 列
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 2:
                B.append(('FORMAT', label_file, i, line)); continue
            rel_path, label = parts[0].strip(), parts[1].strip()
            abs_path = os.path.join(D_root, rel_path)
            # Constraint 2: 图像存在
            if not os.path.isfile(abs_path):
                B.append(('MISSING', label_file, i, abs_path)); continue
            # Constraint 3: 图像可解码
            try:
                img = Image.open(abs_path); img.verify()
            except Exception:
                B.append(('DECODE', label_file, i, abs_path)); continue
            # Constraint 4: 标签非空
            if len(label) == 0:
                B.append(('EMPTY_LABEL', label_file, i)); continue
            # Constraint 5: 标签字符 ⊆ dom(f) 的值域 V_set
            if not set(label) <= V_set:
                B.append(('OUT_OF_VOCAB', label_file, i, label)); continue
            # Constraint 6: 标签长度 ≤ M
            if len(label) > M:
                B.append(('TOO_LONG', label_file, i, len(label))); continue
            # Constraint 7: 无全局重复 (path, label) 对
            # （注：此约束需全局字典去重，存储已见集合 S）
            key = (rel_path, label)
            if key in S:
                B.append(('DUP', label_file, i, key))
            else:
                S.add(key)
    return len(B) == 0, B
```

**处置协议**：`B ≠ ∅` 时，Constraint 1~5 类样本直接删除；6 类样本截断或标记剔除；7 类样本保留 1 条删除其余重复项。

---

## 4.3 数据集划分协议

### 4.3.1 顺序约束（Hard Constraint）

**必须先执行 §4.2 校验 → 再进行划分。** 反序操作的形式化后果：
- 比例失真：设原划分率 α:β:γ，校验移除率为 ρ（train 子集），则实际比例变为 α(1-ρ):β:γ
- 数据泄露：校验出的重复样本若划分后再局部删除，val/test 可能保留与 train 内容相同的样本，导致 val acc 虚高
- 分层失败：未校验的非法标签被划入 val，导致评估时 L_ctc 异常

### 4.3.2 划分算法

**输入**：已通过 §4.2 的样本列表 `S_clean = [(p_1, y_1), …, (p_N, y_N)]`，分层键 `k(y) ∈ K`（如按字符数、首位数字等），划分比例 `(α, β, γ)` = (0.8, 0.1, 0.1)，随机种子 seed。

**输出**：`S_train, S_val, S_test`，满足：
1. `|S_train| / N ≈ α`，`|S_val| / N ≈ β`，`|S_test| / N ≈ γ`
2. 对 ∀κ ∈ K：`#(s ∈ S_* : k(y(s)) = κ) / |S_*| ≈ #(s ∈ S_clean : k(y(s)) = κ) / N`（分层分布一致）
3. `S_train ∩ S_val ∩ S_test = ∅`（无泄露）

```python
def split(S_clean, α, β, γ, seed=42):
    groups = defaultdict(list)
    for s in S_clean:
        groups[k(s['label'])].append(s)
    S_train, S_val, S_test = [], [], []
    for κ, items in groups.items():
        random.Random(seed).shuffle(items)
        n = len(items)
        n_train = int(n * α)
        n_val   = int(n * β)
        S_train.extend(items[:n_train])
        S_val.extend(items[n_train:n_train+n_val])
        S_test.extend(items[n_train+n_val:])
    # 写 train.txt：每行 rel_path \t label；Tab 分隔
    write_tsv('train.txt', S_train)
    write_tsv('val.txt',   S_val)
    write_tsv('test.txt',  S_test)
```

### 4.3.3 泄露检测

额外执行：取 `set(S_train.path)` 与 `set(S_val.path + S_test.path)` 的交集，非空则强制去重。

---

## 4.4 训练启动 SOP

1. **参数装配**：使用 `-o` 逐项覆盖（每对单独 `-o` 前缀）；核心必加项：
   ```
   python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml
     -o Global.mode=train
     -o Global.device=gpu:0
     -o Global.dataset_dir=$DATA_DIR
     -o Global.character_dict_path=$DATA_DIR/dict.txt
     -o Global.max_text_length=8
     -o Global.epoch_num=20
     -o Global.save_inference_dir=$OUT/infer_ep20
     -o Optimizer.lr.learning_rate=0.001
     -o Train.dataset.name=TextRecDataset
     -o Train.sampler.name=BatchSampler
     -o Train.loader.batch_size_per_card=32
     -o Train.loader.num_workers=4
   ```
2. **INFO 8 锚点校验**：执行 §2.1，8 项全部符合再等待。
3. **首轮健康验证**：ep1 完成后，检查 F2 L_ctc < 2.0 ∧ F1 lr 曲线递增。

---

## 4.5 训练终止判据（Early Stopping）

**定义终止条件 `STOP ≡ ∃ e`，对 `∀ e' ∈ [e+1, e+3]`：`val_acc(e') ≤ max_{t < e'} val_acc(t)`（连续 3 epoch 未超历史最优）。**

触发时：
1. 终止训练进程（Ctrl+C）
2. 取 `max val_acc` 对应 epoch 的 `best_accuracy.pdparams` 或 `latest.pdparams` 中最优者
3. 按 §4.6 执行 Inference 导出

---

## 4.6 Inference 模型导出与归档规范

### 4.6.1 自动导出（推荐）

训练命令中追加：
```
-o Global.save_inference_dir=/absolute/path/to/infer_ep$E_best
```

### 4.6.2 手动导出

```
python main.py -c PP-OCRv4_mobile_rec.yaml \
  -o Global.mode=export \
  -o Global.resume=$OUT/best_accuracy.pdparams \
  -o Global.save_inference_dir=$OUT/infer_best
```

### 4.6.3 归档校验清单（三文件 + 一校验）

```
infer_best/
├── inference.pdmodel        # 模型结构（PaddlePaddle IR）
├── inference.pdiparams      # 权重参数
├── inference.pdiparams.info # 权重元数据
└── dict.txt                 # ★ 与训练完全一致的字典文件（md5sum 校验）

验证：
  $ md5sum infer_best/dict.txt $DATASET/dict.txt
  → 两个 md5 必须完全一致
  $ zip -r infer_best_YYYYMMDD.zip infer_best/  → 交付归档
```

---

## 4.7 推理质量评估协议（三桶分箱法）

### 4.7.1 形式化定义

给定预测集 `{(p_i, y_i, ŷ_i, s_i)}`，其中 `s_i` = rec_score，置信度阈值 τ（默认 0.8）：

| 桶 | 条件 | 工程语义 | 优先级 |
|---|---|---|---|
| `incorrect` | `ŷ_i ≠ y_i` | 全量错误样本集 | 基础精度度量 |
| `incorrect_high_conf` | `ŷ_i ≠ y_i ∧ s_i ≥ τ` | **高危险样本**：模型对错误输出给出高置信度，部署时静默引入错误数据 | 最高（排查优先） |
| `correct_low_conf` | `ŷ_i = y_i ∧ s_i < τ` | 不确定性正确样本：决策边界附近，相似样本易误判 | 高（Hard Example Mining 候选） |

### 4.7.2 质量判定矩阵

| 指标 | 健康阈值 | 处置 |
|---|---|---|
| 总体 acc = correct / total | ≥ 0.99 | 通过 |
| incorrect_high_conf / incorrect | ≤ 0.10 | > 0.30 → 首查 A1 字典一致性；否则将该桶样本 100% 纳入训练集重训 |
| correct_low_conf / correct | ≤ 0.05 | > 0.10 → 随机抽样 50% 纳入训练集（Hard Example） |

---

## 4.8 推荐数据集方案与合成数据生成算法

### 4.8.1 开源数据集评估

| 数据集 | 规模 | 字符域 | 适用度（仪表数字识别场景） | 备注 |
|---|---|---|---|---|
| SynthText | 100万+ | 拉丁字符集 | ⭐⭐⭐⭐⭐ | 过滤标签保留 `[0-9.\-]` 子集后使用 |
| IIIT-5K | 5,000 | 拉丁 | ⭐⭐⭐⭐ | 真实街景，标牌类居多，需适配 |
| SVHN | 600,000 | 数字 0-9 | ⭐⭐⭐ | 单数字检测框，需自行拼接连续序列 |
| NIST SD19 | 810,000 | 打印+手写混合 | ⭐⭐⭐ | 打印子集质量可用；手写集与目标域差异大 |
| **自定义合成数据** | 任意 | 精确匹配 {0..9, ., -} | ⭐⭐⭐⭐⭐（首选） | 由 §4.8.2 算法生成，可嵌入真实仪表背景 |

### 4.8.2 合成仪表数字生成算法

**输入**：字体集合 `FONTS`（优先七段数码管/LCD TTF）、真实仪表背景裁剪集 `BG_DIR`、输出数 `N`。

**输出**：`N` 张合成图像 + 对应 `train.txt`。

```python
def synthesize(N, out_dir, FONTS, BG_DIR):
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    lines = []
    for i in range(N):
        L = random.randint(2, 7)
        label = ''.join(random.choices('0123456789', k=L))
        # 小数点插入（20% 概率且 L ≥ 3）
        if random.random() < 0.20 and L >= 3:
            pos = random.randint(1, L-1)
            label = label[:pos] + '.' + label[pos:]
        # 负号前置（5% 概率）
        if random.random() < 0.05:
            label = '-' + label
        # 背景：真实仪表背景 50%，纯色噪声 50%
        if random.random() < 0.5 and BG_DIR and os.listdir(BG_DIR):
            bg = Image.open(os.path.join(BG_DIR, random.choice(os.listdir(BG_DIR)))).convert('RGB')
            bg = bg.resize((320, 48), Image.BILINEAR)
        else:
            g = random.randint(180, 240)
            bg = Image.new('RGB', (320, 48), (g, g, g))
        # 文字渲染
        font = ImageFont.truetype(random.choice(FONTS), random.randint(22, 40))
        draw = ImageDraw.Draw(bg)
        c = random.randint(0, 60)
        draw.text((random.randint(5, 20), random.randint(2, 8)),
                  label, fill=(c, c, c), font=font)
        # 质量退化（可选：高斯模糊、JPEG 压缩伪影、对比度扰动）
        name = f'synth_{i:07d}.jpg'
        bg.save(f'{out_dir}/images/{name}', quality=random.randint(70, 95))
        lines.append(f'images/{name}\t{label}')
    with open(f'{out_dir}/train.txt', 'w') as f:
        f.write('\n'.join(lines))
```

### 4.8.3 数据配比经验模型

| 来源 | 比例 | 作用 |
|---|---|---|
| 合成数据（§4.8.2 算法） | 0.60 | 提供数字形状、小数点/负号规律的充分监督 |
| 真实标注数据（PPOCRLabel） | 0.30 | 域对齐（Domain Adaptation）：降低 Sim-to-Real Gap |
| Hard Example（上轮 incorrect_high_conf ∪ correct_low_conf 抽样） | 0.10 | 针对性修补模型盲区，单次迭代可提升 val acc 0.5%~2% |

---

## 4.9 端到端交付 CheckList

```
Phase 1 · 数据准备
  [ ] 数据采集完成：真实标注 ≥ 500 张 或 合成数据 ≥ 50k
  [ ] 标注格式校验：所有 .txt 均为 Tab 分隔 2 列（§4.2 Constraint 1）
  [ ] §4.2 七项校验全部通过，违规样本已清洗
  [ ] §4.3.2 分层划分完成，泄露检测交集 = ∅
  [ ] dict.txt 行数 = C=12，内容 = {0..9, ., -}，无 BOM / 空行 / Windows 换行
  [ ] 三阶段字典路径一致性确认

Phase 2 · 训练启动
  [ ] §2.1 INFO 8 锚点全部符合预期
  [ ] B7 batch_size 已压测至 max_mem_allocated ∈ [0.60, 0.85]
  [ ] 训练命令含 `-o Global.save_inference_dir` 自动导出
  [ ] ep1 L_ctc < 2.0，warmup 期 lr 单调递增

Phase 3 · 训练监控与终止
  [ ] F2 L_ctc ep3 < 0.5
  [ ] F5 val acc 每轮记录，趋势上升
  [ ] Early Stopping 触发（连续 3 epoch 无新峰值）或达到 Epoch E
  [ ] best_accuracy.pdparams 对应 val acc 已归档记录

Phase 4 · 推理评估
  [ ] Inference 导出：infer_best/ 目录含 3 个 inference.* 文件 + dict.txt（md5 校验一致）
  [ ] 冒烟测试：script_verification/ 21 张样本 acc > 0.95
  [ ] test_batch.py 全量回归：总体 acc ≥ 0.99
  [ ] incorrect_high_conf / incorrect ≤ 0.10；否则该桶样本 100% 并入训练集重训
  [ ] 归档：infer_best_YYYYMMDD.zip + 训练日志 + summary.txt / global_summary.txt

Phase 5 · 上线部署
  [ ] 推理脚本 `create_model()` 显式传入 `rec_char_dict_path`（非默认值）
  [ ] 线上灰度 100 张人工复核：incorrect_high_conf 率 < 1%
  [ ] 全量上线
  [ ] 回归调度：每月 test_batch.py 跑一次，每季度 Hard Example + 真实数据增量重训
```

---

**文档结束**
