# PP-OCRv4 数字识别模型训练指南

> 适用场景：PP-OCRv4_mobile_rec + 10~13 类纯数字（0-9 + `.-`）识别 + PaddleX 训练框架 + 加载官方预训练权重 + 小/中数据集（1k~50k）
> 最后更新：2026-08-20

---

## 目录

- [一、故事版整体理解：我开了一个仪表数字识别培训班](#一故事版整体理解我开了一个仪表数字识别培训班)
- [二、训练启动 INFO 8 锚点核查法（10 秒确认配置生效）](#二训练启动-info-8-锚点核查法10-秒确认配置生效)
- [三、完整参数速查表（合理范围 + 偏离处理）](#三完整参数速查表合理范围--偏离处理)
  - [🅰️ 核心固定参数（Global 块，改 yaml，只配一次）](#️-核心固定参数global-块改-yaml只配一次)
  - [🅱️ 训练超参（Optimizer + Train，常用 `-o` 调）](#️-训练超参optimizer--train常用--o-调)
  - [🅲️ 数据 / 采样器参数（小数据/特殊比例必看）](#️-数据--采样器参数小数据特殊比例必看)
  - [🅳️ Loss / 双 Head 权重](#️-loss--双-head-权重)
  - [🅴️ 评估 / 保存参数](#️-评估--保存参数)
  - [🅵 运行时监控指标（日志只读，诊断上游参数）](#-运行时监控指标日志只读诊断上游参数)
- [四、CTCLoss vs NRTRLoss 数值健康对照表](#四ctcloss-vs-nrtrloss-数值健康对照表)
- [五、学习率（lr）剧本核对表](#五学习率lr剧本核对表)
- [六、训练日志诊断三步流程](#六训练日志诊断三步流程)
- [七、常见报错与快速定位](#七常见报错与快速定位)
- [八、新手最容易踩的 5 个参数坑](#八新手最容易踩的-5-个参数坑)

---

## 一、故事版整体理解：我开了一个仪表数字识别培训班

**角色对照：**
- 你 = 培训班班主任
- 15k 张合成仪表图 = 15000 道练习题 + 标准答案
- 百度预训练识字卡 = 基础很好的识字课本（提前发的）
- **小 C（CTCHead）** = 性子急的学生，逐列扫描、独立判断、快、不看上下文
- **小 N（NRTRHead/GTC）** = 做完形填空的学生，串行自回归，看前面猜后面，慢但懂上下文
- `Global.epoch_num` = 总共多少轮模拟考
- `global_step` = 全班累计做了多少道题
- `lr（学习率）` = 班主任讲解答案的音量（按自动播放曲线调整）
- `CTCLoss / NRTRLoss` = 小 C / 小 N 的错误扣分（0 满分，越低越好）
- `loss` = 全班总扣分（小 C + 小 N，默认 1:1 加权）
- `train acc` = 重做自己做过的题的正确率（开卷考，高分不代表真本事）
- `val acc` = 没见过的新题闭卷考正确率（**真本事，唯一硬指标**）
- `ips` = 每秒做多少道题（吞吐量）
- `max_mem_allocated` = 学生笔记本（显存）实际写了多少页
- `eta` = 距离下课（结课）还有多久

**剧情要点：**
- 第一幕 warmup（前 5 轮）：音量从静音慢慢拧到最大 0.001，怕吓到刚领到新课本的学生
- 第二幕 Cosine 衰减（第 5~20 轮）：音量慢慢变小，让学生自己摸索细节，避免大音量来回冲找不到最优点
- 小 C 学得特别快（第 3 轮 CTCLoss 就能到 0.1 以下），因为它只做逐列独立分类；小 N 慢 3~10 倍是正常的，因为它做完形填空，梯度路径长
- 毕业考试（部署推理）只考"扫题"（CTCLabelDecode），小 N 的完形填空不考——它只是陪练，间接让特征质量更好
- 最终 val acc 连续 3 轮不涨就可以停了，不用硬熬到 20 轮

---

## 二、训练启动 INFO 8 锚点核查法（10 秒确认配置生效）

训练启动后先扫这段打印（`ppocr INFO: XXX : YYY`），**8 项全对了再等训练跑**：

| # | INFO 路径（按缩进拼） | 应该看到的正常值 | 异常说明 |
|---|---|---|---|
| 1 | `Global.character_dict_path` | 指向你自己的 dict.txt 绝对路径，非空 | 空 = 用了默认中文字典，推理必错位 |
| 2 | `Global.use_gpu` + `train with paddle ... device Place(...)` | `True` + `Place(gpu:0)` | False 或 Place(cpu) = 白买显卡，跑在 CPU 上慢 10~50 倍 |
| 3 | `Global.epoch_num` | 你设的 20/30/50 | 和预期不一样 = `-o` 路径写错了（如写成了 epochs_iters 而不是 epoch_num） |
| 4 | `Global.pretrained_model` | 有 URL 或本地路径；后面跟 `load pretrain successful` | 空 = 从 0 训，收敛慢 5~10 倍 |
| 5 | `Train.dataset.label_file_list` + `data_dir` | train.txt 真实存在且路径正确 | 路径错 = 立刻 FileNotFoundError |
| 6 | `Train.dataset.name` | 小数据：`TextRecDataset`（大数据才用 MSTextRecDataset） | 15k 以下还写 MSTextRecDataset → 配合 MultiScaleSampler 容易无限递归爆栈 |
| 7 | `Train.loader.batch_size_per_card` | 16~64（你显卡能塞下的尽可能大值） | = 1 → 慢 + BN 不准，浪费显卡 |
| 8 | `Train.sampler.name` + `scales` | 小数据/特殊比例：`BatchSampler`（scales 行消失） | 还是 MultiScaleSampler + 3 个长条 scales → 特殊比例图被跳过，有效训练样本减少 |

---

## 三、完整参数速查表（合理范围 + 偏离处理）

> 标注方法：A~E 类是 yaml/`-o` 可改的配置；F 类是日志运行指标。每次只改 **1 个参数**，观察 1~2 轮再改下一个。

### 🅰️ 核心固定参数（Global 块，改 yaml，只配一次）

| # | 参数（yaml 完整路径） | 人话含义 | 合理范围（你的场景） | 偏离症状 | 处理方法 |
|---|---|---|---|---|---|
| A1 | `Global.character_dict_path` | 索引→字符的查表字典 | 必须指向训练时的那份 dict.txt（0-9、.-、blank 共 12~13 行），绝对路径 | ① 空/默认 → 推理字符错位，incorrect_high_conf 暴增<br>② 标签有字典外字符 → CTCLoss 永远降不下<br>③ 推理 IndexError 越界 | ① yaml 写死绝对路径<br>② dict.txt 与 train.txt 随机抽 50 条核对<br>③ **导出/部署推理时的 dict 和训练必须字节级完全一致**（换行、BOM、行数） |
| A2 | `Global.device` / `use_gpu` | 用什么硬件训 | `gpu:0` + `use_gpu: True` | ① 用 CPU → ips 只有个位数<br>② CUDA error 启动失败 | ① `-o Global.device=gpu:0`<br>② `nvidia-smi` + `paddle.utils.run_check()` 确认 GPU 版本对 |
| A3 | `Global.model_name` | 模型骨架 | 你的场景：`PP-OCRv4_mobile_rec`（够快够准） | 错选 server 版 → 显存翻倍、慢 2~3 倍，acc 涨 <1% | 你当前 mobile 版足够，别盲上大模型 |
| A4 | `Global.pretrained_model` | 预训练权重 | 保留官方 URL 或本地有效路径 | ① 空 → 第 5 轮 CTCLoss 还 >2<br>② 文件损坏 → weight 加载失败 | ① 不要清，除非 100k+ 真实数据从 0 训<br>② 清缓存 `~/.paddleocr/models/*.pdparams` 重下 |
| A5 | `Global.max_text_length` | 最长允许字符数 | 数字场景：6~10（按最长标签 +2，如 "12345.67"→8） | 默认 25 → NRTRLoss 偏高 20%+（END token 学习慢） | `-o Global.max_text_length=8` |
| A6 | `Global.use_space_char` | 字典是否含空格 | 纯数字：**False**（标签里真有空格才 True） | 数字场景设 True + 字典没空格行 → 类别数不匹配 + 错位 | 看 dict.txt 有没有空格行决定 |

### 🅱️ 训练超参（Optimizer + Train，常用 `-o` 调）

| # | 参数 | 人话含义 | 合理范围 | 偏离症状 | 处理方法 |
|---|---|---|---|---|---|
| B1 | `Global.epoch_num` | 总训练轮数 | 有预训练 + 15k：20~50；无预训练：100~200 | ① <10 → 没收敛就停<br>② 100+ → 浪费时间 + 后期过拟合 | val acc 连续 3 轮不涨就 Ctrl+C 停 |
| B2 | `Optimizer.lr.learning_rate` (base_lr) | 最大学习率（音量上限） | 微调预训练：0.0005~0.002；从 0 训：0.003~0.008 | ① <0.0003 → 第 10 轮 CTCLoss 还 >0.5<br>② >0.005（微调）→ loss NaN 或震荡 | 微调就用默认 0.001，差一个数量级再 ×2/÷2 试 |
| B3 | `Optimizer.lr.name` | lr 曲线形状 | 微调：Cosine；老派小数据：Step | Fixed → 后期震荡不收敛 | 你现在的 Cosine 不用改 |
| B4 | `Optimizer.lr.warmup_epoch` | 前几轮慢慢拧大音量 | 总 epoch 的 1/4 左右：20 轮就 5；50 轮就 10 | ① 0 → 预训练首轮 loss 飙升/NaN<br>② >30% 总 epoch → 白跑，音量不够一直学不动 | 固定比例 1/4 |
| B5 | `Optimizer.name` | 优化器 | **Adam**（默认鲁棒）；大数据：SGD+Momentum | 写错 + 无 momentum → 慢 2 倍 | 别动默认 Adam |
| B6 | `Optimizer.regularizer.factor` (L2) | 防过拟合罚款 | 3e-05 ~ 1e-04 | ① >5e-4 → CTCLoss 高一个量级<br>② 0 → 20 轮后过拟合（train 99.9% / val 80%） | 默认 3e-05 就好 |
| B7 | `Train.loader.batch_size_per_card` | 每批塞多少张图 | **16~64**（目标：显存 70~80%） | ① 1~8 → 慢 4~8 倍 + BN 不准<br>② 128+ → CUDA OOM 显存溢出 | 你现在只用 1.3G/8G，直接从 32 开始试，OOM 就 ÷2 |
| B8 | `Train.loader.num_workers` | CPU 读图线程数 | 4~8（CPU 核数一半） | ① 0 → reader_cost 占 >30% GPU 等喂<br>② >16 → CPU 切换开销大 + 递归报错（filelock） | 从 4 加到 reader_cost 占 <10% 停 |

### 🅲️ 数据 / 采样器参数（小数据/特殊比例必看）

| # | 参数 | 人话含义 | 合理范围 | 偏离症状 | 处理方法 |
|---|---|---|---|---|---|
| C1 | `Train.dataset.name` | 数据集类 | 10k~100k + 特殊比例：**TextRecDataset**；100k+ 均匀比例：MSTextRecDataset | 小数据 MSTextRecDataset + C2 MultiScaleSampler → 无限递归爆栈 | `-o Train.dataset.name=TextRecDataset`（和 C2 配套） |
| C2 | `Train.sampler.name` | 采样器 | 小数据/特殊比例：**BatchSampler**；大数据均匀：MultiScaleSampler | 小数据 MultiScaleSampler → 找不到匹配 scale 的图 → 递归 900+ 次爆栈 | `-o Train.sampler.name=BatchSampler`（和 C1 配套） |
| C3 | `Train.sampler.scales` | 多尺度宽高比列表 | 仪表数字 2~4 位：`[[160,48],[200,64]]`（3~4:1）；关了 MultiScaleSampler 就忽略 | scales 全是 10:1 长条 → 实际有效样本被跳过一大半 | 先关 sampler 最省心；保留就先扫数据真实分布再填 |
| C4 | `RecAug` + `RecConAug.prob` | 数据增强开关与概率 | RecAug 数字可 None；RecConAug.prob 0.3~0.5 | ① 1.0 → 前 3 轮 CTCLoss 不降<br>② 0 → 过拟合，val acc 低 3~5% | 默认值 OK；val acc 差就调 prob 从 0.5 ↔ 0.3 |

### 🅳️ Loss / 双 Head 权重

| # | 参数 | 人话含义 | 合理范围 | 偏离症状 | 处理方法 |
|---|---|---|---|---|---|
| D1 | `Loss.loss_config_list.CTCLoss.weight` | 小 C 扣分权重 | **1.0**（推理用 CTC，必须保证） | <0.5 → CTC 学变慢，推理质量下降 | 固定 1.0，别降 |
| D2 | `Loss.loss_config_list.NRTRLoss.weight` | 小 N 扣分权重 | 默认 1.0；赶时间 0.2；极致加速 0（关） | ① 2.0+ → 总 loss 被 NRTR 主导，CTC 反而变慢<br>② 0.0 → 训练快 20~30%，val acc 小概率降 0.3~1% | ① 要效果 1.0<br>② val acc 已经够高 → 0.2 或直接关 NRTRHead |

### 🅴️ 评估 / 保存参数

| # | 参数 | 人话含义 | 合理范围 | 偏离症状 | 处理方法 |
|---|---|---|---|---|---|
| E1 | `Global.eval_batch_epoch` | 多少轮跑一次验证集 | **1**（每轮必评估） | 设成 5 → 错过 4 轮 val acc 峰值，不知道何时过拟合 | 别改，val acc 是判断"能否停"的硬指标 |
| E2 | `Global.save_epoch_step` | 多少轮存一次 checkpoint | **1** | 设成 5 → 中途崩了回退 5 轮 | 1 轮存一次不耗时 |
| E3 | `Global.save_inference_dir` | 是否训练完直接导出 inference 模型 | 直接写绝对路径，如 `/home/msun/infer_ep20` | None → 训完还得手动 export，多一步 | `-o Global.save_inference_dir=/path/to/infer`，省一道工序 |
| E4 | `Global.cal_metric_during_train` | 训练中是否算 train acc | **True** | False → 失去过拟合判断信号（train acc 和 val acc 的差） | 别关 |

### 🅵 运行时监控指标（日志只读，诊断上游参数）

| # | 日志 key | 合理范围（你的场景） | 偏离时 → 去调哪个上游参数 |
|---|---|---|---|
| F1 | **lr** | warmup 线性 0→0.001（5 轮）；之后 Cosine 降到 ~0（20 轮） | 走势不对 → B2/B3/B4（base_lr / curve / warmup） |
| F2 | **CTCLoss** | ep1<3 → ep3<0.5 → ep10<0.05 → ep20<0.01 | ① ep3 还>1.0 → A1 字典错 或 B2 lr 太小<br>② 一直不降 → A4 预训练 或 C4 增强太猛 或 A1 字符错位 |
| F3 | **NRTRLoss** | ep1<3 → ep5<1.0 → ep15<0.4 → ep20<0.2 | ① 5 轮还>1.5 → D2 调 0.2（它慢不影响推理）<br>② 一直>3 → A5 max_text_length 太长了改 8 |
| F4 | train acc | 只看和 **val acc 的差**：差 <5% 健康；差>15% 过拟合 | 差太大 → C4 加增强概率 / B7 增 bs / B6 加 L2 / 加数据 |
| F5 | **⭐ val acc（eval 行）** | ep5≥95% → ep10≥98% → ep20≥99% | ① <90% → **头号查 A1 字典！** 再查 C4 无增强 / 标签错太多<br>② 连续 3 轮不涨 → 直接停止训练 |
| F6 | norm_edit_dis | 和 acc 同趋势；val NED ≥ 0.99 才叫真的好 | 比 acc 低 5%+ → 大多是单字符混淆（0/8、1/7），针对性补样本 |
| F7 | reader_cost / batch_cost | **<10%** → IO 健康，GPU 不挨饿 | >30% → B8 num_workers 从 4→8，数据挪 SSD |
| F8 | ips samples/s | mobile 单卡：80~300，随 bs 涨 | <30 太慢 → 查 F7 IO / A2 跑在 CPU / B7 bs 太小 |
| F9 | max_mem_allocated / 显存总量 | **60%~85%**（用满不爆） | ① <40% → B7 bs ×2 加到 80%<br>② >95% 且 OOM → B7 bs ÷2 |
| F10 | eta（倒计时） | 仅参考 | 突然暴增 → 查 F7/F8（IO 堵 / GPU 温度降频 / 其他进程抢卡） |

---

## 四、CTCLoss vs NRTRLoss 数值健康对照表

### CTCLoss（小 C 成绩单，核心指标，越低越好）

> 口径：`CTCLoss = -ln(所有合法路径的总概率)`。用 e^-loss 能换算成"模型认为走对路径的把握程度"。

| 阶段 | CTCLoss 范围（你的场景） | 把握程度 e^-loss | 模型状态 | 异常处理 |
|---|---|---|---|---|
| ep0 开始 | 3.0 ~ 5.0 | 0.7% ~ 50% | head 随机初始化，正常 | 无需处理 |
| ep1 结束 | 0.5 ~ 2.0 | 13.5% ~ 60.7% | 预训练起作用了，快速下降 | ✅ |
| ep2 ~ ep3 | 0.05 ~ 0.5 | 60.7% ~ 95.1% | **你当前 ep3=0.071，把握≈93%，非常好 ✅** | ✅ |
| ep5 ~ ep10 | 0.001 ~ 0.05 | 95.1% ~ 99.9% | CTC 头基本学好，进入平台期 | 不用再盯 CTC，改盯 val acc |
| ep10+ | <0.001（不动了） | ≥99.9% | CTC 饱和，再训也没收益 | 可以考虑停了 |

### NRTRLoss（小 N 成绩单，不与 CTC 比绝对值！只看自降趋势）

> 口径：**每一步条件交叉熵的平均值**（自回归），定义和 CTC 完全不同，不能直接比大小！

| 阶段 | NRTRLoss 范围（你的场景） | 含义 | 异常处理 |
|---|---|---|---|
| ep1 开始 | 2.5 ~ 4.0 | 瞎蒙（1/12 理论下界 2.5） | ✅ |
| ep1 结束 | 1.5 ~ 3.0 | 学会了"第一个字符大概率是数字" | ✅ |
| ep2 ~ ep5 | 0.6 ~ 1.8 | 学会了短上下文（如 3→7/.）｜**你 ep3=1.46 正常 ✅** | ✅ |
| ep6 ~ ep15 | 0.15 ~ 0.6 | 记住了小数点/负号位置规律 | 还>1.5 → D2 weight 调 0.2 |
| ep15+ | 0.05 ~ 0.15（饱和了） | 数字场景上下文信息太少，降不到 CTC 的 0.07 很正常 | 不用再等它降了 |

---

## 五、学习率（lr）剧本核对表

你的配置：`name=Cosine` + `base_lr=0.001` + `warmup_epoch=5` + `epoch=20`

```
两幕剧：
  warmup（ep0~5）：linear 0 → 0.001
  cosine（ep5~20）：0.001 → ~0（余弦曲线下降）
```

**你以后每轮可以对着核对（验证剧本对不对）：**

| epoch | 预期 lr 大约值 | 你已跑过的对照 | 验证结果 |
|---|---|---|---|
| ep1 中途 | ≈ 0.0002 | | |
| ep2 结束 | ≈ 0.0004 | | |
| **ep3 中途** | **≈ 0.00055** | **你实际 0.000522** | ✅ 吻合（bs 浮动导致差 6% 内算正常） |
| ep4 结束 | ≈ 0.0008 | | |
| ep5 结束 | = 0.00100（峰值） | | 下一个验证点 |
| ep10 中间 | ≈ 0.00085 | | |
| ep15 中间 | ≈ 0.00045 | | |
| ep18 末尾 | ≈ 0.00008 | | |
| ep20 最后 | ≈ 0.00001（接近静音） | | |

---

## 六、训练日志诊断三步流程

**每次看到刷新的一行日志，按这个顺序扫，不用全看：**

1. **看 CTCLoss 降没降？**  
   → 降了 → ✅，小 C 在继续学；不降 → 查 A1/A4/C4/B2

2. **看 lr 曲线走势对不对？**  
   → warmup 期涨 / cosine 期降 → ✅；长期不变 → B3/B4 配错了

3. **每轮结束**只看 `eval epoch: X, acc: X.XXXX` 那一行（val acc）：  
   → 涨了 → 继续跑  
   → 连续 3 轮不涨 → **直接 Ctrl+C 停，不要浪费时间硬跑完**

> 其他 20 个参数都是辅助，盯死这 3 处，训练就不会跑偏。

---

## 七、常见报错与快速定位

| 报错关键信息 | 根因 90% 概率 | 1 分钟处理方法 |
|---|---|---|
| `simple_dataset.py: __getitem__ repeated 900+ times` + `RecursionError` | MultiScaleSampler 找不到匹配 scale 的图，无限递归重选（小数据/特殊比例常见） | 关多尺度：`-o Train.dataset.name=TextRecDataset -o Train.sampler.name=BatchSampler` |
| `copy.deepcopy` → `filelock/_audit_fork_safety` + `RecursionError` | filelock ≥3.14 版本和 deepcopy 冲突（采样器递归时会放大） | `pip install filelock==3.13.1` 降级 |
| `IndexError: list index out of range`（出现在 rec_postprocess.decode） | 推理时字典类别数 < 模型输出类别索引（模型训练用 13 类字典，推理拿了 12 类或默认 6625 类字典） | 训练/推理**同一份 dict.txt**，路径写死、确认行数 |
| `CUDA out of memory`（`RuntimeError`） | batch_size 开太大，显存装不下 | B7 `batch_size_per_card` ÷2 重试，直到 max_mem_allocated < 85% |
| CTCLoss 连续 5 轮都卡在 2~3 不降 | 头号嫌疑：A1 字典和标签字符不匹配（标签里有字典外字符）；次号：C4 增强概率 1.0 太猛 | ① 随机抽 100 条 train.txt 对照 dict.txt<br>② 把 C4 RecConAug.prob 从 1.0 调到 0.3 |
| train acc 99.9998% 但 val acc <80% | 典型过拟合：训练集太小 / 没开增强 / 标签有重复 | C4 开增强、加真实样本、B6 L2 调 1e-4、增加数据量 |
| 启动 `unrecognized arguments: Global.xxx=xxx` | `-o` 写法不对，多个 KV 对跟在一个 `-o` 后面而不是每个前面加 `-o` | 改成：`-o A=B -o C=D -o E=F` 每一项单独加 `-o` |
| `shape mismatch WARNING: head.ctc_head.fc.weight [120, 12] vs [120, 6625]` | **正常！不是错误！** 自定义 12 类字典和官方预训练 6625 类 head 维度不同，PaddleOCR 自动跳过不匹配的 head 加载 backbone | 不用管，只需要看到后面跟一句 `load pretrain successful` 就 OK |
| `PaddleX script: error: unrecognized arguments: ...` | 写了 `Global.device=gpu:0` 但前面没加 `-o`，shell 当成独立参数解析了 | 每一个 `XXX=YYY` 前必须单独写 `-o` |
| loss 一上来就是 `nan` 或 `inf` | ① lr 太大（微调场景写了 0.01）② 标签里有空串 / 非法字符 | ① B2 lr ÷10 从 0.0001 试起<br>② 跑数据校验脚本扫 train.txt 空标签/超长标签 |

---

## 八、新手最容易踩的 5 个参数坑（翻车概率从高到低）

### 🥇 坑 1：A1 `character_dict_path` 忘记传 / 训练推理字典不一致
**后果：** 推理时字符全错位，val acc 看起来没问题，上线 incorrect_high_conf 暴增，几天白训。  
**避免：** ① 训练前 yaml 写死 dict 绝对路径；② 导出 inference 模型时，把 dict.txt 一并拷到模型目录；③ 推理脚本 `create_model()` 一定要显式传 `rec_char_dict_path`，不要走默认。

### 🥈 坑 2：B7 batch_size 硬写 1
**后果：** 慢 4~10 倍 + BN 层统计不准，val acc 波动大还白浪费显存。  
**避免：** 第一次就从 32 开始试，只在 OOM 时才往下调。目标是显存用到 70~80%。

### 🥉 坑 3：C1+C2 小数据还开 MSTextRecDataset + MultiScaleSampler
**后果：** 就是你上次遇到的 RecursionError 965 次无限递归爆栈。  
**避免：** 100k 以下 + 特殊比例（仪表数字 2~4:1）直接关，用 TextRecDataset + BatchSampler。

### 4 坑 4：warmup_epoch 比例配错（20 轮写 warmup=15 等）
**后果：** 前 15 轮音量一直拧不上来，等于白跑 3/4 的时间，最后 5 轮还没收敛就结束了。  
**避免：** 固定 warmup = 总 epoch 的 1/4，20→5，50→10，100→25。

### 5 坑 5：只看 train acc 不看 val acc
**后果：** 训到 train acc 99.9998% 兴奋到飞起，一上真实图才 70% 准确率。  
**避免：** 每轮 eval 完只把 `eval acc` 那一行截图存进度条。val acc 连续 3 轮不涨就是该停的信号，别硬熬到总 epoch 数。

---

## 附：你当前第 3 轮训练状态的健康度总结（日志基准）

| 指标 | 实际值 | 评估 |
|---|---|---|
| CTCLoss | 0.071119 | ✅ 优秀，CTC 头已学好 |
| NRTRLoss | 1.457928 | ✅ 正常，比中文场景 ep3 还快一点，不用和 CTC 比 |
| lr | 0.000522 | ✅ 和预期 0.00055 吻合，warmup 曲线正确 |
| train acc | 0.999998 | ⚠️ 太高，下一轮务必盯 val acc 是否也 >98%，防过拟合 |
| reader/batch 占比 | 1.41ms / 53ms ≈ 2.6% | ✅ IO 非常健康 |
| ips | 112.5 samples/s | 正常 |
| max_mem_allocated | 1.3G（8G 显卡） | 💡 建议 bs 从平均 6 开到 32~64，提速 3~5 倍 |
| eta | 42:27 | 正常（bs 开大后会降到 10 分钟内） |

**下一步建议：**
1. 下一轮看 ep5 结束时 lr 是否精确到 0.001 附近 ✔️
2. 下一轮 eval 结束贴 val acc 数值，判断过拟合情况和是否需要早停
3. 可以 `-o Train.loader.batch_size_per_card=32` 试一次，看显存是否爆

---

## 九、项目结构与数据流向（从零理解这个仓库）

### 9.1 目录树总览

```
paddlex_ppocr_rec_predict/
├── demo/                              # ✅ 单图推理最小示例（3 行代码跑通）
│   ├── demo.py                        #    create_model → predict → save 三件套
│   ├── general_ocr_rec_001.png        #    示例输入图片
│   └── output/                        #    示例输出（标注图 + json）
│       ├── general_ocr_rec_001_res.png
│       └── res.json
│
├── test/                              # ✅ 批量测试脚本（核心模块）
│   ├── test_single_ppocrv5.py         #    单数据集批量测（命令行参数驱动）
│   ├── test_batch.py                  #    多数据集批量测 + 分类归档（配置文件驱动）
│   ├── test_batch_config_example.json #    test_batch.py 的配置模板
│   └── result/no_1/                   #    历史测试结果归档
│       ├── test_set_1/ ~ test_set_10/ #      每个测试集独立目录
│       │   ├── incorrect/             #        ❌ 所有识别错误的图 + info.txt
│       │   ├── incorrect_high_conf/   #        ⚠️  认错了还很自信（>0.8）→ 字典/样本问题
│       │   ├── correct_low_conf/      #        ⚠️  认对了但不自信（<0.8）→ 补相似样本
│       │   ├── summary.txt            #        该测试集统计汇总
│       │   └── global_summary.txt     #    全部测试集总体汇总
│
├── script_verification/               # ✅ 脚本冒烟测试用（改完代码先跑这个）
│   ├── test_img/                      #    21 张真实仪表裁剪图（8~46 范围的 1~2 位整数）
│   └── test.txt                       #    标注文件：相对路径\t标签（Tab 分隔）
│
└── 数字OCR模型训练指南_PP-OCRv4.md   # 📖 本文档
```

### 9.2 三大核心脚本对比

| 脚本 | 适用场景 | 输入方式 | 输出 |
|---|---|---|---|
| `demo/demo.py` | 学习 PaddleX API / 快速验证环境是否装好 | 单张图片路径 | 终端 print + 保存标注图和 JSON |
| `test/test_single_ppocrv5.py` | 单个数据集跑完看准确率 | 命令行：`--model_dir --input_dataset_dir --output_result_dir` | 3 类文件夹 + summary.txt |
| `test/test_batch.py` | 多个模型 / 多个数据集横向对比，回归测试 | JSON 配置文件：`model{model_name, model_dir}` + `test_sets[]` | 每个测试集独立目录 + global_summary.txt |

### 9.3 推理数据流向（以 test_batch.py 为例）

```
 step 1  读取 test.txt（Tab 分隔）
          每行: "相对图片路径\t真实标签"
                        │
                        ▼
 step 2  os.path.join() → 拼成图片绝对路径列表 []
          真实标签 → 对应字符串列表 []
                        │
                        ▼
 step 3  model.predict(input=图片路径列表, batch_size=20)
          PaddleX 内部: resize → backbone → CTCHead → CTCLabelDecode → (rec_text, rec_score)
                        │
                        ▼
 step 4  逐条比较: rec_text == truth_label ?
          ├─ 正确 ✅ + rec_score < 0.8  → correct_low_conf/   (认对了但不自信)
          └─ 错误 ❌ ┬─ rec_score >= 0.8 → incorrect_high_conf/  (认错了还很自信 → 严重)
                     └─ rec_score <  0.8 → incorrect/           (认错了但知道自己不确定)
                        │
                        ▼
 step 5  保存: 标注图(jpg) + info.txt(文件名/真值/预测值/置信度,固定宽度对齐)
          汇总: summary.txt (总数/准确率/三类计数) + global_summary.txt(跨测试集)
```

### 9.4 三个输出目录的诊断意义（为什么这样分桶）

| 目录 | 数学定义 | 出现多了说明什么 | 整改方向 |
|---|---|---|---|
| `incorrect/` | 预测≠真值 的全部样本 | 基础准确率不够 | 加数据 / 调参 / 训更久 |
| `incorrect_high_conf/` | 预测≠真值 **且** 置信度≥阈值（默认 0.8） | **头号危险信号**：模型以为自己对了，实际错了，上线会静默生产错数据 | ① 90% 是字典不一致（查 A1）<br>② 训练集中这类样本缺失，把这里的图复制进训练集重训 |
| `correct_low_conf/` | 预测=真值 **但** 置信度<阈值 | 模型不确定但蒙对了，说明相似干扰样本多 | 把这些图加进训练集做 Hard Example Mining |

> 💡 经验：`incorrect_high_conf/` 比 `incorrect/` 重要 10 倍。**`incorrect_high_conf` 非空，先别动训练参数，先查字典。**

---

## 十、字符字典（dict.txt）深度原理：没有它真的判不了别

### 10.1 一句话回答你：没有字典为什么不行？

> **模型输出的是数字索引（0、1、2...），不是字符。字典就是"索引→字符"的翻译表。没字典，你只知道第 5 类概率最大，但不知道第 5 类对应数字几。**

### 10.2 一张图看明白字典在整条链路中的位置

```
  输入图片 "37"
      │
      ▼
  Backbone（PPLCNetV3）提取视觉特征
      │
      ▼
  CTCHead 全连接层输出 shape: [时间步T, 类别数C]
      │   每个时间步是一个"概率分布"，加和=1
      │   例如某一步: [0.01, 0.02, 0.90, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00]
      │                                     ↑
      │                                第 3 类概率最大 = 0.90
      ▼
  CTCLabelDecode 去重 + 去 blank
      │   输出一串索引: [3, 8]
      │
      ▼
  ┌──────────────── 查字典 dict.txt ────────────────┐
  │  行号(索引)  字符                                │
  │  0            0                                  │
  │  1            1                                  │
  │  2            2                                  │
  │  3            3   ← 索引 3 → 字符 "3"           │
  │  4            4                                  │
  │  5            5                                  │
  │  6            6                                  │
  │  7            7                                  │
  │  8            8   ← 索引 8 → 字符 "8"   ❌ 错了！│
  │  9            9                                  │
  │  10           .                                  │
  │  11           -                                  │
  │  (blank = 12，CTC 内部使用，不出现在 dict.txt)    │
  └──────────────────────────────────────────────────┘
      │
      ▼
  输出 "38"（真值是 "37" → ❌ 识别错误）
```

**这就是为什么你第 7 行标签是 "37"，如果字典里索引 7 和 8 写反了，CTCLoss 看起来很低但识别全错。**

### 10.3 纯数字场景的标准 dict.txt 写法

```
0
1
2
3
4
5
6
7
8
9
.
-
```

共 **12 行**（不含 blank，blank 由 PaddleOCR 自动追加为最后一类）。  
验证：模型 head.fc.weight 第二维应该是 **12**（你日志里 shape `[120, 12]` ✅ 吻合）。

### 10.4 三类字典错误的后果速查

| 错误类型 | 具体表现 | 训练时 | 推理时 |
|---|---|---|---|
| **字典行数少了**（如少写了 `.`） | dict 11 行，标签有 `.` | CTCLoss 卡在 2~3 不降，报 "label contains unknown char" | - |
| **字典顺序错了**（7 和 8 反了） | 行数对但行内容错位 | CTCLoss 正常下降，train acc 和 val acc **都很高** ✅ 骗人！ | incorrect_high_conf 暴增，具体表现为 7↔8、0↔8 等互换 |
| **训练推理字典不一致**（训练 12 行推理用默认 6625 行） | `rec_char_dict_path` 推理时没传 | - | IndexError 越界，或识别出中文字符 |

> 🚨 **最坑的是第 2 类**——所有指标都告诉你训练成功了，一上线全错。排查办法：**拿 10 张你肉眼能认的图跑推理，对比输出和 test.txt 的真值，如果不是随机错而是"固定互换某几个数字"，100% 字典顺序错了。**

### 10.5 验证字典正确的 30 秒脚本

```bash
# 1. 看行数（应该 12）
wc -l dict.txt

# 2. 看内容（顺序 0~9 然后 . -，没有空行没有 BOM）
cat -A dict.txt
# 正常输出每行末尾只有 $，没有 ^@（空字符）或 ^M（Windows 换行）

# 3. 和 train.txt 随机对 10 条字符是否都在字典里
cat train.txt | awk -F'\t' '{print $2}' | grep -v '^[0-9.\-]*$'
# 无输出 = 所有标签字符都合法
```

---

## 十一、数据校验 → 数据集划分：为什么顺序不能反？

### 11.1 什么是数据校验（数据体检）？

> **训练前对 train.txt / val.txt / test.txt 做的"全身体检"，确保每条数据合法、可用、格式正确。**

### 11.2 数据校验的 7 项必查清单

| # | 检查项 | 怎么做 | 不合格后果 |
|---|---|---|---|
| 1 | **图片文件存在** | `os.path.isfile(绝对路径)` 扫一遍 | 训练中 FileNotFoundError 直接崩 |
| 2 | **图片能正常解码** | `cv2.imread() is not None` 或 PIL 打开 | 训练中 DecodeImage 抛异常，整批丢掉 |
| 3 | **标签非空** | `len(label.strip()) > 0` | CTCLoss 除以 0 → NaN |
| 4 | **标签字符全部在字典内** | 每个字符 `in dict_set` | 训练时 CTCLoss 死不降；推理静默错 |
| 5 | **标签长度 ≤ max_text_length** | `len(label) <= A5 参数` | NRTRHead 截断后面字符，学了一半 |
| 6 | **无重复标注行** | `sort train.txt \| uniq -d` | 过拟合信号（train acc 虚高） |
| 7 | **标注格式正确（Tab 分隔，2 列）** | `awk -F'\t' '{if(NF!=2) print NR": "$0}'` | 解析为 1 列 → 标签为 None → 崩 |

### 11.3 为什么"先校验，再划分"，反过来不行？

**反例灾难：先划分 8:1:1，再校验发现 train 有 5% 坏图 → 删除 → train 实际比例变成 7.6:1:1，而且 val/test 的坏图没删。**

标准顺序：
```
 step 1  合并所有数据+标注（一个大列表）
            ↓
 step 2  对大列表跑上面 7 项校验，不合格的删
            ↓  得到干净列表（比如原来 15200 → 删完 15000）
 step 3  **再** 对干净列表做 shuffle + 切分 8:1:1
            │   train = 12000
            │   val   = 1500
            │   test  = 1500
            ▼
 step 4  额外校验：train/val/test 之间是否有重复图片（数据泄露）
            ↓
 step 5  保存 train.txt / val.txt / test.txt，路径是绝对路径或相对 data_dir
```

**先校验再划分的 3 个理由：**
1. **比例准确**：你要 8:1:1，最后得到的真就是 8:1:1，不是 7.6:1.03:0.9
2. **避免数据污染**：校验时发现的重复图，如果划分完再删，可能 val 里那张删了但 train 那张还留着，变成"考的题就是练习题"
3. **标签分布一致**：切分前能做分层抽样（如确保 train/val/test 里 1 位/2 位/3 位数字的比例都一样），避免 val 全是难样本导致 val acc 假低

---

## 十二、推荐：数字识别开源数据集

> 你的场景是"仪表盘上的 1~6 位数字，可能带小数点/负号"。纯手写数字（MNIST）完全不适用，需要的是**打印体/七段数码管/LCD 屏幕数字**。

### 12.1 数据集对比表

| 数据集 | 规模 | 类型 | 适用度 | 获取方式 | 备注 |
|---|---|---|---|---|---|
| **SynthText (数字子集)** | 100万+ | 合成打印体 | ⭐⭐⭐⭐⭐ | 官网 + 过滤只保留 `[0-9.\-]` 标签 | 质量最高，字体/背景/光照全，用脚本抽 5 万条够用 |
| **IIIT-5K (数字子集)** | 5000 | 真实街景打印体 | ⭐⭐⭐⭐ | 学术官网 | 真实场景，但很多是户外大标牌，不是仪表 |
| **SVHN (Street View House Numbers)** | 60万 | 门牌号数字 | ⭐⭐⭐ | 官网 / torchvision 自带 | 单数字检测+识别，需要自己拼连续数字，字体单一 |
| **NIST SD19** | 81万 | 打印体+手写 | ⭐⭐⭐ | NIST 官网 | 老数据集，扫描件质量不一，打印体部分可用 |
| **自制合成数据（强烈推荐）** | 任意 | 你需要的仪表字体+背景 | ⭐⭐⭐⭐⭐ | Python PIL + 仪表截图背景，100行代码 | **和真实域最接近**，建议 70% 合成 + 30% 真实 |

### 12.2 5 分钟生成 5 万张合成仪表数字（推荐直接用）

```python
"""
synth_digits.py - 生成合成仪表数字训练集
用法: python synth_digits.py --out_dir /home/msun/dataset/synth_50k --num 50000
"""
import argparse, os, random
from PIL import Image, ImageDraw, ImageFont

random.seed(42)
FONT_PATHS = [  # 放 5~10 个系统里的 ttf，优先 LCD/七段数码管字体
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    # 自己加: simhei.ttf, DS-Digital.ttf (七段管字体, 必加), etc.
]
BG_DIR = "/path/to/your/real_meter_background_crops"  # 10~20 张真实仪表背景裁剪图

def gen_one():
    # 1. 生成 2~7 位随机数字标签，20% 概率小数点，5% 概率负号
    n = random.randint(2, 7)
    label = ''.join(random.choices('0123456789', k=n))
    if random.random() < 0.20 and n >= 3:          # 插小数点
        pos = random.randint(1, n-1)
        label = label[:pos] + '.' + label[pos:]
    if random.random() < 0.05:                      # 加负号
        label = '-' + label
    # 2. 背景: 真实仪表背景 / 纯色噪声各半
    if random.random() < 0.5 and os.path.isdir(BG_DIR):
        bg = Image.open(os.path.join(BG_DIR, random.choice(os.listdir(BG_DIR)))).convert('RGB')
    else:
        gray = random.randint(180, 240)
        bg = Image.new('RGB', (320, 48), (gray, gray, gray))
    # 3. 写字
    font = ImageFont.truetype(random.choice(FONT_PATHS), random.randint(22, 40))
    draw = ImageDraw.Draw(bg)
    color = random.randint(0, 60)
    draw.text((random.randint(5, 20), random.randint(2, 8)),
              label, fill=(color, color, color), font=font)
    # 4. 轻度退化: 模糊 / 压缩噪声 / 对比度 (可选, 自己加 PIL.ImageFilter)
    return bg, label

if __name__ == '__main__':
    args = argparse.ArgumentParser().parse_args()
    os.makedirs(args.out_dir + '/images', exist_ok=True)
    with open(args.out_dir + '/train.txt', 'w') as f:
        for i in range(args.num):
            img, label = gen_one()
            name = f'synth_{i:07d}.jpg'
            img.save(f'{args.out_dir}/images/{name}', quality=random.randint(70, 95))
            f.write(f'images/{name}\t{label}\n')
    print('Done. Synth dataset at', args.out_dir)
```

### 12.3 数据配比建议（经验公式）

| 数据来源 | 占比 | 作用 |
|---|---|---|
| 合成数据（上面脚本） | 60% | 保证数字形状 + 小数点/负号学习 |
| 真实数据（PPOCRLabel 标注） | 30% | 对齐真实域，减少 Domain Gap |
| Hard Example（从 incorrect_high_conf / correct_low_conf 捞） | 10% | 专门修补模型的"盲区" |

> **每跑一轮新训练，把上一轮 incorrect_high_conf/ 的图复制进训练集，这叫 Hard Example Mining，通常提 acc 最快。**

---

## 十三、推理流程速查：单图 / 单数据集 / 多数据集

### 13.1 单图推理（demo/demo.py，验证环境）

```python
from paddlex import create_model

# ① 官方模型快速试（不推荐用于你自己的数字识别）
model = create_model(model_name="PP-OCRv5_server_rec")

# ② 你自己训的模型（必须同时传 model_dir 和字典！）
model = create_model(
    model_name="PP-OCRv4_mobile_rec",
    model_dir="/home/msun/infer_ep20",                       # inference 导出目录
    rec_char_dict_path="/home/msun/dataset/dataset_02/dict.txt",  # ⚠️ 和训练时完全同一份！
)

output = model.predict(input="/home/msun/dataset/aha.png", batch_size=1)
for res in output:
    res.print()                                    # 终端看 rec_text + rec_score
    res.save_to_img(save_path="./output/")          # 保存标注后的图片
    res.save_to_json(save_path="./output/res.json") # 保存结构化结果
```

### 13.2 单数据集批量测试（test_single_ppocrv5.py）

```bash
cd test/
python test_single_ppocrv5.py \
    --model_dir /home/msun/infer_ep20 \
    --input_dataset_dir /home/msun/dataset/dataset_02/test \
    --output_result_dir ./output/test_run_1
```

**要求：** `input_dataset_dir` 下必须有 `test.txt`，格式：
```
images/meter_0001.png	12.34
images/meter_0002.png	-5.6
...
```

### 13.3 多数据集批量回归测试（test_batch.py）

第一步：复制 `test_batch_config_example.json` 改路径
```json
{
  "model": {
    "model_name": "PP-OCRv4_mobile_rec",
    "model_dir": "/home/msun/infer_ep20"
  },
  "output_result_dir": "./output/batch_20260821",
  "test_sets": [
    {"input_dataset_dir": "/home/msun/dataset/pc_screen/test"},
    {"input_dataset_dir": "/home/msun/dataset/power_supply/test"},
    {"input_dataset_dir": "/home/msun/dataset/temperature_lcd/test"}
  ]
}
```

第二步：运行
```bash
cd test/
python test_batch.py --config my_config.json --conf_threshold 0.8 --batch_size 32
```

### 13.4 导出 inference 模型（训练完一步到位）

训练时加 `-o Global.save_inference_dir` 可以直接导出，训完不用手跑：
```bash
python main.py -c PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=train \
    -o Global.device=gpu:0 \
    -o Global.save_inference_dir=/home/msun/infer_ep20
```

如果已经训完想手动导出：
```bash
python main.py -c PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=export \
    -o Global.resume=/home/msun/PaddleX/output/best_accuracy.pdparams \
    -o Global.save_inference_dir=/home/msun/infer_ep20
```

> 💡 **导出完成后三件事：**
> 1. 把 dict.txt 拷到 infer_ep20/ 目录下（防止以后找不到）
> 2. 跑 script_verification/ 那 21 张图确认没回归
> 3. 把整个 infer_ep20/ 打 zip 归档备份

---

## 十四、端到端完整操作流程 CheckList（从 0 到上线按顺序打勾）

```
☐ 1. 数据准备
   ☐ 1.1 收集真实仪表图 500+ 张或运行 synth_digits.py 造 5 万合成图
   ☐ 1.2 PPOCRLabel 标注（或已有标注），一个总列表 all.txt
   ☐ 1.3 数据校验：运行 7 项检查（11.2 节），删掉不合格
   ☐ 1.4 切分 train.txt : val.txt : test.txt = 80% : 10% : 10%，shuffle，查泄露
   ☐ 1.5 写 dict.txt（0~9 + . + - 共 12 行，无空行无 BOM），验证 wc -l = 12

☐ 2. 训练启动
   ☐ 2.1 写 yaml 或用 -o 覆盖 A1~A6 固定参数，character_dict_path 写死绝对路径
   ☐ 2.2 -o Train.dataset.name=TextRecDataset -o Train.sampler.name=BatchSampler
   ☐ 2.3 -o Train.loader.batch_size_per_card=32（爆显存再降到 16/8）
   ☐ 2.4 运行训练命令，确认 INFO 8 锚点（第二节）全对再离开

☐ 3. 训练中监控
   ☐ 3.1 每 20 步扫 CTCLoss 在不在降（ep3 < 0.5 正常）
   ☐ 3.2 lr 走势和剧本（第五节）对得上
   ☐ 3.3 每轮 eval 完截图存 val acc，连续 3 轮不涨 → Ctrl+C

☐ 4. 训练后评估
   ☐ 4.1 导出 inference 模型，dict.txt 一并归档
   ☐ 4.2 跑 script_verification/（21 张）做冒烟测试，准确率 >95% 才继续
   ☐ 4.3 跑 test_batch.py 测全部 test_sets
   ☐ 4.4 打开 incorrect_high_conf/info.txt：
         - 有内容 → 先查字典是否一致，再把这些图补进训练集重训
   ☐ 4.5 correct_low_conf/ 内容多 → 也挑一半补训练集

☐ 5. 上线部署
   ☐ 5.1 推理脚本 create_model 必须传 rec_char_dict_path（和训练字节级一致）
   ☐ 5.2 线上首批 100 张结果人工复核，incorrect_high_conf 率 <1% 再放量
   ☐ 5.3 每月跑一次回归测试（test_batch.py），新数据加训练集周更模型
```

---

> 📌 文档到此已完整覆盖：从字典原理、项目结构、数据校验、参数速查、Loss 诊断、训练流程、到推理部署和回归测试。建议把本文件直接钉在你 PaddleX 目录的最上方，跑训练前必过一遍第二节点 8 锚点和十四节 Checklist。
