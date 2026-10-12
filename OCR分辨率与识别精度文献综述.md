# OCR 输入分辨率与识别精度文献综述

> 核心论点：提高输入分辨率能显著改善 OCR 识别精度，尤其对易混淆字符（如 1/7、0/8）的区分至关重要。

---

## 一、输入分辨率与 OCR 精度的直接关系

### 1. Dynamic Low-Resolution Distillation for Cost-Efficient End-to-End Text Spotting

- **作者**: Ying Chen, Liang Qiao, Zhanzhan Cheng 等
- **发表**: ECCV 2022
- **链接**: https://arxiv.org/abs/2207.06694
- **核心结论**: Input scale 是 OCR 最难权衡的 trade-off——识别小文本需要放大整张图，但计算成本极高。低分辨率模型精度显著低于高分辨率模型。提出动态分辨率选择器，在精度和成本之间取得平衡。

### 2. One-stage Low-resolution Text Recognition with High-resolution Knowledge Transfer

- **作者**: Hang Guo, Tao Dai, Mingyan Zhu 等
- **发表**: ACM MM 2023
- **链接**: https://arxiv.org/abs/2308.02770
- **核心结论**: 低分辨率文本识别面临 **信息缺失、噪声、模糊** 三大问题，是当前最难解决的场景之一。提出用高分辨率知识蒸馏到低分辨率模型（visual focus loss + semantic contrastive loss + soft logits loss），验证了分辨率对精度的决定性影响。代码: https://github.com/csguoh/KD-LTR

### 3. An end-to-end OCR approach for ultra-low-resolution printed text images

- **作者**: Julian D. Gilbey, Carola-Bibiane Schönlieb
- **发表**: 2021
- **链接**: https://arxiv.org/abs/2105.04515
- **核心结论**: 60 dpi 图像字符级准确率 99.7%，词级 98.9%；**75 dpi 时字符级提升到 99.9%，词级 99.4%**——仅提高 15 dpi，字符级精度就提升 0.2%（词级提升 0.5%）。直接证明了分辨率提升对精度的量化影响。

### 4. Scene Text Detection and Recognition with Aria Glasses Egocentric Vision Cameras

- **作者**: Joseph De Mathia, Carlos Francisco Moreno-García
- **发表**: 2025
- **链接**: https://arxiv.org/abs/2507.16330
- **核心结论**: 图像放大（upscaling）使 CER（字符错误率）从 0.65 降到 0.48，**相对改善 26%**。分辨率和距离对识别精度有显著影响。

---

## 二、超分辨率（SR）提升 OCR——间接证明分辨率是瓶颈

### 5. GLYPH-SR: VLM-guided Latent Diffusion for Text Super-Resolution

- **作者**: Mingyu Sung, Seungjae Ham, Kangwoo Kim 等
- **发表**: 2025
- **链接**: https://arxiv.org/abs/2510.26339
- **核心结论**: 当字符模糊或失真时，OCR 即使在其他区域清晰的情况下也会失败。SR 提升 OCR F1 **+15.18 个百分点**（x4/x8 放大）。满足可读性和视觉真实性双重需求。

### 6. TextSR: Diffusion Super-Resolution with Multilingual OCR Guidance

- **作者**: Keren Ye, Ignacio Garcia Dorado, Michalis Raptis 等
- **发表**: 2025
- **链接**: https://arxiv.org/abs/2505.23119
- **核心结论**: 利用 OCR 文本先验引导扩散超分模型，UTF-8 文本编码器 + 交叉注意力提供多语言字符形状先验，恢复字符细节和可辨性。在 TextZoom 和 TextVQA 上建立新 benchmark。

### 7. On-Device Text Image Super Resolution

- **作者**: Dhruval Jain, Arun D Prabhu, Gopi Ramena 等
- **发表**: ICPR 2020
- **链接**: https://arxiv.org/abs/2011.10251
- **核心结论**: SR 重建更锐利的字符边缘，提升 OCR 置信度。在 ICDAR 2015 TextSR 上达到 75.89%（GT 78.10%）。设备端推理仅需 11.7ms/图。

### 8. TATSR: Scene Text Image Super-Resolution via Content Perceptual Loss and Criss-Cross Transformer Blocks

- **作者**: Rui Qin, Bin Wang, Yu-Wing Tai
- **发表**: 2022
- **链接**: https://arxiv.org/abs/2210.06924
- **核心结论**: Content Perceptual Loss 利用多尺度文本识别特征监督文本重建。Criss-Cross Transformer Blocks 提取垂直和水平内容信息。在多种语言的识别精度和人类感知质量上均超越 SOTA。

### 9. DualTSR: Unified Dual-Diffusion Transformer for Scene Text Image Super-Resolution

- **作者**: Axi Niu, Kang Zhang, Qingsen Yan 等
- **发表**: 2026
- **链接**: https://arxiv.org/abs/2603.14207
- **核心结论**: 场景文本图像超分辨率恢复 LR 文本中的高分辨率细节——**对人类可读性和机器识别都至关重要**。统一端到端双扩散框架，连续分布处理图像，离散分布处理文本内容。

---

## 三、SVTR / PP-OCR 官方对分辨率的处理

### 10. SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition

- **作者**: Yongkun Du, Zhineng Chen, Hongtao Xie 等
- **发表**: ICCV 2025
- **链接**: https://arxiv.org/abs/2411.15858
- **核心结论**: **直接提出 multi-size resizing 策略**——将文本实例 resize 到合适的预定义尺寸，有效避免严重文本失真。引入特征重排模块确保视觉特征满足 CTC 要求。语义引导模块集成语言上下文到视觉特征（推理时可省略，无额外开销）。代码: https://github.com/Topdu/OpenOCR

### 11. PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System

- **作者**: Chenxia Li, Weiwei Liu, Ruoyu Guo 等
- **发表**: 2022
- **链接**: https://arxiv.org/abs/2206.03001
- **核心结论**: PP-OCRv3 将识别基线从 CRNN 替换为 SVTR_LCNet，引入 guided CTC training by attention、TextConAug 数据增强、TextRotNet 自监督预训练。Hmean 比 PP-OCRv2 提高 5%，推理速度相当。

### 12. SVTR: Scene Text Recognition with a Single Visual Model

- **作者**: Yongkun Du, Zhineng Chen, Caiyan Jia 等
- **发表**: IJCAI 2022
- **链接**: https://arxiv.org/abs/2205.00159
- **核心结论**: 提出 patch-wise 图像 tokenization，将图像分解为字符组件 patch，应用层次化全局/局部混合 block 实现多粒度感知。图像分辨率直接决定 patch 数量和特征粒度——**更高分辨率 = 更多 patch = 更细粒度特征**。

### 13. PaddleOCR 3.0 Technical Report

- **作者**: Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao 等（19 位作者）
- **发表**: 2025
- **链接**: https://arxiv.org/abs/2507.05595
- **核心结论**: PP-OCRv6 在数字显示器、点阵字符、轮胎印、工业文本识别等场景做了**重大改进**。提供 tiny (1.5M) / small (7.7M) / medium (34.5M) 三个级别，分别面向边缘、移动和服务器部署。

### 14. An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CRNN)

- **作者**: Baoguang Shi, Xiang Bai, Cong Yao
- **发表**: IEEE TPAMI 2017 (arXiv 2015)
- **链接**: https://arxiv.org/abs/1507.05717
- **核心结论**: CNN+RNN+CTC 经典架构，端到端可训练。自然处理任意长度序列，不需要字符分割或水平尺度归一化。不局限于预定义词典。模型远小于之前方法但效果相当。

---

## 四、易混淆字符与 Hard Negative Mining

### 15. NCL++: Nested Collaborative Learning for Long-Tailed Visual Recognition

- **作者**: Zichang Tan, Jun Li, Jinhao Du 等
- **发表**: 2023
- **链接**: https://arxiv.org/abs/2306.16709
- **核心结论**: **Hard Category Mining (HCM)**——选择预测分数高的负类别作为"硬类别"进行专门学习。在全类别（全局视角）和硬类别（局部视角）上同时训练，提升对 **易混淆类别** 的细致区分能力。

### 16. Billet Number Recognition Based on Test-Time Adaptation

- **作者**: Yuan Wei, Xiuzhuang Zhou
- **发表**: 2025
- **链接**: https://arxiv.org/abs/2502.09026
- **核心结论**: 使用 DB 网络 + SVTR 做工业钢号识别（机打+手写）。测试时自适应最小化模型熵以适应测试分布。**利用编码规则先验知识验证识别结果**——与表计数字后处理规则兜底思路一致。

### 17. Selectively Hard Negative Mining for Alleviating Gradient Vanishing in Image-Text Matching

- **作者**: Zheng Li, Caili Guo, Xin Wang 等
- **发表**: 2023
- **链接**: https://arxiv.org/abs/2303.00181
- **核心结论**: 当正负样本相似度差异趋近于 0 时，梯度消失。提出 **Selectively Hard Negative Mining (SelHN)**，基于梯度消失条件有选择性地挖掘硬负样本。即插即用，适用于各类匹配模型。

---

## 五、表计读数 / 数字显示 OCR

### 18. Deep Learning for Image-based Automatic Dial Meter Reading: Dataset and Baselines

- **作者**: Gabriel Salomon, Rayson Laroca, David Menotti
- **发表**: IJCNN 2020
- **链接**: https://arxiv.org/abs/2005.03106
- **核心结论**: **首个深度学习表计读数方案**。Faster R-CNN + YOLO 表盘检测 F1=100%，但字符识别仍有 6.4% 错误率（单表盘 93.6%，整表 75.25%）。错误分析指出**低分辨率和字符相似性是主要错误原因**。公开数据集: UFPR-ADMR。

---

## 六、视觉特征（含分辨率）对模型一致性的影响

### 19. Same Content, Different Answers: Cross-Modal Inconsistency in MLLMs

- **作者**: Angela van Sprang, Laurens Samson 等
- **发表**: CVPR 2026
- **链接**: https://arxiv.org/abs/2512.08923
- **核心结论**: 视觉特征（包括文本颜色和分辨率）影响模型性能。Vision token 数量直接影响 OCR 精度。即使 OCR 正确，视觉特征差异仍可能导致不一致。

### 20. MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios

- **作者**: Yang Shi, Huanqian Wang 等（22 位作者）
- **发表**: NeurIPS 2025
- **链接**: https://arxiv.org/abs/2505.21333
- **核心结论**: **高分辨率视觉输入对可靠 OCR 至关重要**。最强 MLLM（Gemini-2.5 Pro）仅达 73.7% 准确率。运动模糊和时间变化降低性能。

---

## 总结

| 论点 | 最强佐证 |
|---|---|
| 提高分辨率 → 精度提升 | Ultra-Low-Res OCR（60→75dpi 提升 0.2%），Aria Glasses（CER 降 26%） |
| 低分辨率 → 字符混淆 | GLYPH-SR（+15pp F1），On-Device SR（锐利字符边缘） |
| SVTR/PP-OCR 官方支持多尺度 | SVTRv2（multi-size resizing），PP-OCRv3（多尺度训练） |
| 易混淆字符需要专门处理 | NCL++（Hard Category Mining），Billet Number（先验验证） |
| 表计读数场景特殊性 | Dial Meter Reading（低分辨率+字符相似性是主要错误原因） |
