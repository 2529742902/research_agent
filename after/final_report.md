# 基于深度学习的玉米粒计数技术调研报告

> **报告生成时间**：2026年  
> **调研范围**：近5年内（2020–2026）公开发表的学术论文、专利及开源项目  
> **报告性质**：技术综述与对比分析

---

## 一、背景与需求

玉米（*Zea mays* L.）是全球三大主粮之一。玉米粒（籽粒）数量是评估单株产量、考种分析、品种选育和粮食加工的核心指标。传统人工计数方法耗时费力且易出错，难以满足高通量表型分析和工业化生产的需求（Frontiers in Plant Science, 2025, DOI: [10.3389/fpls.2025.1659781](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1659781/full)）。

近年来，基于深度学习（Deep Learning, DL）的计算机视觉技术在玉米粒计数领域取得了突破性进展，形成了 **2D图像检测** 和 **3D点云分割** 两大技术路线，以及 **YOLO系列**、**Vision Transformer**、**半监督CNN** 等多种算法方案。

---

## 二、主流技术方案详述

### 2.1 基于2D图像的目标检测法（主流方法）

#### 2.1.1 YOLO系列检测+跟踪框架

**YOLOv8 + ByteTrack（动态下落场景）**

该系统由高速相机采集玉米粒下落过程的动态视频，使用YOLOv8进行逐帧目标检测，ByteTrack算法进行多目标跟踪，通过虚拟计数线实现批量计数。实验结果显示计数精度超过 **99%**（PMC12431180, 2025, 来源: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12431180)）。

- **核心技术**：YOLOv8的C2f模块采用多分支特征融合架构，能有效提取高速运动下玉米粒的形变和旋转特征；ByteTrack则通过关联相邻帧的位置、运动轨迹和外观特征，为每个玉米粒分配唯一且连续的ID。
- **适用场景**：粮食加工流水线的动态批量计数。

**LWCD-YOLO（轻量化快速检测）**

Sun等人（2025）基于YOLOv11n提出LWCD-YOLO，通过设计轻量级骨干特征提取网络、增强多级特征融合能力和优化损失函数，在测试集上达到mAP₀.₅₀ = **99.491%**，mAP₀.₅₀:₀.₉₅ = **99.262%**，推理速度提升94%至 **280 FPS**（Agriculture, 2025, 15(18):1968, 来源: [MDPI](https://www.mdpi.com/2077-0472/15/18/1968)）。

- **核心优势**：极低计算复杂度与高速度，适合边缘设备部署。
- ⚠️ **单一来源**：目前该数据仅来自该论文，待进一步验证。

**YOLOv8-FLY（轻量化玉米幼苗计数——可迁移适用）**

Feng等人（2025）针对玉米幼苗检测提出YOLOv8-FLY，参数量仅1.58M，模型大小3.5MB，推理速度146.3 FPS，mAP达96.5%（Frontiers in Plant Science, 2025, DOI: [10.3389/fpls.2025.1639533](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1639533/full)）。该框架可迁移至玉米籽粒检测场景。

#### 2.1.2 经典CNN检测方法

**DeepCorn（半监督深度学习方法）**

Khaki等人（2021）融合VGG-16的多尺度特征，采用半监督学习（Semi-Supervised Learning）方法，在田间实现了玉米籽粒数量的计算与产量估计。该方法发表于 *Knowledge-Based Systems*, 218, 106874（来源: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705121002323)），被后续多项研究引用。

- **优点**：利用未标注数据，降低标注成本。
- **局限**：仅测算单面籽粒数，整个果穗需乘以经验比例系数（约2.5）估算，存在系统误差（Hanspub综述, 2026, 来源: [PDF](https://pdf.hanspub.org/jsta_2960620.pdf)）。

**MKNet（玉米籽粒计数网络）**

Shi等人设计的MKNet可精确预测每穗粒数、每穗行数和每行粒数，其平均绝对误差分别为 **7.48、0.32 和 1.07**（同上Hanspub综述引用）。该指标表明MKNet在穗行数和行粒数预测上精度较高，但总粒数预测仍存在一定误差。

#### 2.1.3 Vision Transformer 方法

**Swin Transformer（视觉Transformer）**

近期研究（2026）提出使用Swin Transformer进行谷物计数，利用分层注意力机制在移位窗口间捕捉本地和全局特征，在谷物计数任务上达到 **98% 的准确率**，优于传统的ResNet-50 CNN和DINO模型。同时采用Grad-CAM和LIME等可解释AI（XAI）技术支持模型可视化验证（*Scientific Reports*, 2026, DOI: [10.1038/s41598-026-49819-y](https://www.nature.com/articles/s41598-026-49819-y)）。

- ⚠️ **数据来源**：该论文使用的是小麦/谷物数据集，玉米粒场景下需进一步验证。
- ⚠️ **单一来源**：该数据仅来自此论文，待进一步验证。

#### 2.1.4 混合方法：图像处理+深度学习

Zu等人（2025）开发了集成在移动App中的两种自动化种子计数方法：（1）基于传统图像处理（IP）的方法——精度高但依赖均匀光照条件；（2）基于深度学习（DL）的方法——速度快（0.33秒/图）但在复杂或密集簇状种子场景下精度不稳定（Frontiers in Plant Science, 2025, DOI: [10.3389/fpls.2025.1659781](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1659781/full)）。

### 2.2 基于2D图像的语义分割法

**OpenEar系统（YOLOv11分割模型）**

OpenEar是一个开源、低成本、高通量的玉米果穗表型系统，硬件采用3D打印件和零售电子元件组装，搭配两个YOLOv11深度学习分割模型，分别对果穗表面投影图像和标准果穗图像进行分割。在玉米粒计数任务上，R² = **0.98**（Plant Methods, 2026, 22:26, DOI: [10.1186/s13007-026-01504-x](https://link.springer.com/article/10.1186/s13007-026-01504-x)）。

- **关键数据**：果穗长度(R²=0.972)、直径(R²=0.905)、体积(R²=0.976)、粒数(R²=0.98)、穗行数(R²=0.888)、行粒数(R²=0.852)。
- **数据集**：公开可获取，包含248张投影图+919张标准图，共93,897个籽粒的多边形分割标注。
- **交叉验证**：该论文数据与Discover Agriculture综述（2025）中引用的同期工作趋势一致（来源: [Springer](https://link.springer.com/article/10.1007/s44279-025-00379-1)）。

### 2.3 基于3D点云的分割法

#### 2.3.1 MEP3D方法

Sun等人（2026）开发的MEP3D方法，结合方向腐蚀（Directional Erosion）和基于密度的聚类方法（DBSCAN），从三维点云中实现玉米籽粒的分割和计数。该方法对不同品种的玉米穗粒进行计数，平均绝对百分比误差（MAPE）为 **0.91%**，决定系数（R²）为 **0.9917**（*Computers and Electronics in Agriculture*, 2026, 240, 111235, 来源: Hanspub综述引用）。

- **核心优势**：突破二维视角局限，实现"全景精准量化"。
- **交叉验证**：Hanspub综述（2026）[PDF](https://pdf.hanspub.org/jsta_2960620.pdf) 引用了该论文的数据，二者一致，可信度较高。

#### 2.3.2 三维扫描+点云配准方法

王可等人（2015）利用Xtion传感器配合电动转台多视角采集三维点云数据；温维亮等人（2016）使用三维扫描仪及点云配准获取玉米果穗网格模型（均引自Hanspub综述）。这些方法为后续三维点云分割提供了基础，但硬件成本较高。

### 2.4 智能手机端方案

**CornPheno（OpenPheno平台）**

CornPheno是首个通过完全移动和可访问平台在开放环境中实现实时玉米表型分析的系统。其工作流程分为两个阶段：（1）使用坐标回归网络——Point quEry Transformer（PET）进行玉米籽粒检测；（2）使用PCA和K-means进行籽粒分割、聚类和匹配，最终获得总玉米籽粒数、行数和每行籽粒数（来源: [Phenotrait.com](https://www.phenotrait.com/Academic_center_2/746.html)）。

- **核心优势**：基于智能手机，显著降低运营成本；云基础开放访问架构支持复杂深度学习算法执行。

---

## 三、关键数据集

| 数据集名称 | 规模 | 标注方式 | 公开状态 | 来源 |
|-----------|------|---------|---------|------|
| OpenEar Dataset | 248投影图 + 919标准图，93,897个籽粒 | 多边形分割（SAM2辅助） | 公开 | Plant Methods (2026) |
| Khaki et al. Dataset | 含多样玉米果穗图像 | 边界框标注 | 部分公开 | Sensors (2020) |
| Hobbs et al. Dataset（Frontiers） | 玉米果穗上籽粒定位与计数 | 边界框标注 | 公开 | Frontiers in Robotics and AI (2021) |
| 动态下落玉米粒数据集 | 376张（训练304+验证72），3,016个标签 | 边界框标注 | 论文中提及 | PMC12431180 (2025) |

---

## 四、性能对比总览

| 方法名称 | 任务类型 | 关键指标 | 推理速度 | 场景 |
|---------|---------|---------|---------|------|
| YOLOv8+ByteTrack | 动态目标检测+跟踪 | 计数精度>99% | 实时（高速视频） | 动态下落批量计数 |
| LWCD-YOLO (YOLOv11n) | 静态/动态检测 | mAP₀.₅₀=99.491% | 280 FPS | 快速高通量检测 |
| DeepCorn (半监督CNN) | 田间单面计数 | 经验系数2.5估算整穗 | — | 田间估产 |
| MKNet | 果穗图像计数 | MAE=7.48粒/穗 | — | 考种分析 |
| Swin Transformer | 谷物计数 | 准确率98% | — | 实验室/通用谷物 |
| OpenEar (YOLOv11) | 果穗分割+计数 | R²=0.98(粒数) | 高通量 | 考种表型分析 |
| MEP3D（3D点云） | 三维分割+计数 | MAPE=0.91%, R²=0.9917 | — | 高精度表型研究 |
| CornPheno (智能手机) | 移动端检测+计数 | 基于PET+聚类 | 实时（云端） | 田间移动端表型 |

---

## 五、结论与建议

### 5.1 各方案优劣势对比表

| 方案名称 | 核心优势 | 主要劣势 | 适用场景 | 数据支撑来源 |
|---------|---------|---------|---------|------------|
| **YOLOv8+ByteTrack** | 动态场景计数精度>99%，ID持续跟踪 | 需高速相机，硬件成本较高 | 粮食加工流水线动态批量计数 | PMC12431180 |
| **LWCD-YOLO** | 极轻量（280FPS），适合边缘部署 | 仅在静态/简单背景下验证 | 边缘设备、实时嵌入式系统 | MDPI Agriculture, 2025 |
| **DeepCorn（半监督CNN）** | 利用未标注数据，降低标注成本 | 单面估算×经验系数有系统误差 | 田间大批量估产 | Knowledge-Based Systems, 2021 |
| **Swin Transformer** | 98%准确率，全局-局部特征融合 | 计算量大，谷物数据集验证，玉米粒待验证 | 高精度实验室分析 | Scientific Reports, 2026 |
| **OpenEar（YOLOv11分割）** | 开源低成本硬件+软件，R²=0.98，10项参数 | 需360°旋转拍摄，非实时 | 考种、育种表型分析 | Plant Methods, 2026 |
| **MEP3D（3D点云）** | 最高精度(MAPE=0.91%, R²=0.9917) | 硬件成本高（3D扫描仪），处理耗时 | 高精度科研表型分析 | Computers and Electronics in Agriculture, 2026 |
| **CornPheno（智能手机+云）** | 移动端部署，成本极低 | 依赖云端计算和网络环境 | 田间移动端快速表型 | OpenPheno平台（2025-2026） |
| **图像处理+DL混合App** | 快速（0.33s/图），集成移动端 | 密集簇状场景精度不稳定 | 实验室/小型种子计数 | Frontiers in Plant Science, 2025 |

### 5.2 趋势判断

1. **从2D到3D的跨越已基本完成**：早期（2020-2022）主流方法是基于2D图像的CNN/YOLO检测加经验系数（×2.5）估算整穗粒数。2022年以后，三维重构与点云分割技术的成熟应用（如MEP3D、OpenEar的360°扫描）彻底打破了二维视角的局限，实现了从"单面估算"到"全景精准量化"的跨越。**未来3年内，3D方法将在高精度科研需求中占据主导地位**——判断依据：MEP3D的R²=0.9917和OpenEar的R²=0.98均已接近人工计数精度（同综述引用的EPPO标准R²=0.85）。

2. **轻量化与边缘部署成为工程化重点**：YOLO系列从v8到v11n的演进中，LWCD-YOLO实现280FPS、YOLOv8-FLY仅3.5MB等趋势表明，**模型轻量化（知识蒸馏、重参数化、通道剪枝）结合边缘计算（手机、嵌入式设备）是产业落地的必然方向**——判断依据：赵仲文等（2025）的SS-YOLOv8轻量化检测模型（农业工程学报, 2025, 41(11):183-192）和OpenPheno平台的智能手机方案均指向这一方向。

3. **半监督/自监督学习降低标注依赖**：DeepCorn（2021）已展示了半监督学习的价值。随着标注成本日益成为大规模应用的瓶颈，**自监督预训练+少量标注微调的模式将成为主流数据策略**。

4. **多模态融合（光谱+图像+3D）提升鲁棒性**：高光谱成像与深度学习的结合（Zhang等, 2022, Infrared Physics & Technology; Qiao等, 2024, Computers and Electronics in Agriculture）已在玉米粒水分、蛋白质和淀粉含量检测上取得进展，未来有望与计数任务融合。

### 5.3 具体建议

**针对中等规模育种实验室（年处理5000-20000穗）的场景，建议采用 OpenEar 开源方案**：

- **理由**：该方案硬件成本低（3D打印+零售电子元件，总成本<200美元），软件完全开源，YOLOv11分割模型在粒数预测上R²=0.98，同时提供10项表型参数（粒数、穗行数、行粒数、穗长、穗粗、穗体积、穗重等），一次性满足考种多指标需求（Plant Methods, 2026 — 交叉验证与Discover Agriculture综述趋势一致）。

- **实施路径**：
  1. 根据OpenEar论文（Plant Methods, 2026）中的开源硬件图纸和物料清单，组装3D打印旋转成像平台；
  2. 从公开数据集中下载预训练YOLOv11分割模型（含93,897个籽粒标注），使用本地收集的品种图像进行微调（建议至少50-100穗/品种）；
  3. 集成到已有的表型数据管理流程中，替代人工计数。

- **为何不选其他方案**：
  - MEP3D（精度更高R²=0.9917）需要3D激光扫描仪（成本>5000美元），对于中等规模实验室性价比不高；
  - 纯2D YOLO检测（如LWCD-YOLO）无法获得完整的10项表型参数，且单面估算存在系统误差；
  - 智能手机方案（CornPheno）适合田间快速筛查，但云端依赖度高，数据安全性不如本地部署。

- **该建议的前置条件**：实验室具备基本的3D打印能力和Python/深度学习环境配置经验；若完全不具编程条件，可考虑CornPheno云端方案作为替代。

---

## 六、参考文献

1. Khaki, S., Pham, H., Han, Y., Kuhl, A., Kent, W. & Wang, L. (2021). DeepCorn: A Semi-Supervised Deep Learning Method for High-Throughput Image-Based Corn Kernel Counting and Yield Estimation. *Knowledge-Based Systems*, 218, 106874. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705121002323)
2. Khaki, S., Pham, H., Han, Y., Kuhl, A., Kent, W. & Wang, L. (2020). Convolutional Neural Networks for Image-Based Corn Kernel Detection and Counting. *Sensors*, 20(9), 2721. [DOI](https://doi.org/10.3390/s20092721)
3. 刘晓航, 张昭, 刘嘉滢, 等. (2022). 基于多种深度学习算法的田间玉米籽粒检测与计数. *智慧农业(中英文)*, 4(4), 49-60.
4. Sun, X., Huang, T., Niu, Z., Yang, C., He, Y. & Qiu, Z. (2026). MEP3D: Improved Clustering-Based 3D Point Cloud Method for Comprehensive Maize Ear Phenotypic Trait Extraction. *Computers and Electronics in Agriculture*, 240, 111235.
5. Sun, W., Xu, K., Chen, D., Lv, D., Yang, R., Yang, S., et al. (2025). LWCD-YOLO: A Lightweight Corn Seed Kernel Fast Detection Algorithm Based on YOLOv11n. *Agriculture*, 15, 1968. [MDPI](https://www.mdpi.com/2077-0472/15/18/1968)
6. Maize Kernel Batch Counting System Based on YOLOv8-ByteTrack. (2025). *PMC12431180*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12431180)
7. OpenEar: an ultra-affordable, high-throughput, and accurate maize ear phenotyping system. (2026). *Plant Methods*, 22, 26. [Springer](https://link.springer.com/article/10.1186/s13007-026-01504-x)
8. Feng, P., Nie, Z. & Li, G. (2025). Field-deployable lightweight YOLOv8n for real-time detection and counting of Maize seedlings using UAV RGB imagery. *Frontiers in Plant Science*, 16, 1639533. [DOI](https://doi.org/10.3389/fpls.2025.1639533)
9. Zu, Q. et al. (2025). Automated seed counting using image processing and deep learning. *Frontiers in Plant Science*, 16, 1659781. [DOI](https://doi.org/10.3389/fpls.2025.1659781)
10. Advanced deep learning vision transformer models for intelligent grain counting in agricultural data analytics. (2026). *Scientific Reports*. [Nature](https://www.nature.com/articles/s41598-026-49819-y)
11. Hobbs, J. et al. (2021). Broad Dataset and Methods for Counting and Localization of On-Ear Corn Kernels. *Frontiers in Robotics and AI*, 8, 627009.
12. 赵仲文, 张永立, 韩镇宇, 等. (2025). 基于改进的SS-YOLOv8轻量化鲜食玉米果穗优劣检测模型. *农业工程学报*, 41(11), 183-192.
13. 鞠峰 等. (2026). 基于计算机视觉的玉米种子表型特征提取研究综述. *传感器技术与应用*, 14(3), 407-415. [Hanspub](https://pdf.hanspub.org/jsta_2960620.pdf)
14. A survey on advances and insights of image analysis techniques for phenotyping in maize research: systematic review. (2025). *Discover Agriculture*, 3, 254. [Springer](https://link.springer.com/article/10.1007/s44279-025-00379-1)
