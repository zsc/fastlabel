为了给 Claude 或 Codex 提供清晰的指导，让他们实现一个 **FastLabel** 的 Python+HTML Demo，我们需要一个详细的技术规格文档（spec）。下面是这个 **FastLabel** 项目的完整规范：

---

# **FastLabel Spec: Python+HTML Demo** (多模态主动学习标注)

## 1. **项目简介**

**目标**：创建一个 Python 后端（Flask）与前端（HTML）结合的 Demo，支持主动学习的标注流程。用户通过网页标注数据，系统基于已标注的数据进行增量训练。
**核心理念**：最大化复用图像标注的交互逻辑。对于音频数据，系统将其转换为**Mel声谱图（Mel-Spectrogram）**，利用 CLIP 对声谱图的理解能力进行分类。
**示例数据**：系统默认以图像模式启动，并自动下载加载 **CelebA 48x48** 示例数据集（Sprite Sheet）。

**主要应用场景**：
* **图像模式**：二分类任务（如：CelebA 人脸属性）。
* **音频模式**：基于 Mel 声谱图的音频二分类（如：检测特定指令、异常声响）。
* **核心目标**：**配置化数据源 + 快速标注 + 闭环反馈**。

## 2. **技术栈**

* **后端**：Python 3 + Flask (默认端口: **8008**)
  * **音频处理**：使用 `librosa` 将 `.wav` 转换为 **Log-Mel Spectrogram**（频率采用 Mel 非线性缩放，强度采用 Log 缩放）。
  * **模型**：`open_clip` (CLIP) 提取图像或声谱图特征 + `torch` MLP 分类头。
  * **存储**：内存（State），具备 **Undo (至多 3 步)** 功能。
* **前端**：HTML5 + CSS3 + Vanilla JS
  * **异步加载**：利用 `IntersectionObserver` 实现无限滚动。
  * **流式合并**：Autolabel 采用分片后端拉取，前端动态**全量合并并全局置信度重排序**。
* **数据源**：支持本地文件夹绝对路径配置。

## 3. **整体工作流程**

### **1. 配置与初始化 (Configuration)**
* 默认加载 CelebA。
* 可在 **Config Tab** 切换数据类型（Image/Audio）和本地路径。
* 切换配置会重置内存中的标注状态和模型。

### **2. 标注循环 (Active Learning Loop)**
* **加载数据**：按策略（Random, K-Means, Borderline, Verify Pos/Neg）加载。
* **排序优化**：所有返回给前端的 Batch 必须按置信度排序（Verify Pos 降序，Verify Neg 升序）。
* **撤销功能**：支持 Undo 按钮，回退至多 3 次提交。

### **3. 数据全览 (Data Views)**
* **Review Confirmed**：展示用户已确认标注的数据，按标签分块展示。
* **Autolabel (AI)**：异步、非阻塞地流式加载**全量数据集**的预测结果。前端将新到的预测结果即时合并入本地列表并**触发全局排序**，确保高置信度样本始终动态置顶。

## 4. **后端：功能实现与 API**

### **API 路由**

1.  **`/api/config` (POST)**: 输入类型和路径，扫描目录，初始化状态。
2.  **`/api/next_batch` (GET)**: 获取下一批待标数据（包含 `undo_available` 标志）。
3.  **`/api/data_view?mode=model|dataset&offset=N&limit=K` (GET)**:
    *   `dataset`: 返回用户已标/未标状态。
    *   `model`: 返回模型预测。前端通过循环调用此接口覆盖全量数据。
4.  **`/api/image/<id>` (GET)**: 返回图像或 Mel 声谱图。
5.  **`/api/audio/<id>` (GET)**: 返回原始音频流。
6.  **`/api/submit_labels` (POST)**: 提交前**保存快照**，更新标注并训练。
7.  **`/api/undo` (POST)**: 恢复上一个快照。
8.  **`/api/export` (GET)**: 导出 `labels.json`。

## 5. **前端：用户交互**

### **界面布局 (四 Tab)**
1.  **Labeling**：主标注区，支持文本 Query (Zero-shot) 初始化。
2.  **Review Confirmed**：已标数据审核（仅显示用户已确认的部分）。
3.  **Autolabel (AI)**：全量数据 AI 预测视图，支持动态合并、重排序与无限滚动。
4.  **Configuration**：系统设置。

### **音频交互**
*   **Mel 声谱图展示**：作为卡片背景。
*   **互斥播放**：同一时间仅一个音频播放。
*   **同步游标**：播放时红线游标扫过声谱图，与音频进度实时同步。

---

### 总结
FastLabel 是一个面向实战的工具。它通过将“音频”转为“图像纹理”成功复用了 CLIP 的表征能力，并通过异步流式加载和全量数据的实时重排序确保了在处理万级数据集时 UI 的丝滑体验与结论的即时可见性。