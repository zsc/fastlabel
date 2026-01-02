为了给 Claude 或 Codex 提供清晰的指导，让他们实现一个 **FastLabel** 的 Python+HTML Demo，我们需要一个详细的技术规格文档（spec）。下面是这个 **FastLabel** 项目的完整规范：

---

# **FastLabel Spec: Python+HTML Demo** (多模态主动学习标注)

## 1. **项目简介**

**目标**：创建一个 Python 后端（Flask）与前端（HTML）结合的 Demo，支持主动学习的标注流程。用户通过网页标注数据，系统基于已标注的数据进行增量训练。
**核心理念**：最大化复用图像标注的交互逻辑。对于音频数据，系统将其转换为**声谱图（Spectrogram）**，利用 CLIP 对声谱图的理解能力进行分类。

**主要应用场景**：
* **图像模式**：二分类任务（如：CelebA 人脸属性）。
* **音频模式**：基于声谱图的音频二分类（如：检测某特定语音指令、异常声响）。
* **核心目标**：**配置化数据源 + 快速标注 + 闭环反馈**。

## 2. **技术栈**

* **后端**：Python 3 + Flask (默认端口: **8008**)
  * **音频处理**：使用 `librosa` 或 `torchaudio` 将 `.wav` 转换为声谱图图像。
  * **模型**：`open_clip` (CLIP) 提取图像或声谱图特征 + `torch` MLP 分类头。
  * **存储**：内存（State），无需数据库。
* **前端**：HTML5 + CSS3 + Vanilla JS
  * **音频交互**：Web Audio API 或 HTML5 Audio，支持可视化播放游标（Cursor）。
* **数据源**：支持本地文件夹路径配置，不 Hardcode 数据集。

## 3. **整体工作流程**

### **1. 配置与初始化 (Configuration)**
* 新增 **Config Tab**。
* 用户输入/选择：
  * **数据类型**：`Image` 或 `Audio`。
  * **数据路径**：本地文件夹绝对路径（包含图片或 `.wav` 文件）。
* **预处理（音频特有）**：
  * 若选择 Audio 模式，后端需遍历文件夹，为每个 `.wav` 生成对应的 Mel-Spectrogram 图像并缓存（内存或临时目录），供 CLIP 提取特征使用。

### **2. 标注循环 (Active Learning Loop)**
* **加载数据**：根据策略（如 Random, Verify Positives）加载一批数据的**图像表示**（原图或声谱图）。
* **交互**：
  * **图像**：点击切换正/负。
  * **音频**：显示声谱图。点击图片切换正/负。**点击播放按钮**试听原音。
* **训练**：提交后，基于 CLIP 特征（图像/声谱图）训练 MLP。

### **3. 导出 (Export)**
* 用户可随时导出当前所有已标注数据的 ID、文件名、标签为 JSON 格式。

## 4. **后端：功能实现与 API**

### **数据模型扩展**
后端需抽象 `DataSource` 类，根据配置指向不同文件夹。
* **Image Mode**：直接读取图片文件。
* **Audio Mode**：读取 `.wav` -> 生成 `.png` 声谱图缓存 -> 读取该声谱图。

### **API 路由**

1.  **`/api/config` (POST)**
    *   输入：`{ "type": "image|audio", "path": "/path/to/data" }`
    *   逻辑：重置系统状态，扫描目录，(音频模式) 预生成/检查声谱图缓存，重新初始化 CLIP 特征缓存。

2.  **`/api/next_batch` (GET)**
    *   输出：`{ "items": [...], "batch_type": "..." }`
    *   **排序**：必须按置信度排序（验证正例时降序，验证负例时升序）。

3.  **`/api/image/<id>` (GET)**
    *   **Image Mode**：返回原图。
    *   **Audio Mode**：返回该音频对应的**声谱图**图像（用于前端展示和 CLIP 推理）。

4.  **`/api/audio/<id>` (GET)**
    *   仅 Audio Mode 有效。返回原始 `.wav` 音频流。

5.  **`/api/export` (GET)**
    *   下载 `labels.json`，格式：`[{"id": "filename.wav", "label": 1}, ...]`。

6.  **`/api/submit_labels` (POST)**
    *   提交标注结果。
    *   逻辑：**保存快照** -> 更新内存状态 -> 触发 `train_step()`。

7.  **`/api/undo` (POST)**
    *   作用：回退到上一次提交前的状态（包括已标注数据、未标注池、模型参数）。
    *   限制：最多回退 3 步。

## 5. **前端：用户交互**

### **界面布局**

包含三个 Tab：
1.  **Configuration**：设置数据源路径和类型。
2.  **Labeling**：核心标注网格。
3.  **Review All**：查看已标数据。

### **音频卡片交互设计 (Audio Mode)**

为了保持高密度和极速操作，音频卡片复用图像卡片的逻辑，但增加播放层：

```html
<div class="card is-positive" data-id="101">
    <!-- 1. 视觉层：显示声谱图 -->
    <img src="/api/image/101" />
    
    <!-- 2. 播放控制层（悬浮或角落） -->
    <div class="play-btn" onclick="playAudio(101, this)">▶</div>
    
    <!-- 3. 播放游标（仅播放时显示） -->
    <div class="cursor-overlay"></div>
    
    <!-- 4. 预测信息 -->
    <div class="overlay">98% Pos</div>
</div>
```

*   **标注操作**：点击卡片图片区域（非播放按钮区域） -> 切换 正/负 标签（绿/红框）。
*   **播放操作**：
    *   点击播放按钮 -> 播放 `/api/audio/101`。
    *   **互斥逻辑**：同一时间只能播放一个音频（前一个自动停止）。
    *   **游标动画**：播放时，一个半透明垂直红线（Cursor）从左向右扫过声谱图，时长与音频时长匹配。
    *   播放结束后游标消失。

## 6. **主动学习策略 (保持不变)**

所有策略（Zero-shot Text, Uncertainty, Verify Pos/Neg）均基于 CLIP 特征。在音频模式下，实际上是**"Text-to-Spectrogram"** 或 **"Spectrogram-based Classification"**。

## 7. **开发注意事项**

*   **性能**：音频转声谱图可能较慢，建议在 Config 阶段利用多线程预处理，或者采用 Lazy Loading + 缓存机制。
*   **Spectrogram 参数**：使用 `Log-Mel Spectrogram`，参数需调整以便生成视觉上纹理清晰的图像（利于 CLIP 识别）。
*   **浏览器兼容性**：确保大量 `<audio>` 标签或频繁创建 `AudioContext` 不会导致内存泄漏。建议使用单例 Audio 对象。

---

### 总结
这是一个通用的多模态标注工具。通过将“音频标注”转化为“看图听音”的任务，我们复用了图像标注的高效 UI，同时利用 CLIP 强大的跨模态（或对图像纹理的）表征能力来加速音频分类模型的冷启动。