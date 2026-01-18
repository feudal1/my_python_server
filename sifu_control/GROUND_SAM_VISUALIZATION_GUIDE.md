# Ground-SAM 可视化使用指南

## 概述

Ground-SAM (Grounding DINO + SAM) 特征提取和可视化功能已集成到 PPO 训练系统中。可以在实时可视化窗口中显示 Grounding DINO 检测框和 SAM 分割掩码。

## 功能说明

### 1. 检测框可视化

- **蓝色边框**: Grounding DINO 检测到的目标
  - 标签格式: `DINO [类别]: [置信度]`
  - 默认检测阈值: `0.2` (可调整)
  - 默认文本阈值: `0.2` (可调整)

- **绿色边框**: YOLO 检测结果(保持不变)
  - 标签格式: `YOLO [类别]: [置信度]`

### 2. SAM 分割掩码可视化

- **彩色半透明叠加层**: 显示 SAM 生成的目标分割掩码
  - 每个掩码使用随机颜色
  - 透明度: 40% 掩码颜色 + 60% 原始图像
  - 掩码只在检测到对象时生成

### 3. 信息面板

新增字段:
- `Ground-SAM: Active/Inactive` - 显示 Ground-SAM 特征提取状态

## 配置

在 `ppo_config.json` 中配置:

```json
{
    "USE_GROUND_SAM": true,           // 是否启用Ground-SAM
    "SAM_MODEL_TYPE": "vit_b",        // SAM模型类型(vit_h, vit_l, vit_b)
    "DINO_MODEL_NAME": "IDEA-Research/grounding-dino-tiny",
    "SAM_FEATURE_DIM": 256            // SAM特征维度
}
```

## 使用方法

### 在训练中自动启用

当 `USE_GROUND_SAM` 为 `true` 时,训练过程会自动:

1. 在每个 step 中提取 Ground-SAM 特征
2. 在可视化窗口显示检测框和分割掩码
3. 在日志中输出特征提取信息

### 测试可视化功能

运行测试脚本:

```bash
# 测试掩码可视化
python sifu_control/test_mask_visualization.py

# 测试完整可视化(包含Tkinter窗口)
python sifu_control/test_ground_sam_visualization.py
```

## 调试信息

### 日志输出

系统会在日志中输出以下信息:

```
Ground-SAM特征键: ['image_features', 'detection_results', 'mask_features']
检测到 2 个对象
提取到 2 个掩码
绘制 2 个SAM分割掩码
```

### 掩码未显示的可能原因

1. **检测阈值太高**
   - 解决: 降低 `threshold` 和 `text_threshold` 参数
   - 默认值: `threshold=0.2`, `text_threshold=0.2`

2. **文本提示不匹配**
   - 确保 `text_prompt` 包含目标类别
   - 使用点号分隔多个类别: `"gate . door ."`

3. **图像质量或光照问题**
   - SAM 在低质量图像上表现较差
   - 确保图像清晰且对比度足够

4. **模型未正确加载**
   - 检查日志中的错误信息
   - 确认网络连接(首次运行需要下载模型)

## API 使用

### 提取特征并自定义阈值

```python
from grounding_dino.ground_sam_feature_extractor import get_groundsam_extractor

# 初始化提取器
extractor = get_groundsam_extractor()

# 提取特征(自定义阈值)
features = extractor.extract_features(
    image,
    text_prompt="gate . door .",
    threshold=0.15,      # 检测阈值
    text_threshold=0.15   # 文本阈值
)
```

### 手动绘制掩码

```python
import cv2
import numpy as np

# 特征字典中包含:
# - detection_results: 检测框信息
# - mask_features: 掩码列表

if 'mask_features' in features:
    for mask_info in features['mask_features']:
        mask = mask_info['mask']
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()

        # 绘制掩码
        mask_bool = mask.astype(bool)
        image[mask_bool] = [255, 0, 255]  # 紫色
```

## 性能优化建议

1. **批处理特征提取**: 如果有多张图像,可以批量处理
2. **缓存提取器**: 使用 `get_groundsam_extractor()` 单例模式
3. **调整阈值**: 根据场景调整检测阈值,平衡精度和速度
4. **GPU加速**: 确保CUDA可用,模型会自动使用GPU

## 故障排除

### 问题: 没有显示掩码

**检查步骤**:
1. 查看日志中的 "提取到 X 个掩码" 消息
2. 如果显示 "提取到 0 个掩码",说明没有检测到对象
3. 尝试降低阈值: `threshold=0.15, text_threshold=0.15`
4. 检查 `text_prompt` 是否正确

### 问题: 掩码不完整

**可能原因**:
- 检测框不准确
- 对象被遮挡
- 图像模糊

**解决方案**:
- 使用多个文本提示
- 提高图像质量
- 调整SAM模型类型(使用 `vit_h` 获得更好精度)

### 问题: 运行缓慢

**优化建议**:
1. 使用 `vit_b` 或 `vit_l` 而不是 `vit_h`
2. 减小图像尺寸
3. 降低 `points_per_side` 参数(在自动分割中)
4. 使用GPU加速

## 更新日志

### v1.1 (2026-01-18)
- 降低默认检测阈值从 0.3 到 0.2
- 添加详细的调试日志
- 改进掩码可视化(使用随机颜色)
- 添加掩码有效性检查

### v1.0 (2026-01-18)
- 初始版本
- 支持 Grounding DINO 检测框可视化
- 支持 SAM 分割掩码可视化
- 集成到 PPO 训练系统
