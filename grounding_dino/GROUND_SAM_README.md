# Ground-SAM 视觉特征提取器

基于Grounding DINO和SAM的视觉特征提取模块,所有预训练模型均已冻结,用于强化学习等下游任务。

## 特性

- ✅ **预训练模型冻结**: SAM和Grounding DINO的所有预训练参数均已冻结,不参与训练
- ✅ **文本引导检测**: 支持通过文本提示进行目标检测
- ✅ **高质量分割**: 结合SAM进行精确的图像分割
- ✅ **批量处理**: 支持批量图像特征提取
- ✅ **PPO集成**: 提供与PPO强化学习算法的集成示例
- ✅ **掩码可视化**: 支持SAM自动分割掩码的可视化和保存

## 文件说明

- `ground_sam_feature_extractor.py` - 核心特征提取器模块
- `ground_sam_rl_integration.py` - 与强化学习(PPOR)集成示例
- `ground_sam_usage_example.py` - 使用示例和测试代码
- `demo_visualize_masks.py` - SAM掩码可视化演示脚本(位于grounding_dino目录)

## 快速开始

### 1. 基本使用

```python
from ground_sam_feature_extractor import get_groundsam_extractor
import numpy as np

# 获取特征提取器
extractor = get_groundsam_extractor(sam_model_type='vit_b')

# 创建图像
image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

# 提取特征
features = extractor.extract_features(image)
```

### 2. 文本引导的目标检测

```python
from ground_sam_feature_extractor import get_groundsam_extractor
from PIL import Image
import numpy as np

# 获取特征提取器
extractor = get_groundsam_extractor()

# 加载图像
image = Image.open("test.jpg")
image_np = np.array(image)

# 文本提示检测
text_prompt = "cat . dog . person ."
features = extractor.extract_features(image_np, text_prompt=text_prompt)

# 查看检测结果
if 'detection_results' in features:
    boxes = features['detection_results']['boxes']
    print(f"检测到 {len(boxes)} 个对象")
```

### 3. 与PPO强化学习集成

```python
from ground_sam_rl_integration import create_ppo_agent_with_ground_sam
import numpy as np

# 配置
config = {
    'sam_model_type': 'vit_b',
    'encoder_feature_dim': 256,
    'hidden_dim': 256,
    'learning_rate': 3e-4
}

# 创建PPO智能体
trainer = create_ppo_agent_with_ground_sam(config)

# 获取动作
observation = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
move_action, turn_action, value = trainer.agent.get_action(observation)
```

### 4. SAM掩码可视化

在PPO训练过程中可视化SAM自动分割结果:

```python
import numpy as np
from ground_sam_feature_extractor import get_groundsam_extractor

# 加载图像
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# 获取特征提取器
extractor = get_groundsam_extractor(sam_model_type='vit_b', device='cuda')

# 生成SAM掩码
with torch.no_grad():
    masks = extractor.sam_backbone.automatic_segmentation(image)

print(f"检测到 {len(masks)} 个对象")
```

**快速演示脚本:**

```bash
# 从grounding_dino目录运行演示
cd grounding_dino
python demo_visualize_masks.py

# 或从项目根目录运行
python demo_show_sam_masks.py --image your_image.jpg --output result.png
```

**在PPO训练中启用掩码可视化:**

在 `sifu_control/ppo_training.py` 中修改 `_extract_sam_embedding` 调用:

```python
# 提取SAM特征并可视化掩码
sam_embedding = self._extract_sam_embedding(
    image_np,
    visualize_mask=True,                      # 启用掩码可视化
    save_path='./output/sam_masks.png'       # 保存路径
)
```

可视化结果包含:
- 彩色半透明掩码叠加层
- 每个对象的边界框
- 信息标签(编号、面积、IoU分数)

详细使用说明请参考 [SAM_MASKS_README.md](./SAM_MASKS_README.md)

## API文档

### GroundSAMFeatureExtractor

主要特征提取类。

**初始化参数:**
- `sam_model_type`: SAM模型类型 ('vit_h', 'vit_l', 'vit_b')
- `dino_model_name`: Grounding DINO模型名称
- `sam_checkpoint_path`: SAM模型权重路径(可选)

**主要方法:**
- `extract_features(image, text_prompt=None, boxes=None)`: 提取图像特征
- `extract_image_features(image)`: 提取图像嵌入特征
- `extract_mask_features(image, boxes)`: 提取掩码特征
- `automatic_segmentation(image)`: SAM自动分割(无文本提示)
- `get_feature_dim()`: 获取特征维度

### FrozenSAMBackbone

冻结的SAM骨干网络。

**特性:**
- 所有预训练参数已冻结(`requires_grad=False`)
- 支持图像特征和掩码特征提取
- 支持批量处理
- **SAM自动分割**: 使用 `SamAutomaticMaskGenerator` 进行无文本提示的自动分割
- **掩码可视化**: 提供自动分割结果的可视化功能

**主要方法:**
- `extract_image_features(image)`: 提取图像嵌入特征
- `extract_mask_features(image, boxes)`: 提取给定边界框的掩码特征
- `automatic_segmentation(image)`: SAM自动分割,返回所有检测到的对象掩码

**自动分割参数:**
- `points_per_side`: 采样点网格密度(默认32)
- `pred_iou_thresh`: 预测IoU阈值(默认0.86)
- `stability_score_thresh`: 稳定性分数阈值(默认0.92)
- `min_mask_region_area`: 最小掩码区域面积(默认100像素)

### FrozenGroundingDINO

冻结的Grounding DINO模型。

**特性:**
- 所有预训练参数已冻结
- 文本引导的目标检测
- 提取检测特征和中间层特征

### PPOWithGroundSAM

结合Ground-SAM的PPO智能体。

**特性:**
- SAM特征提取器冻结
- SAM特征编码器可训练
- 支持移动和转向动作
- 包含价值网络

## 运行示例

```bash
# 运行使用示例
python grounding_dino/ground_sam_usage_example.py

# 运行集成示例
python grounding_dino/ground_sam_rl_integration.py

# SAM掩码可视化演示(从grounding_dino目录)
cd grounding_dino
python demo_visualize_masks.py

# 或从项目根目录运行完整演示
python demo_show_sam_masks.py

# 指定图像进行掩码可视化
python demo_show_sam_masks.py --image your_image.jpg --output result.png
```

## 模型说明

### SAM (Segment Anything Model)

提供三个预训练模型:
- `vit_h`: 最大模型,质量最高 (~2.4GB)
- `vit_l`: 中等模型,平衡性能 (~1.2GB)
- `vit_b`: 最小模型,速度最快 (~375MB)

### Grounding DINO

提供两个预训练模型:
- `IDEA-Research/grounding-dino-tiny`: 轻量级模型
- `IDEA-Research/grounding-dino-base`: 基础模型

## 参数冻结验证

所有预训练模型的参数均已设置 `requires_grad=False`,可以通过以下方式验证:

```python
extractor = get_groundsam_extractor()

# 检查SAM参数
for name, param in extractor.sam_backbone.sam.named_parameters():
    print(f"SAM {name}: requires_grad={param.requires_grad}")
    break

# 检查Grounding DINO参数
for name, param in extractor.grounding_dino.model.named_parameters():
    print(f"DINO {name}: requires_grad={param.requires_grad}")
    break
```

输出应显示 `requires_grad=False`,表示参数已冻结。

## 性能优化建议

1. **模型选择**: 根据任务需求选择合适的模型大小
   - 推理速度优先: 使用 `vit_b` + `grounding-dino-tiny`
   - 精度优先: 使用 `vit_h` + `grounding-dino-base`

2. **批处理**: 尽量使用批量处理提高效率

3. **缓存**: 使用单例模式,模型加载一次后缓存复用

4. **设备**: 确保在有CUDA支持的情况下使用GPU

## 依赖要求

- torch >= 2.0
- torchvision
- segment-anything
- modelscope
- transformers
- PIL/Pillow
- numpy
- opencv-python

## 注意事项

1. 首次运行时会自动下载预训练模型(约几百MB到几GB)
2. 确保有足够的磁盘空间存储模型缓存
3. 如遇到模型下载问题,可以手动下载并指定路径
4. 预训练模型已冻结,只能训练自定义的编码器和策略网络

## 常见问题

**Q: 如何更换模型?**
```python
# 在创建时指定不同的模型类型
extractor = get_groundsam_extractor(sam_model_type='vit_l')
```

**Q: 如何使用本地模型?**
```python
# 指定本地模型路径
extractor = GroundSAMFeatureExtractor(
    sam_model_type='vit_h',
    sam_checkpoint_path='/path/to/sam_vit_h_4b8939.pth'
)
```

**Q: 训练时为什么SAM参数不更新?**
SAM参数已被冻结,这是设计目标。只有SAM特征编码器和策略网络会参与训练。如果需要微调SAM,需要手动解冻参数(不推荐)。

**Q: 如何保存和加载训练的模型?**
```python
# 保存检查点
trainer.save_checkpoint('checkpoint.pth')

# 加载检查点
trainer.load_checkpoint('checkpoint.pth')
```

**Q: 如何在PPO训练中可视化SAM掩码?**
在 `sifu_control/ppo_training.py` 中启用掩码可视化:
```python
sam_embedding = self._extract_sam_embedding(
    image_np,
    visualize_mask=True,
    save_path='./sam_masks_step_1.png'
)
```

**Q: SAM掩码如何使用?**
SAM自动分割返回的每个掩码包含:
```python
{
    'segmentation': np.ndarray,  # 二值掩码 (H, W)
    'area': float,               # 面积(像素数)
    'bbox': tuple,               # 边界框 [x, y, w, h]
    'predicted_iou': float,      # 预测IoU分数
    'stability_score': float,    # 稳定性分数
}
```

详细使用方法请参考 `SAM_MASKS_README.md` 或运行 `demo_show_sam_masks.py` 查看演示。

## 许可证

本项目使用的预训练模型遵循其各自的许可证:
- SAM: Apache 2.0
- Grounding DINO: Apache 2.0
