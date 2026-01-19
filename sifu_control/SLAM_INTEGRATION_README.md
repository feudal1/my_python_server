# PPO + SLAM 地图集成说明

## 功能概述

将 Ground-SAM 特征提取与 SLAM（同步定位与地图构建）相结合，构建语义占用栅格地图，并融合到 PPO 训练状态中。

## 核心组件

### 1. SLAMMapBuilder - 轻量级 SLAM 地图构建器

```python
class SLAMMapBuilder:
    """基于Ground-SAM特征的轻量级SLAM地图构建器"""
```

**功能：**
- 构建语义占用栅格地图（100x100）
- 根据检测结果更新地图
- 记录路径历史
- 生成地图可视化图像

**地图栅格类型：**
- `0` = 未知（黑色）
- `1` = 空闲（白色）
- `2` = 障碍物（红色）
- `3` = 门/目标（绿色）
- `4` = 楼梯（蓝色）

### 2. PolicyNetwork SLAM 地图编码器

```python
self.slam_map_encoder = nn.Sequential(
    nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
    ...
    nn.Linear(128, 128),  # 输出 128 维特征
)
```

**输入：** 4通道地图特征
1. 占用状态（归一化）
2. 语义类别（归一化）
3. 访问频率
4. 距离中心的距离

**输出：** 128 维特征向量

### 3. 状态融合

```
SAM特征（256维） + SLAM地图特征（128维） = 融合特征（384维）
             ↓
    Actor/Critic网络（384 → 256 → 动作/价值）
```

## 可视化

### 主界面显示

- **左侧**：实时游戏画面 + YOLO检测框
- **右上角**：SLAM 小地图（200x200像素）

### 地图图例

| 颜色 | 含义 |
|------|------|
| 黑色 | 未知区域 |
| 白色 | 空闲区域 |
| 红色 | 障碍物 |
| 绿色 | 门/目标 |
| 蓝色 | 楼梯 |
| 黄色线 | 历史路径 |
| 紫色十字 | 当前位置 |

## 配置方法

### ppo_config.json

```json
{
    "USE_SLAM_MAP": true,
    "SLAM_MAP_SIZE": 100,
    "SLAM_MAP_RESOLUTION": 0.1
}
```

**参数说明：**
- `USE_SLAM_MAP`: 是否启用SLAM地图（默认 false）
- `SLAM_MAP_SIZE`: 地图尺寸（栅格数，默认 100）
- `SLAM_MAP_RESOLUTION`: 每栅格的实际距离（米，默认 0.1）

### 初始化 PPO Agent

```python
ppo_agent = PPOAgent(
    state_dim=(3, 480, 640),
    move_action_dim=4,
    turn_action_dim=2,
    use_ground_sam=True,
    use_slam_map=True  # 新增参数
)
```

## 使用场景

### 场景1：纯导航（当前推荐）

```json
{
    "USE_GROUND_SAM": true,
    "USE_SLAM_MAP": false
}
```

**优势：**
- 训练速度快
- 只使用 SAM 图像嵌入
- 适合空间推理任务

### 场景2：SLAM 辅助导航

```json
{
    "USE_GROUND_SAM": true,
    "USE_SLAM_MAP": true
}
```

**优势：**
- 提供全局环境理解
- 记录已探索区域
- 语义信息辅助决策

**注意：**
- 训练速度略慢（需要计算地图特征）
- 地图更新依赖 YOLO 检测结果
- 当前版本地图是相对坐标系

### 场景3：完整 SLAM 系统（未来扩展）

- **前端**：图像嵌入 + 光流 → 实时定位
- **后端**：自动分割 → 语义建图（关键帧）
- **循环检测**：对象匹配 → 位姿图优化

## 性能对比

| 配置 | 单步耗时 | 内存占用 | 适用场景 |
|------|---------|----------|---------|
| 无 Ground-SAM | ~50ms | 低 | 简单导航 |
| Ground-SAM（图像嵌入） | ~200ms | 中 | 复杂场景导航 |
| Ground-SAM + SLAM地图 | ~250ms | 中 | 语义导航 |
| Ground-SAM（自动分割） | ~2000ms | 高 | 对象识别 |

## 地图更新机制

### 每步更新

```python
slam_map_builder.update(
    detections,      # YOLO检测结果
    move_action,      # 移动动作
    move_step,       # 移动步长
    turn_action,      # 转向动作
    turn_angle       # 转向角度
)
```

### 地图映射逻辑

1. **屏幕坐标 → 地图坐标**
   ```python
   offset_x = (screen_x - 320) / 320.0  # 归一化 [-1, 1]
   offset_y = (screen_y - 240) / 240.0  # 归一化 [-1, 1]
   map_x = center + offset_x * 10
   map_y = center - offset_y * 10  # Y轴反转
   ```

2. **检测结果 → 栅格类型**
   ```python
   if 'gate' in label or 'door' in label:
       occupancy_grid[y, x] = 3  # 门
   elif 'climb' in label or 'stair' in label:
       occupancy_grid[y, x] = 4  # 楼梯
   else:
       occupancy_grid[y, x] = 2  # 障碍物
   ```

## 训练流程

### 1. 启用 SLAM

```bash
cd sifu_control
# 编辑 ppo_config.json: "USE_SLAM_MAP": true
python ppo_training.py continue_train_ppo_agent
```

### 2. 训练过程

```
Episode 0, Step 1:
- SAM特征提取: ~200ms
- SLAM地图更新: ~10ms
- SLAM地图编码: ~20ms
- PPO前向传播: ~30ms
总耗时: ~260ms/步

可视化界面:
┌─────────────────────┐
│  游戏画面 + YOLO  │
│                     │ [SLAM Map]
│                     │ [100x100]
└─────────────────────┘
```

### 3. 查看 SLAM 统计

```python
stats = slam_map_builder.get_stats()
print(f"探索率: {stats['explored_ratio']:.2%}")
print(f"障碍物: {stats['obstacle_count']}")
print(f"门: {stats['door_count']}")
```

## 扩展方向

### 1. 绝对坐标系 SLAM

当前版本使用相对坐标系（当前位置始终在中心）。

**改进：**
```python
class AbsoluteSLAMMapBuilder:
    def __init__(self, world_size=(50, 50)):
        self.global_pose = (0, 0)  # 世界坐标
        self.map = np.zeros(world_size)

    def update(self, detections, action, step, angle):
        # 根据动作更新全局位姿
        dx, dy = self._pose_from_action(action, step, angle)
        self.global_pose = (self.global_pose[0] + dx, self.global_pose[1] + dy)

        # 在全局地图上标记检测结果
        self._mark_global_objects(detections, self.global_pose)
```

### 2. 循环检测

```python
def loop_closure(self, current_map, historical_maps):
    for historical_map in historical_maps:
        similarity = self._compare_maps(current_map, historical_map)
        if similarity > 0.85:
            # 发现循环
            self._correct_drift(current_map, historical_map)
            break
```

### 3. 多层次地图

```python
# 粗粒度地图（100x100）
coarse_map = SLAMMapBuilder(map_size=100, resolution=0.1)

# 细粒度地图（200x200）
fine_map = SLAMMapBuilder(map_size=200, resolution=0.05)
```

## 故障排查

### 问题：地图不更新

**检查：**
1. `USE_SLAM_MAP` 是否为 `true`
2. YOLO 是否正常检测到对象
3. 环境是否调用了 `update_image_and_detections`

### 问题：地图全是黑色

**原因：** 没有检测结果

**解决：**
- 检查 YOLO 模型是否加载
- 检查 `DETECTION_CONFIDENCE` 是否太低
- 查看日志中是否有检测结果

### 问题：训练速度变慢

**原因：** SLAM 地图编码增加了计算量

**解决：**
- 减小 `SLAM_MAP_SIZE`（100 → 80）
- 使用 GPU 加速地图编码
- 降低地图更新频率（每 N 步更新一次）

## 相关文档

- `ppo_config.json` - 配置文件
- `ppo_training.py` - 主训练文件
- `EXPERIENCE_README.md` - 经验数据文档
- `SAM_OPTIMIZATION_README.md` - SAM优化文档

## 总结

SLAM 地图为 PPO 训练提供了全局环境理解，通过语义占用栅格记录已探索区域和障碍物信息。当前版本适合需要环境记忆的导航任务，未来可扩展为完整的 SLAM 系统支持循环检测和绝对定位。
