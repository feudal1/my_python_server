# PPO 经验数据持久化功能

## 功能说明

已为PPO训练添加完整的数据持久化功能,自动保存训练过程中的图像向量和动作记录,以便后续训练使用。

## 主要特性

1. **自动数据收集**
   - 在训练过程中自动保存每个步骤的图像状态
   - 记录所有动作信息(move, turn, 参数)
   - 保存奖励和完成标志

2. **高效存储**
   - 使用HDF5格式存储,支持大规模数据
   - 压缩存储,节省磁盘空间
   - 增量写入,性能优异

3. **离线训练支持**
   - 可以使用保存的数据进行离线训练
   - 无需与游戏环境交互
   - 支持批量训练和数据分析

## 配置说明

在 `ppo_config.json` 中新增以下配置项:

```json
{
    "SAVE_EXPERIENCE": true,              // 是否保存经验数据
    "EXPERIENCE_SAVE_DIR": "./experience_data",  // 保存目录
    "MAX_EXPERIENCES": 100000,           // 最大经验数
    "EXPERIENCE_FLUSH_INTERVAL": 100      // 刷新间隔(条数)
}
```

## 使用方法

### 1. 正常训练(自动收集)

运行PPO训练时,数据会自动保存:

```bash
cd sifu_control
python ppo_training.py continue_train_ppo_agent
```

训练过程中,数据会实时保存到 `experience_data/` 目录。

### 2. 查看已保存的数据

```bash
python experience_replay.py list
```

输出示例:
```
找到 1 个经验文件:
  experience_20250119_143025.h5
    经验数: 12345
    文件大小: 156.78 MB
```

### 3. 分析经验数据

```bash
python experience_replay.py analyze --path ./experience_data/experience_20250119_143025.h5
```

输出包括:
- 总经验数
- 状态形状和数值范围
- 动作分布统计
- 奖励统计
- 完成率等

### 4. 使用保存的数据进行离线训练

```bash
python experience_replay.py train --path ./experience_data/experience_20250119_143025.h5 --output ./model/offline_trained_model.pth --updates 1000
```

## 数据格式

HDF5文件包含以下数据集:

- `states`: 图像状态 (N, 3, H, W), uint8
- `move_actions`: 移动动作 (N,), int64
- `turn_actions`: 转向动作 (N,), int64
- `move_steps`: 移动步长 (N,), float32
- `turn_angles`: 转向角度 (N,), float32
- `rewards`: 奖励 (N,), float32
- `dones`: 完成标志 (N,), bool

元数据存储在 `/metadata` 组中:
- `total_experiences`: 总经验数
- `last_update`: 最后更新时间

## 训练流程

1. **数据收集阶段**
   - 运行PPO训练
   - 经验自动保存到HDF5文件
   - 每100条数据刷新一次到磁盘

2. **数据分析阶段**
   - 使用 `analyze` 命令查看数据质量
   - 确认数据分布合理

3. **离线训练阶段**
   - 使用保存的数据进行离线训练
   - 可以多次训练,无需重新收集

4. **模型部署**
   - 保存训练好的模型
   - 用于实际环境

## 性能优化

1. **内存优化**
   - 使用缓冲区,避免频繁I/O
   - 增量写入,减少内存占用

2. **存储优化**
   - HDF5压缩存储
   - uint8格式保存图像,节省空间

3. **训练优化**
   - 批量加载和处理
   - 支持数据打乱

## 注意事项

1. **磁盘空间**
   - 图像数据占用较大空间
   - 建议定期清理旧数据
   - 可以调整 `MAX_EXPERIENCES` 限制

2. **训练中断**
   - 程序被Ctrl+C中断时会自动保存
   - 数据会完整保存到磁盘

3. **数据质量**
   - 建议定期检查数据质量
   - 确保奖励分布合理
   - 检查动作多样性

## 故障排查

### 问题: 数据未保存

**解决方案:**
1. 检查配置 `SAVE_EXPERIENCE` 是否为 `true`
2. 检查磁盘空间是否充足
3. 检查写入权限

### 问题: 加载数据失败

**解决方案:**
1. 确认HDF5文件完整
2. 检查文件权限
3. 使用 `analyze` 命令验证数据

### 问题: 训练速度慢

**解决方案:**
1. 增加 `EXPERIENCE_FLUSH_INTERVAL`
2. 使用更快的存储设备(SSD)
3. 减小图像尺寸

## 扩展功能

可以根据需要添加以下功能:

1. **优先经验回放(PER)**
2. **数据增强**
3. **多任务学习支持**
4. **分布式训练**

## 联系方式

如有问题,请查看日志文件或联系开发者。
