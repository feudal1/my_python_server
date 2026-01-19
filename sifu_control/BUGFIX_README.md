# Bug修复说明

## 问题描述

运行训练时出现 `KeyError: 'ENV'` 错误:

```
KeyError: 'ENV'
  File "E:\code\my_python_server\sifu_control\ppo_training.py", line 2279, in initialize_model
    env = config['ENV']()
```

## 原因分析

`ppo_training.py` 文件中存在重复的 `initialize_model` 函数:

1. **第2126行** - 正确版本
   - 使用 `create_environment_and_agent()`
   - 返回 `env, ppo_agent, visualizer, start_episode`
   - 正确导入所需的模块

2. **第2272行** - 错误版本(已注释)
   - 尝试使用 `config['ENV']`
   - 但配置文件中没有 `ENV` 键
   - 这是旧代码遗留的问题

## 修复方案

### 1. 注释掉错误的 `initialize_model` 函数 (✅ 已完成)

将第2272行的重复函数注释掉,避免名称冲突。

### 2. 更新 `continue_training_ppo_agent` 函数 (✅ 已完成)

移除对旧的 `perform_training_loop` 的调用,改用直接训练循环:

```python
# 旧代码(错误):
training_stats = perform_training_loop(env, ppo_agent, visualizer, start_episode, total_training_episodes)

# 新代码(正确):
for episode in range(start_episode, total_training_episodes):
    # 训练逻辑...
    ppo_agent.update(memory)
```

### 3. 更新 `train_new_ppo_agent` 函数 (✅ 已完成)

同样移除对 `perform_training_loop` 的调用。

### 4. 集成ExperienceCollector (✅ 已完成)

在训练循环中自动收集数据:

```python
# 定期输出经验收集统计
if episode % 50 == 0:
    collector = get_experience_collector()
    stats = collector.get_stats()
    print(f"经验收集统计: 总数={stats['total_experiences']}")
```

## 保留的代码

为了向后兼容,以下函数仍保留(但已标记为弃用):

- `EXPERIENCE_BUFFER` - 旧式经验缓冲区
- `train_with_experience_buffer` - 使用旧缓冲区训练
- `save_experience_buffer` - 保存旧缓冲区
- `load_experience_buffer` - 加载旧缓冲区
- `perform_training_loop` - 旧训练循环

这些函数不会影响新的训练流程。

## 使用方法

现在可以正常运行训练:

### 继续训练
```bash
cd sifu_control
python ppo_training.py continue_train_ppo_agent
```

### 从头训练
```bash
cd sifu_control
python ppo_training.py train_new_ppo_agent
```

### 评估模型
```bash
cd sifu_control
python ppo_training.py evaluate_trained_ppo_agent
```

## 验证修复

运行以下命令验证:

```bash
cd sifu_control
python ppo_training.py continue_train_ppo_agent
```

预期输出:
```
基于现有模型继续训练: ./model/gate_finder_ppo_enhanced.pth, 额外训练 2000 轮
从第 0 轮开始，继续训练 2000 轮，总共到第 2000 轮

=== Episode 0 Started ===
Episode 0, Total Reward: 1.23, Steps: 45
...
```

## 相关文件

- `ppo_training.py` - 主训练文件
- `ppo_config.json` - 配置文件
- `experience_replay.py` - 经验数据工具
- `EXPERIENCE_README.md` - 经验收集文档
- `SAM_OPTIMIZATION_README.md` - SAM优化文档
