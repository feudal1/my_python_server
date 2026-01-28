name: Blender Mod制作助手
description: 专业的Blender 3D模型处理助手，提供模型导入、修复、缩放和骨骼绑定等完整工作流

detail: # Blender MOD 制作流程

## 第一步：准备工作
- **工具**: 无需特定工具
- **操作**: 确保Blender已安装并运行

## 第二步：激活Blender窗口
- **工具**: activate_blender_window
- **操作**: 调用此工具将Blender窗口激活并带到前台

## 第三步：清理场景
- **工具**: delete_all_objects
- **操作**: 清空当前Blender场景中的所有物体，为导入新模型做准备

## 第四步：导入模型
根据模型类型选择合适的导入工具：
- **PMX模型**: import_pmx
  - **参数**: file_path (PMX文件路径)
- **PSK模型**: import_psk
  - **参数**: file_path (PSK文件路径)

## 第五步：修复模型
- **工具**: fix_model
- **操作**: 执行模型修复操作，解决骨骼和网格问题

## 第六步：调整缩放
- **工具**: set_scale
  - **参数**: scale (缩放比例，默认1.0)
- **或工具**: scale_to_object_name
  - **参数**: object_name (目标物体名称)
- **操作**: 根据需要调整模型的缩放比例

## 第七步：骨骼绑定
- **工具**: set_parent_bone
  - **参数**: object_name (物体名称)
- **操作**: 将选中的对象设置为骨骼绑定父级

## 第八步：姿态模式
- **工具**: switch_pose_mode
  - **参数**: object_name (物体名称)
- **操作**: 切换到姿态模式并应用选中的骨架

## 第九步：顶点组传输
- **工具**: add_vertex_group_transfer
  - **参数**: object_name (物体名称)
- **操作**: 添加数据传输修改器并配置顶点组权重传输

## 第十步：清理工作
- **工具**: delete_object
  - **参数**: object_name (物体名称)
- **操作**: 删除不需要的物体

## 第十一步：打开Blender文件夹
- **工具**: open_blender_folder
- **操作**: 打开Blender的工作文件夹，方便查看和管理文件