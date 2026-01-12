from gate_find_ppo import execute_ppo_tool
test_result = execute_ppo_tool("load_and_test_ppo_agent", "./model/gate_search_ppo_model.pth", "gate")
print(f"\n测试结果: {test_result}")