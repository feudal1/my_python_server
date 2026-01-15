from sifu_control.ppo_agents import execute_ppo_tool
test_result = execute_ppo_tool("load_and_test_ppo_agent", "./model/gate_search_ppo_model_checkpoint_ep_40.pth", "gate", "0.2")
print(f"\n测试结果: {test_result}")