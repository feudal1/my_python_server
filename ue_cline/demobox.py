import ue_api_tool

def main():
    while True:
        print("\n请选择要执行的操作：")
        print("1. 执行 import_fbx")
        print("2. 执行 build_sifu_mod")
        print("3. 退出程序")
        
        choice = input("请输入选项（1/2/3）: ").strip()
        
        if choice == "1":
            print("\n正在执行 import_fbx...")
            result = ue_api_tool.import_fbx()
            print(f"import_fbx 执行完成，结果: {result}")
            
        elif choice == "2":
            print("\n正在执行 build_sifu_mod...")
            result = ue_api_tool.build_sifu_mod()
            print(f"build_sifu_mod 执行完成，结果: {result}")
            
        elif choice == "3":
            print("程序退出")
            break
            
        else:
            print("无效选项，请重新输入")

if __name__ == "__main__":
    main()