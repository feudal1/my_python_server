import tkinter as tk
from tkinter import messagebox

class ArcCutCalculator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("弧切参数计算器")
        self.root.geometry("600x500")
        self.root.minsize(600, 500)
        
        # 输入变量
        self.outer_arc_var = tk.StringVar()
        self.inner_arc_var = tk.StringVar()
        
        self.create_widgets()
        
    def create_widgets(self):
        # 标题
        title_label = tk.Label(self.root, text="弧切参数计算器", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # 输入框架
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10)
        
        # 外弧长输入
        outer_label = tk.Label(input_frame, text="外弧长:", font=("Arial", 12))
        outer_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        outer_entry = tk.Entry(input_frame, textvariable=self.outer_arc_var, font=("Arial", 12), width=15)
        outer_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # 内弧长输入
        inner_label = tk.Label(input_frame, text="内弧长:", font=("Arial", 12))
        inner_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        inner_entry = tk.Entry(input_frame, textvariable=self.inner_arc_var, font=("Arial", 12), width=15)
        inner_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # 计算按钮
        calc_button = tk.Button(self.root, text="计算", command=self.calculate, 
                               font=("Arial", 12), bg="#4CAF50", fg="white", width=15)
        calc_button.pack(pady=20)
        
        # 结果文本框
        self.result_text = tk.Text(self.root, height=20, width=70, state=tk.DISABLED)
        self.result_text.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        scrollbar = tk.Scrollbar(self.result_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.result_text.yview)
        
    def calculate_arc_parameters(self, outer_arc_length, inner_arc_length):
        """
        计算弧切参数并返回结果
        """
        # 计算弧差
        arc_difference = outer_arc_length - inner_arc_length

        # 弯角补偿(固定值)
        bend_compensation = 20



        # 计算总长度
        total_length = arc_difference + bend_compensation
 
        # 口宽(固定值)
        mouth_width = 20.

        # 直段避让(固定值)
        straight_section_evasion = 20

        # 计算初始刀数
        initial_knife_count = total_length / mouth_width

        # 刀数最少取5刀
        final_knife_count = max(5, initial_knife_count)

        # 重新计算口宽
        recalculated_mouth_width = total_length / final_knife_count

        # 计算段数
        segment_count = final_knife_count - 1

        # 计算刀间距
        knife_spacing = (outer_arc_length-straight_section_evasion) / segment_count
        
        # 返回所有计算结果
        return {
            "弧差": arc_difference,
            "弯角补偿": bend_compensation,
            "总长度": total_length,
            "初始刀数": initial_knife_count,
            "最终刀数": final_knife_count,
            "重新计算的口宽": recalculated_mouth_width,
            "段数": segment_count,
            "刀间距": knife_spacing
        }
        
    def calculate(self):
        try:
            # 获取输入值
            outer_arc_length = float(self.outer_arc_var.get())
            inner_arc_length = float(self.inner_arc_var.get())
            
            # 验证输入
            if outer_arc_length <= inner_arc_length:
                messagebox.showerror("错误", "外弧长必须大于内弧长")
                return
                
            if outer_arc_length <= 0 or inner_arc_length <= 0:
                messagebox.showerror("错误", "弧长必须为正数")
                return
            
            # 执行计算
            results = self.calculate_arc_parameters(outer_arc_length, inner_arc_length)
            
            # 显示结果
            self.display_results(outer_arc_length, inner_arc_length, results)
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")
        except Exception as e:
            messagebox.showerror("错误", f"计算出错: {str(e)}")
            
    def display_results(self, outer_arc_length, inner_arc_length, results):
        # 清空结果文本框
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        # 插入计算过程
        self.result_text.insert(tk.END, "=== 输入参数 ===\n")
        self.result_text.insert(tk.END, f"外弧长: {outer_arc_length}\n")
        self.result_text.insert(tk.END, f"内弧长: {inner_arc_length}\n\n")
        
        self.result_text.insert(tk.END, "=== 计算过程 ===\n")
        self.result_text.insert(tk.END, f"弧差=外弧长{outer_arc_length}-内弧长{inner_arc_length}={results['弧差']}\n")
        self.result_text.insert(tk.END, f"弧差{results['弧差']}+弯角补偿{results['弯角补偿']}={results['总长度']}\n")
        self.result_text.insert(tk.END, f"刀数={results['总长度']}/口宽{20}={results['初始刀数']}\n")
        
        if results['初始刀数'] < 5:
            self.result_text.insert(tk.END, f"{results['初始刀数']}<5=5(刀数最少取5刀)\n")
        else:
            self.result_text.insert(tk.END, f"刀数={results['最终刀数']}\n")
            
        self.result_text.insert(tk.END, f"口宽={results['总长度']}/刀数{results['最终刀数']}={results['重新计算的口宽']}\n")
        self.result_text.insert(tk.END, f"段数=刀数{results['最终刀数']}-1={results['段数']}\n")
        self.result_text.insert(tk.END, f"刀间距=外弧长{outer_arc_length}/段数{results['段数']}={results['刀间距']}\n")
        
        self.result_text.insert(tk.END, "\n=== 结果摘要 ===\n")
        self.result_text.insert(tk.END, f"弧差: {results['弧差']}\n")
        self.result_text.insert(tk.END, f"最终刀数: {results['最终刀数']}\n")
        self.result_text.insert(tk.END, f"重新计算的口宽: {results['重新计算的口宽']:.2f}\n")
        self.result_text.insert(tk.END, f"刀间距: {results['刀间距']:.2f}\n")
        
        # 设置文本框为只读
        self.result_text.config(state=tk.DISABLED)
        
    def run(self):
        self.root.mainloop()

def main():
    app = ArcCutCalculator()
    app.run()

if __name__ == "__main__":
    main()