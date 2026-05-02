import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import jieba
import re
from PIL import Image
from wordcloud import ImageColorGenerator # 從圖片提取顏色

# 準備英文停用詞庫 - NLTK
import nltk
nltk.download('stopwords')    #下載停用詞資料
from nltk.corpus import stopwords
en_stops = stopwords.words('english')

# 中文 - stopwordsiso
# import stopwordsiso
# zh_stops = stopwordsiso.stopwords("zh")


class ElegantWordCloudApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hash 詞頻文字雲分析系統")
        self.root.geometry("1000x800")
        
        # UI 字體放大設定
        self.title_font = ("Microsoft JhengHei", 16, "bold")
        self.ui_font = ("Microsoft JhengHei", 12)
        self.btn_font = ("Microsoft JhengHei", 12, "bold")

        # 視窗佈局權重配置
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        # 1. 控制區
        self.top_frame = tk.Frame(root, pady=15)
        self.top_frame.grid(row=0, column=0, sticky="ew", padx=30)
        
        tk.Label(self.top_frame, text="設定 Top N 詞數: ", font=self.title_font).pack(side=tk.LEFT)
        self.n_entry = tk.Entry(self.top_frame, width=8, font=self.ui_font)
        self.n_entry.insert(0, "30") # 預設更多一點詞，以便填滿形狀
        self.n_entry.pack(side=tk.LEFT, padx=10)

        # 2. 文字輸入區
        self.text_area = scrolledtext.ScrolledText(root, undo=True, font=("Consolas", 11))
        self.text_area.grid(row=1, column=0, sticky="nsew", padx=30, pady=10)

        # 3. 按鈕區
        self.btn_frame = tk.Frame(root, pady=25)
        self.btn_frame.grid(row=2, column=0)

        btn_style = {"font": self.btn_font, "padx": 20, "pady": 10, "cursor": "hand2"}
        
        tk.Button(self.btn_frame, text="📂 開啟檔案", command=self.load_file, **btn_style).pack(side=tk.LEFT, padx=15)
        tk.Button(self.btn_frame, text="☁️ 產生文字雲", command=self.create_elegant_cloud, bg="#FFDAB9", **btn_style).pack(side=tk.LEFT, padx=15)
        tk.Button(self.btn_frame, text="💾 儲存圖片", command=self.save_cloud, **btn_style).pack(side=tk.LEFT, padx=15)

        self.current_wc = None

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.text_area.delete("1.0", tk.END)
                self.text_area.insert(tk.END, f.read())

    def process_data(self, text, n):
        """核心 Hash 統計 + 進階過濾邏輯"""
        # A. 建立停用詞 Set (Hash Set, O(1) 查詢)
        # 英文 Stop words 與常見中文贅字
        # eng_stop = {"the", "a", "an", "is", "are", "am", "to", "in", "of", "and", "it", "that", "this", "with", "for", "on"}
        # chi_stop = {"的", "了", "在", "是", "我", "你", "他", "她", "它", "們", "也", "就", "都", "而", "及", "與", "這", "有"}
        # stop_words = eng_stop.union(chi_stop)
        
        # B. 中文斷詞 (支援自定義詞典以提升精煉度)
        # 如果你有 user_dict.txt，Jieba 會優先辨識裡面的詞 (如: 人工智慧)
        # try:
        #     jieba.load_userdict("user_dict.txt") # 你需要自己準備這個檔案
        # except:
        #     pass 
            
        raw_words = jieba.cut(text)
        
        # C. Hash Table 統計頻率
        hash_counts = {}
        for w in raw_words:
            w = w.strip().lower()
            
            # 過濾條件：
            # 1. 排除長度等於 1 (排除單個字)
            # 2. 排除停用詞
            # 3. 必須是字母或數字的組合 (排除標點符號與特殊雜訊)
            # 4. 使用正則表示式排除任何包含數字的詞

            if len(w) > 1 and w not in en_stops and w.isalnum():
                if not re.search(r'\d', w): 
                    hash_counts[w] = hash_counts.get(w, 0) + 1
        
        # D. 取出前 n 個高頻詞
        top_n = sorted(hash_counts.items(), key=lambda x: x[1], reverse=True)[:n]
        return dict(top_n)

    def create_elegant_cloud(self):
        content = self.text_area.get("1.0", tk.END).strip()
        try:
            n_val = int(self.n_entry.get())
        except ValueError:
            messagebox.showerror("錯誤", "n 請輸入數字")
            return

        if not content:
            messagebox.showwarning("警告", "請先輸入內容")
            return

        # 執行斷詞與 Hash 統計
        freq_data = self.process_data(content, n_val)

        if not freq_data:
            messagebox.showwarning("警告", "沒有足夠的詞彙可以產生。")
            return

        try:
            # 1. 處理形狀 Mask - 美化文字雲分布
            # 我們會嘗試讀取 hw2_mask.png (白色背景, 黑色形狀)
            try:
                mask_path = "hw2_mask.png" # 請在程式碼目錄下準備這個檔案
                mask_raw = np.array(Image.open(mask_path))
                
                # 處理 PIL 圖片數據格式，使其可以被 numpy 正常當作 mask 使用
                if len(mask_raw.shape) > 2: # 如果是 RGBA 圖片
                    mask_final = mask_raw[:, :, 0] # 僅取一個通道
                    # 將白色區域 (背景) 轉化為 255, 黑色 (形狀) 轉化為 0
                    mask_final = np.where(mask_final > 128, 255, 0).astype(np.uint8)
                else:
                    mask_final = mask_raw
                    
                use_mask = True
                bg_color = "white" # 當有 mask 時，背景用白色比較乾淨

            except Exception as e:
                # Fallback：如果沒有 hw2_mask.png，預設建立一個圓形的 mask
                
                print(f"找不到或無法讀取 hw2_mask.png ({e})，使用 Fallback 圓形 Mask")
                width, height = 1000, 1000
                x, y = np.ogrid[:height, :width]
                circle_mask = (x - height//2)**2 + (y - width//2)**2 > (width//2)**2
                mask_final = 255 * circle_mask.astype(int)
                use_mask = True
                bg_color = "black" # 圓形Fallback用黑色背景

            # 2. 繪製文字雲物件
            
            wc = WordCloud(
                font_path="msjh.ttc", # 中文字型，MacOS改為 /System/Library/Fonts/STHeiti Light.ttc
                width=1000, height=1000, 
                background_color=bg_color,
                mask=mask_final,        # 應用 Mask 形狀
                prefer_horizontal=1.0,  # 全橫排顯示
                max_words=n_val,
                colormap='plasma',   # 高對比,繽紛
                repeat=False,
                scale=2              # 提高渲染精度
            ).generate_from_frequencies(freq_data)

            self.current_wc = wc

            # 顯示結果
            plt.figure(figsize=(8, 8), facecolor='white')
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.tight_layout(pad=0) # 鋪滿視窗
            plt.show()
            
        except Exception as e:
            messagebox.showerror("錯誤", f"產生失敗: {str(e)}")

    def save_cloud(self):
        if self.current_wc:
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG圖片", "*.png")])
            if path:
                self.current_wc.to_file(path)
                messagebox.showinfo("成功", "圖片已存檔")

if __name__ == "__main__":
    root = tk.Tk()
    app = ElegantWordCloudApp(root)
    root.mainloop()
