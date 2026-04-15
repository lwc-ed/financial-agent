import pickle
from pathlib import Path
import pandas as pd

# 1. 設定路徑 (指引程式去哪裡找你的心血結晶)
THIS_DIR = Path(__file__).resolve().parent
MODEL_PATH = THIS_DIR / "rf_output" / "best_rf_model.pkl"
OUTPUT_PATH = THIS_DIR / "rf_output" / "feature_importance_report.txt"

def main():
    print("🔍 開始解剖隨機森林大腦...")

    # 2. 打開我們剛才存好的「最強模型」保鮮盒
    try:
        with MODEL_PATH.open("rb") as f:
            saved_data = pickle.load(f)
    except FileNotFoundError:
        print("❌ 找不到模型檔案！請先確定你有跑過 my_first_rf.py 喔！")
        return

    # 把模型和它當時看過的特徵名單拿出來
    rf_model = saved_data["model"]
    features = saved_data["features"]

    # 3. 🌟 呼叫 RF 的隱藏超能力：抓出特徵重要性分數
    importances = rf_model.feature_importances_

    # 4. 把特徵名字和分數配對，然後「從高到低」排好隊
    feature_ranking = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # 5. 印出超酷的排行榜 (給你看的)
    print("\n🏆 特徵影響力排行榜 (前 10 名)：")
    print(feature_ranking.head(10).to_string(index=False))

    # 6. 自動把完整排行榜存成 txt 報告 (給專案用的)
    report_lines = [
        "Random Forest Feature Importance Report",
        "========================================",
        "這份報告顯示了模型在預測「未來7天花費」時，最看重哪些歷史線索。",
        ""
    ]
    
    # 把每一名依序寫入報告
    for idx, row in feature_ranking.iterrows():
        # 讓排版整齊一點
        report_lines.append(f"{row['Feature']:<25} {row['Importance']:.4f}")

    # 存檔！
    OUTPUT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n📄 完整排行榜已儲存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
