# 遠端操作指南
## 進虛擬環境
所有訓練套件都裝在虛擬環境裡了，不管跑哪個模型都進入下方虛擬環境
```bash
tmux #防止遠端掉線
cd financial-agent
conda activate ./mlenv
```
## 跑訓練程式
```bash
cd #進入該資料夾
bash run_all.sh
```

## 離開虛擬環境
```bash
conda deactivate
```

## 離開遠端
```bash
exit
```