# 遠端操作指南
## 進虛擬環境
所有訓練套件都裝在虛擬環境裡了，不管跑哪個模型都進入下方虛擬環境
```bash
git checkout ... #切到自己的branch
tmux new -s lwc #建立自己的session，防止遠端掉線
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

## 如果斷線後可以重連自己的session
```bash
tmux attach -t lwc
```

## 跑完之後就刪掉自己的 session
```bash
tmux kill-session -t lwc
```

## 離開遠端
```bash
exit
```