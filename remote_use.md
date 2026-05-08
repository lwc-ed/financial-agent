# 遠端操作指南
## 進虛擬環境
所有訓練套件都裝在虛擬環境裡了，不管跑哪個模型都進入下方虛擬環境
```bash
git checkout ... #切到自己的branch
tmux new -s lwc #建立自己的session，防止遠端掉線
cd financial-agent
conda activate ./mlenv #如果有缺套件就直接pip install
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

## 訓練跑著時，如果想離開遠端但保留程式：
```bash
Ctrl+b 然後按 d
```

## 如果斷線後可以重連自己的session
```bash
tmux attach -t lwc
```

## 如果確定不需要該session
```bash
exit
```

## 再來就是上傳 GitHub
```bash
git add .
git commit -m"commit messages" 
git push
```

## 離開遠端
```bash
exit
```