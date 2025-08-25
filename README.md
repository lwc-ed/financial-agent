# financial-agent

## 事前作業
1. git pull -> 更新repo
2. cd financial-agent
3. **打開兩個終端機**  
   **終端機A:**  
   ```bash
   python3 app.py
   ```  
   **終端機B:**  
   ```bash
   ngrok http 8000
   ```  
4. 複製 `https://81f3d5915d67.ngrok-free.app` (在終端機B裡，要找一下)  
5. 去[Line官方帳號](https://developers.line.biz/console/channel/2007892068/messaging-api)更改Webhook URL  
   `https://81f3d5915d67.ngrok-free.app/callback`（記得加）callback

## Line畫面
```
┌────────────┬────────────┬────────────┐
│   Area A   │   Area B   │   Area C   │  ← 上半部 (y=0 ~ 421)
├────────────┼────────────┼────────────┤
│   Area D   │   Area E   │   Area F   │  ← 下半部 (y=421 ~ 843)
└────────────┴────────────┴────────────┘
```