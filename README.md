# financial-agent

## 事前作業
1.  git pull -> 更新repo
2.  cd financial-agent
3.  **打開兩個終端機**  
    **終端機A:**  
    ```bash
    python3 app.py
    ```  
    **終端機B:**  
    ```bash
    ngrok http 8000
    ```  
4.  複製 `https://81f3d5915d67.ngrok-free.app` (在終端機B裡，要找一下)  
    範例：
    ```
    ngrok                                                                               (Ctrl+C to quit)
                                                                                                        
    🧱 Block threats before they reach your services with new WAF actions →  https://ngrok.com/r/waf    
                                                                                                        
    Session Status                online                                                                
    Account                       supergreatfinancialagent@gmail.com (Plan: Free)                       
    Version                       3.26.0                                                                
    Region                        Japan (jp)                                                            
    Latency                       40ms                                                                  
    Web Interface                 http://127.0.0.1:4040                                                 
    Forwarding                    ***https://13d62ea100e6.ngrok-free.app*** -> http://localhost:8000          
                                                                                                        
    Connections                   ttl     opn     rt1     rt5     p50     p90                           
                                0       0       0.00    0.00    0.00    0.00                          
                                                                                
    ```
5.  去[Line官方帳號](https://developers.line.biz/console/channel/2007892068/messaging-api)更改Webhook URL  
   `https://81f3d5915d67.ngrok-free.app/callback`（記得加）callback

## Line畫面
```
┌────────────┬────────────┬────────────┐
│   Area A   │   Area B   │   Area C   │  ← 上半部 (y=0 ~ 421)
├────────────┼────────────┼────────────┤
│   Area D   │   Area E   │   Area F   │  ← 下半部 (y=421 ~ 843)
└────────────┴────────────┴────────────┘
```