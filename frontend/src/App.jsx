const liffId = import.meta.env.VITE_LIFF_ID;
const API_BASE = import.meta.env.VITE_API_BASE;
import { useEffect, useState } from "react";

function App() {
  const [msg, setMsg] = useState("Loading...");

useEffect(() => {
  const url = `${import.meta.env.VITE_API_BASE}/api/hello`;
  console.log("要呼叫的 API URL:", url, "長度:", url.length);
  for (let i = 0; i < url.length; i++) {
    console.log(i, url.charCodeAt(i), url[i]);
  }

  fetch(url)
    .then(res => res.json())
    .then(data => setMsg(data.message))
    .catch(err => setMsg("API 錯誤：" + err));
}, []);

  return (
    <div>
      <h1>{msg}</h1>
      <p>API Base: {API_BASE || "⚠️ 沒讀到 .env"}</p>
    </div>
  );
}
console.log("ENV VITE_API_BASE =", import.meta.env.VITE_API_BASE);
export default App;