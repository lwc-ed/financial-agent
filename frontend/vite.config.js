import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: [
      '44597ad8a092.ngrok-free.app'  // ✅ 換成你現在 ngrok 的 domain
    ]
  }
})