import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: [
      '9a66ccb60e44.ngrok-free.app'  // ✅ 換成你現在 ngrok 的 domain
    ]
  }
})