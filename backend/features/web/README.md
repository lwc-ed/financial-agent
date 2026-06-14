# web

瀏覽器 / LIFF 前端入口：登入頁、儀表板、使用者建立與個人資料 API。前端頁面為 Flask 直接服務的純 HTML + LIFF SDK（非 React）。

- **對外接口（Blueprint）**：
  - `auth_bp` → `GET /login_page`（登入頁）
  - `dashboard_bp` → `GET /dashboard`（儀表板）
  - `liff_test_bp` → `POST /api/check_user`（依 LINE profile 查/建使用者）
  - `profile_bp` → `/api/profile/*`（個人資料 set/get）
- **主要檔案**：auth.py、dashboard.py、liff_test.py、profile.py
- **資料 / 模型**：user.py（`User` ORM）、liff/*.html（前端頁面）
- **相依**：core.database
