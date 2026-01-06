// backend/static/js/app.js
import { initHome } from "./pages/home.js";
import { initSaving } from "./pages/saving.js";

// 🔥 上線用正式 LIFF ID（從 login.html 複製）
const LIFF_ID = "2008065321-vlAGLNjW";

export const state = {
  lineUserId: "",
  userName: "",
  balance: 0,
};

/* ===== UI ===== */
function updateUI() {
  const hiUser = document.getElementById("hi-user");
  if (hiUser) {
    hiUser.textContent = state.userName
      ? `歡迎使用，${state.userName}`
      : "歡迎使用 LIFF";
  }
}

/* ===== Dashboard LIFF 初始化（簡化版） ===== */
async function initFromLIFF() {
  try {
    if (typeof liff === "undefined") {
      console.warn("LIFF SDK not loaded");
      return;
    }

    await liff.init({ liffId: LIFF_ID });
    
    if (!liff.isLoggedIn()) {
      console.warn("LIFF not logged in");
      window.location.href = "/login_page";
      return;
    }

    const profile = await liff.getProfile();
    state.lineUserId = profile.userId;
    state.userName = profile.displayName;  // 🔥 用 LIFF 名稱
    
    console.log("[Dashboard LIFF] line_user_id =", state.lineUserId);

    // 🔥 Dashboard 不重複 check_user（login_page 已驗證）
    // 直接顯示主頁
    updateUI();
    showPage("home");

  } catch (err) {
    console.error("Dashboard LIFF error:", err);
    // 不 alert，避免干擾用戶
    state.userName = "訪客模式";
    updateUI();
    showPage("home");
  }
}

/* ===== Page 切換 ===== */
window.showPage = (page) => {
  document.querySelectorAll(".page").forEach(p =>
    p.classList.remove("active")
  );
  document.getElementById(`page-${page}`)?.classList.add("active");

  if (page === "home") initHome(state);
  if (page === "saving") initSaving({ userId: state.lineUserId });
};

/* ===== Entry ===== */
window.addEventListener("DOMContentLoaded", async () => {
  await initFromLIFF();  // 🔥 非 try-catch，讓錯誤顯示在 console
});
