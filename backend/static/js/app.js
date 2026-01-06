// backend/static/js/app.js
import { initHome } from "./pages/home.js";
import { initSaving } from "./pages/saving.js";

const LIFF_ID = "你的正式 LIFF ID";

export const state = {
  lineUserId: "",
  userName: "",
  balance: 0,
};

/* ===== 只負責畫面 ===== */
function updateUI() {
  const hiUser = document.getElementById("hi-user");
  if (hiUser) {
    hiUser.textContent = state.userName
      ? `歡迎使用，${state.userName}`
      : "歡迎使用";
  }
}

/* ===== 唯一的登入流程：LIFF ===== */
async function initFromLIFF() {
  await liff.init({ liffId: LIFF_ID });

  if (!liff.isLoggedIn()) {
    liff.login();
    return;
  }

  const profile = await liff.getProfile();
  state.lineUserId = profile.userId;

  console.log("[LIFF] line_user_id =", state.lineUserId);

  // 👉 用 line_user_id 問後端「你是誰」
  const res = await fetch("/api/check_user", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      line_user_id: state.lineUserId,
    }),
  });

  if (!res.ok) {
    throw new Error(`check_user failed ${res.status}`);
  }

  const data = await res.json();

  if (!data.exists) {
    // 尚未註冊 / 尚未綁定
    window.location.href = "/login_page";
    return;
  }

  // ✅ 後端資料才是最終狀態
  state.userName = data.user.name;
  state.balance = data.user.balance || 0;
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
  try {
    await initFromLIFF();
    showPage("home");
    updateUI();
  } catch (err) {
    console.error("LIFF init failed", err);
    alert("初始化失敗，請從 LINE 重新開啟");
  }
});