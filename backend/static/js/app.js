// backend/static/js/app.js
import { initHome } from "./pages/home.js";
import { initSaving } from "./pages/saving.js";

const LIFF_ID = "請填入你的正式 LIFF ID";

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
      : "歡迎使用";
  }
}

/* ===== 唯一登入流程（LIFF） ===== */
async function initFromLIFF() {
  if (typeof liff === "undefined") {
    throw new Error("LIFF SDK not loaded");
  }

  await liff.init({ liffId: LIFF_ID });

  if (!liff.isLoggedIn()) {
    liff.login();
    return;
  }

  const profile = await liff.getProfile();
  state.lineUserId = profile.userId;

  console.log("[LIFF] line_user_id =", state.lineUserId);

  const res = await fetch("/api/check_user", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      line_user_id: state.lineUserId,
    }),
  });

  if (!res.ok) {
    throw new Error(`check_user failed: ${res.status}`);
  }

  const data = await res.json();

  if (!data.exists) {
    window.location.href = "/login_page";
    return;
  }

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
    alert("初始化失敗，請務必從 LINE 內開啟");
  }
});