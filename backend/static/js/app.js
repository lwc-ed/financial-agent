// backend/static/js/app.js
import { initHome } from "./pages/home.js";
import { initSaving } from "./pages/saving.js";

const LIFF_ID = "2008065321-vlAGLNjW";

export const state = {
  lineUserId: "",
  userName: "",
  balance: 0,
};

/* ===== Drawer 控制 ===== */
let drawerOpen = false;
window.toggleDrawer = (open) => {
  const overlay = document.getElementById('drawer-overlay');
  const sideDrawer = document.getElementById('side-drawer');
  if (open !== undefined) drawerOpen = open;
  if (drawerOpen) {
    overlay.classList.remove('hidden');
    sideDrawer.style.right = '0px';
  } else {
    overlay.classList.add('hidden');
    sideDrawer.style.right = '-100%';
  }
};

function updateDrawerUI() {
  const userNameEl = document.getElementById('drawer-user-name');
  const userIdEl = document.getElementById('drawer-user-id');
  if (userNameEl) userNameEl.textContent = state.userName || '訪客';
  if (userIdEl) userIdEl.textContent = state.lineUserId.substring(0, 20) + '...';
}

/* ===== UI ===== */
function updateUI() {
  const hiUser = document.getElementById("hi-user");
  if (hiUser) {
    hiUser.textContent = state.userName
      ? `歡迎使用，${state.userName}`
      : "歡迎使用 LIFF";
  }
  updateDrawerUI();
}

/* ===== Dashboard LIFF 初始化 ===== */
async function initFromLIFF() {
  try {
    if (typeof liff === "undefined") {
      console.warn("LIFF SDK not loaded");
      return;
    }
    await liff.init({ liffId: LIFF_ID });
    if (!liff.isLoggedIn()) {
      window.location.href = "/login_page";
      return;
    }
    const profile = await liff.getProfile();
    state.lineUserId = profile.userId;
    state.userName = profile.displayName;
    console.log("[Dashboard LIFF] line_user_id =", state.lineUserId);
    updateUI();
    showPage("home");
  } catch (err) {
    console.error("Dashboard LIFF error:", err);
    state.userName = "訪客模式";
    updateUI();
    showPage("home");
  }
}

/* ===== Page 切換 ===== */
window.showPage = (page) => {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.getElementById(`page-${page}`)?.classList.add("active");
  if (page === "home") initHome(state);
  if (page === "saving") initSaving({ userId: state.lineUserId });
};

/* ===== Entry ===== */
window.addEventListener("DOMContentLoaded", async () => {
  await initFromLIFF();
});
