// backend/static/js/app.js
import { initHome } from "./pages/home.js";
import { initSaving } from "./pages/saving.js";

export const state = {
  userId: "",
  userName: "",
  balance: 0,
};

function updateUI() {
  const hiUser = document.getElementById("hi-user");
  const drawerName = document.getElementById("drawer-user-name");
  const drawerId = document.getElementById("drawer-user-id");
  const balanceEl = document.getElementById("display-balance");

  if (hiUser) hiUser.textContent = `Hi: ${state.userName}`;
  if (drawerName) drawerName.textContent = state.userName;
  if (drawerId) drawerId.textContent = state.userId;
  if (balanceEl) balanceEl.textContent = `$${state.balance.toLocaleString()}`;
}

const IS_LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);

/* ===== 本地假登入 ===== */
async function initLocalMock() {
  state.userId = "LOCAL-TEST-USER-001";
  state.userName = "測試使用者";
  state.balance = 45280;
  updateUI();
}

/* ===== LIFF 登入 ===== */
async function initLIFF() {
  await liff.init({ liffId: "你的 LIFF ID" });
  const profile = await liff.getProfile();

  state.userId = profile.userId;
  state.userName = profile.displayName;
  state.balance = 45280; // 暫用
  updateUI();
}

/* ===== UI 共用 ===== */
window.showPage = (page) => {
  document.querySelectorAll(".page").forEach(p =>
    p.classList.remove("active")
  );
  document.getElementById(`page-${page}`)?.classList.add("active");

  // Toggle floating + button (only show on saving page)
  const fab = document.getElementById("fab-add-wishlist");
  if (fab) {
    fab.classList.toggle("hidden", page !== "saving");
  }

  if (page === "home") initHome(state);
  if (page === "saving") initSaving(state);
};

window.toggleDrawer = (show) => {
  document.getElementById("drawer-overlay")
    .classList.toggle("hidden", !show);
  document.getElementById("side-drawer").style.right =
    show ? "0" : "-100%";
};

/* ===== Entry ===== */
window.addEventListener("DOMContentLoaded", async () => {
  if (IS_LOCAL) {
    await initLocalMock();
  } else {
    await initLIFF();
  }
  updateUI();
  showPage("home");
});