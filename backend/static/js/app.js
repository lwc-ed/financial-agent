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

  // 必須在 LINE App 內
  if (!liff.isInClient()) {
    document.body.innerHTML = "<p style='padding:20px'>請從 LINE App 開啟</p>";
    return;
  }

  // 尚未登入 → 走 LINE Login
  if (!liff.isLoggedIn()) {
    liff.login();
    return;
  }

  // 取得 LINE Profile
  const profile = await liff.getProfile();
  state.userId = profile.userId;
  state.userName = profile.displayName;
  state.balance = 45280; // 暫用
  updateUI();

  // 🔍 檢查是否已在後端註冊
  const res = await fetch("/api/check_user", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ line_user_id: state.userId })
  });

  const data = await res.json();

  if (data.exists === false) {
    // 未註冊 → 導向登入頁（避免重複導向）
    if (location.pathname !== "/login_page") {
      window.location.href = "/login_page";
    }
    return;
  }
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
// ===== expose functions for inline onclick (ES module fix) =====
window.showPage = window.showPage;
window.toggleDrawer = window.toggleDrawer;

// Wishlist / Saving Challenge
if (typeof window.openWishlistPicker === "function")
  window.openWishlistPicker = window.openWishlistPicker;
if (typeof window.closeWishlistPicker === "function")
  window.closeWishlistPicker = window.closeWishlistPicker;
if (typeof window.confirmCreateChallenge === "function")
  window.confirmCreateChallenge = window.confirmCreateChallenge;

// Pet picker modal
if (typeof window.openPetPickerModal === "function")
  window.openPetPickerModal = window.openPetPickerModal;
if (typeof window.closePetPickerModal === "function")
  window.closePetPickerModal = window.closePetPickerModal;
if (typeof window.confirmPetSelection === "function")
  window.confirmPetSelection = window.confirmPetSelection;

// Pet interaction modal
if (typeof window.openPetInteractModal === "function")
  window.openPetInteractModal = window.openPetInteractModal;
if (typeof window.closePetInteractModal === "function")
  window.closePetInteractModal = window.closePetInteractModal;
if (typeof window.feedPet === "function")
  window.feedPet = window.feedPet;