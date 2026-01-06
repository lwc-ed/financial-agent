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

async function initSessionUser() {
  const res = await fetch("/api/check_user", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
    credentials: "include",
  });

  const data = await res.json();

  if (!data.exists) {
    window.location.href = "/login_page";
    return;
  }

  state.userId = data.user.line_user_id;
  state.userName = data.user.name;
  state.balance = data.user.balance || 0;
  console.log("登入成功 userId =", state.userId);
  // 不在這裡呼叫 updateUI
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
  await initSessionUser();
  showPage("home");
  updateUI(); // 🔴 這行一定要在 showPage 後
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