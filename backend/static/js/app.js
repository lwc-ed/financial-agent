// backend/static/js/app.js
import { initHome } from "./pages/home.js";
import { initSaving } from "./pages/saving.js";

const LIFF_ID = "請替換成你的 LIFF ID";

export const state = {
  userId: "",
  userName: "",
  balance: 0,
};

async function initLineUser() {
  await liff.init({ liffId: LIFF_ID });
  if (!liff.isLoggedIn()) {
    liff.login();
    return;
  }
  const profile = await liff.getProfile();
  state.userId = profile.userId;
  state.userName = profile.displayName;
  state.balance = 0;
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
  await initLineUser();
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