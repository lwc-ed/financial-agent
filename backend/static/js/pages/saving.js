// saving.js
const IS_LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);

/* ===============================
   Local mock data
   =============================== */
const MOCK_WISHLIST = [
  { item_name: "iPhone 16", price: 42000 },
  { item_name: "日本旅遊基金", price: 80000 },
  { item_name: "機械式鍵盤", price: 6500 }
];

const PET_TYPES = [
  { type: "cat", icon: "🐱" },
  { type: "dog", icon: "🐶" },
  { type: "rabbit", icon: "🐰" },
  { type: "dragon", icon: "🐲" },
  { type: "fox", icon: "🦊" },
  { type: "bear", icon: "🐻" },
  { type: "panda", icon: "🐼" },
  { type: "frog", icon: "🐸" },
  { type: "chick", icon: "🐣" },
  { type: "penguin", icon: "🐧" }
];

let savingState = {
  userId: null,
  challenges: [],   // active pets
  pendingChallenge: null,
  selectedPet: null
};

/* ===============================
   Page init
   =============================== */
export async function initSaving(state) {
  savingState.userId = state.userId;
  await loadSavingChallenges();
  renderSavingPets();
}

/* ===============================
   Load existing challenges
   =============================== */
async function loadSavingChallenges() {
  if (IS_LOCAL) {
    savingState.challenges = [];
    return;
  }

  try {
    const res = await fetch("/api/saving-challenge/list", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ line_user_id: savingState.userId })
    });
    const data = await res.json();
    savingState.challenges = data.items || [];
  } catch (err) {
    console.error("loadSavingChallenges failed", err);
  }
}

/* ===============================
   Render pet grid
   =============================== */
function renderSavingPets() {
  const container = document.getElementById("saving-pet-list");
  if (!container) return;

  container.innerHTML = "";

  if (savingState.challenges.length === 0) {
    container.innerHTML = `
      <div class="text-center text-slate-400 text-sm py-10">
        尚未開始任何儲蓄挑戰<br/>
        點擊右下角 + 新增第一個夥伴
      </div>
    `;
    return;
  }

  savingState.challenges.forEach(ch => {
    const pet = document.createElement("div");
    pet.className = "rounded-xl bg-slate-100 p-3 flex flex-col items-center text-xs";
    pet.innerHTML = `
      <div class="text-4xl mb-2">${ch.pet_icon || "🐣"}</div>
      <div class="font-bold">${ch.item_name}</div>
      <div class="text-slate-500">Stage ${ch.stage}</div>
    `;
    pet.onclick = () => openPetDetail(ch);
    container.appendChild(pet);
  });
}

/* ===============================
   Modal – wishlist picker
   =============================== */
window.openWishlistPicker = async () => {
  document.getElementById("wishlist-modal").classList.remove("hidden");
  const selectEl = document.getElementById("wishlist-select");
  selectEl.innerHTML = "<option>載入中...</option>";

  if (IS_LOCAL) {
    selectEl.innerHTML = "";

    // 已建立挑戰的 item_name 清單
    const usedItems = savingState.challenges.map(ch => ch.item_name);

    // 過濾尚未被使用的 wishlist
    const availableWishlist = MOCK_WISHLIST.filter(
      item => !usedItems.includes(item.item_name)
    );

    if (availableWishlist.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "已無可新增的願望項目";
      opt.disabled = true;
      opt.selected = true;
      selectEl.appendChild(opt);
      return;
    }

    availableWishlist.forEach(item => {
      const opt = document.createElement("option");
      opt.value = item.item_name;
      opt.textContent = `${item.item_name}（$${item.price.toLocaleString()}）`;
      opt.dataset.price = item.price;
      selectEl.appendChild(opt);
    });

    return;
  }

  // production: wishlist already comes from backend in previous step
};

/* ===============================
   Confirm create challenge (two-step)
   =============================== */
window.confirmCreateChallenge = async () => {
  const selectEl = document.getElementById("wishlist-select");
  const selected = selectEl.selectedOptions[0];
  if (!selected) return;

  if (savingState.challenges.length >= 10) {
    alert("最多只能同時培養 10 隻夥伴");
    return;
  }

  savingState.pendingChallenge = {
    item_name: selected.value,
    target_amount: Number(selected.dataset.price)
  };

  closeWishlistPicker();
  openPetModal();
};

/* ===============================
   Pet Modal logic
   =============================== */
function openPetModal() {
  const modal = document.getElementById("pet-modal");
  const grid = document.getElementById("pet-option-grid");
  modal.classList.remove("hidden");
  grid.innerHTML = "";

  PET_TYPES.forEach(pet => {
    const btn = document.createElement("button");
    btn.className = "text-3xl p-2 rounded border";
    btn.textContent = pet.icon;
    btn.onclick = () => {
      savingState.selectedPet = pet;
      [...grid.children].forEach(b => b.classList.remove("bg-emerald-100"));
      btn.classList.add("bg-emerald-100");
    };
    grid.appendChild(btn);
  });
}

window.closePetModal = () => {
  document.getElementById("pet-modal").classList.add("hidden");
  savingState.selectedPet = null;
};

/* ===============================
   Confirm pet selection
   =============================== */
window.confirmPetSelection = () => {
  if (!savingState.selectedPet || !savingState.pendingChallenge) return;

  savingState.challenges.push({
    ...savingState.pendingChallenge,
    pet_type: savingState.selectedPet.type,
    pet_icon: savingState.selectedPet.icon,
    current_amount: 0,
    stage: 1
  });

  savingState.pendingChallenge = null;
  savingState.selectedPet = null;

  closePetModal();
  renderSavingPets();
};

/* ===============================
   Helpers
   =============================== */
window.closeWishlistPicker = () => {
  document.getElementById("wishlist-modal").classList.add("hidden");
};

function openPetDetail(challenge) {
  alert(`進入 ${challenge.item_name} 的養成頁（下一步實作）`);
}