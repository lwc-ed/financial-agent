// saving.js
//const IS_LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
const IS_LOCAL = false

// 進化提示元素（動態建立）
let evolveToast = null;

function showEvolveToast(text) {
  if (!evolveToast) {
    evolveToast = document.createElement("div");
    evolveToast.className =
      "fixed inset-0 flex items-center justify-center pointer-events-none z-50";
    evolveToast.innerHTML = `
      <div id="evolve-toast-inner"
           class="bg-emerald-600 text-white px-6 py-3 rounded-xl text-lg font-bold
                  opacity-0 scale-75 transition-all duration-300">
      </div>
    `;
    document.body.appendChild(evolveToast);
  }

  const inner = document.getElementById("evolve-toast-inner");
  inner.textContent = text;

  // 顯示
  inner.classList.remove("opacity-0", "scale-75");
  inner.classList.add("opacity-100", "scale-100");

  // 自動隱藏
  setTimeout(() => {
    inner.classList.remove("opacity-100", "scale-100");
    inner.classList.add("opacity-0", "scale-75");
  }, 1200);
}

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
  { type: "chicken", icon: "🐣" },
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
    const res = await fetch(`/api/saving-challenge/list?line_user_id=${savingState.userId}`, {
      credentials: "include",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    console.log('🔥 API 回傳 pettype:', data.challenges);  // debug
    
    savingState.challenges = (data.challenges || []).map(ch => ({
      ...ch,
      pet_icon: getPetEmoji(ch.stage),
      pettype: ch.pettype  // 🔥 確保有 pettype
    }));
    
    console.log('🔥 前端狀態 pettype:', savingState.challenges.map(c => ({name: c.item_name, pettype: c.pettype})));
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

    const percent = Math.min(
      100,
      (ch.current_amount / ch.target_amount) * 100
    );

    const display = getPetDisplay(ch.stage, ch.pettype || 'cat');

    pet.innerHTML = `
      <div class="mb-2">
        ${renderPetVideo({
          src: display.src,
          loop: true,
          size: "w-16 h-16"
        })}
      </div>
      <div class="font-bold">${ch.item_name}</div>
      <div class="text-slate-500 text-[10px] mb-1">Stage ${ch.stage} · ${display.label}</div>
      <div class="w-full h-1 bg-slate-200 rounded overflow-hidden">
        <div class="h-full bg-emerald-500" style="width:${percent}%"></div>
      </div>
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

  // 🔥 已建立挑戰的 item_name 清單
  const usedItems = savingState.challenges.map(ch => ch.item_name);

  if (IS_LOCAL) {
    selectEl.innerHTML = "";

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

  // 🔥 Production：API + 已用過灰掉
  try {
    const res = await fetch(`/api/saving-challenge/wishlist?line_user_id=${savingState.userId}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    
    const data = await res.json();
    selectEl.innerHTML = '';

    if (!data.wishlist || data.wishlist.length === 0) {
      selectEl.innerHTML = '<option disabled selected>暫無願望清單</option>';
      return;
    }

    data.wishlist.forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.itemname;
      opt.dataset.price = item.price;
      
      // 🔥 已選過的願望：灰掉 + ✅ 標記
      if (usedItems.includes(item.itemname)) {
        opt.textContent = `✅ ${item.itemname}（$${item.price.toLocaleString()}）`;
        opt.disabled = true;  // 不可選
        opt.style.color = '#9CA3AF';  // 灰色
      } else {
        opt.textContent = `${item.itemname}（$${item.price.toLocaleString()}）`;
      }
      
      selectEl.appendChild(opt);
    });

  } catch (e) {
    console.error('載入願望清單失敗', e);
    selectEl.innerHTML = '<option disabled selected>載入失敗，請重試</option>';
  }
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

window.closePetPickerModal = () => {
  document.getElementById("pet-modal").classList.add("hidden");
};

/* ===============================
   Confirm pet selection
   =============================== */
window.confirmPetSelection = async () => {
  if (!savingState.selectedPet || !savingState.pendingChallenge) return;

  if (IS_LOCAL) {
    savingState.challenges.push({
      ...savingState.pendingChallenge,
      pet_type: savingState.selectedPet.type,
      pet_icon: savingState.selectedPet.icon,
      current_amount: 0,
      stage: 1
    });
  } else {
    const res = await fetch("/api/saving-challenge/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        line_user_id: savingState.userId,
        item_name: savingState.pendingChallenge.item_name,
        target_amount: savingState.pendingChallenge.target_amount,
        pettype: savingState.selectedPet.type  // 🔥 加這行！雞=chicken
      })
    });

    if (!res.ok) {
      alert("建立儲蓄挑戰失敗");
      return;
    }

    const data = await res.json();
    console.log('🔥 /list 回傳:', data.challenges);  // 🔥 看有沒有 pettype
    
    savingState.challenges.push({
      ...data.challenge,
      pet_icon: getPetEmoji(data.challenge.stage)
    });
  }

  savingState.pendingChallenge = null;
  savingState.selectedPet = null;

  closePetPickerModal();
  renderSavingPets();
};


let activePet = null;
let evolveTimer = null;

/* ===============================
   Helpers
   =============================== */


// 進化動畫對照表（fromStage-toStage）
const EVOLVE_ANIMATION_MAP = {
  "1-2": "evolve12.mp4",
  "2-3": "evolve23.mp4",
  "3-4": "evolve34.mp4"
};

// Stage calculation
function calcStage(current, target) {
  const ratio = current / target;
  if (ratio < 0.25) return 1;
  if (ratio < 0.5) return 2;
  if (ratio < 0.75) return 3;
  return 4;
}

// Temporary animation (replace with gif later)
function getPetEmoji(stage) {
  return ["🐣", "🐤", "🐥", "🦅"][stage - 1];
}

// Stage 對應顯示設定（之後可換成 gif / lottie）
function getPetDisplay(stage, petType) {
  return {
    src: `/static/assets/pets/${petType}/stage${stage}.mp4`,
    label: ["幼年", "成長中", "成熟期", "完成體"][stage - 1]
  };
}

function renderPetVideo({ src, loop = true, size = "w-24 h-24" }) {
  return `
    <video
      src="${src}"
      autoplay
      ${loop ? "loop" : ""}
      muted
      playsinline
      preload="auto"
      class="${size} object-contain"
      onerror="this.style.display='none'; this.nextElementSibling.style.display='block'"
    ></video>
  `;
}

window.closeWishlistPicker = () => {
  document.getElementById("wishlist-modal").classList.add("hidden");
};

function openPetDetail(challenge, opts = {}) {
  const { skipVideo = false } = opts;

  activePet = challenge;

  // const prevStage = challenge._lastStage ?? challenge.stage;
  // const newStage = calcStage(
  //   challenge.current_amount,
  //   challenge.target_amount
  // );

  // if (newStage > prevStage) {
  //   playEvolveTransition(prevStage, newStage);
  // }

  // challenge.stage = newStage;
  // challenge._lastStage = newStage;

  document.getElementById("pet-title").textContent = challenge.item_name;
  document.getElementById("pet-current").textContent =
    challenge.current_amount.toLocaleString();
  document.getElementById("pet-target").textContent =
    challenge.target_amount.toLocaleString();

  const petType = challenge.pettype || 'cat';
  const display = getPetDisplay(challenge.stage, petType);  
  document.getElementById("pet-stage").textContent =
    `Stage ${challenge.stage} · ${display.label}`;

  if (!skipVideo) {
    document.getElementById("pet-animation").innerHTML =
      renderPetVideo({
        src: display.src,
        loop: true,
        size: "w-28 h-28"
      });
  }

  const percent = Math.min(
    100,
    (challenge.current_amount / challenge.target_amount) * 100
  );
  document.getElementById("pet-progress").style.width = `${percent}%`;

  document.getElementById("feed-input").value = "";
  document
    .getElementById("pet-interact-modal")
    .classList.remove("hidden");
}

window.feedPet = async () => {
  if (!activePet) return;

  const input = document.getElementById("feed-input");
  const amount = Number(input.value);
  if (!amount || amount <= 0) return;

  // ⭐ 正確記住舊 stage
  const prevStage = activePet.stage;

  if (IS_LOCAL) {
    activePet.current_amount += amount;
    activePet.stage = calcStage(
      activePet.current_amount,
      activePet.target_amount
    );
  } else {
    const res = await fetch("/api/saving-challenge/feed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        line_user_id: savingState.userId,
        item_name: activePet.item_name,
        amount
      })
    });

    if (!res.ok) {
      alert("餵食失敗");
      return;
    }

    const data = await res.json();
    activePet.current_amount = data.challenge.current_amount;
    activePet.stage = data.challenge.stage;
  }

  // ⭐ 關鍵：只在這裡判斷進化
  if (activePet.stage > prevStage) {
    showEvolveToast("✨ 夥伴進化了！");

    // 先更新數字/進度/標題，但不要覆蓋影片區
    openPetDetail(activePet, { skipVideo: true });

    // 播放轉場，播完後再切回對應 stage loop（同時補一次 UI）
    playEvolveTransition(prevStage, activePet.stage, () => {
      openPetDetail(activePet); // 這次允許覆蓋影片，顯示 loop
    });

    renderSavingPets();
    return;
  }

  // 沒進化才直接更新畫面（含 loop 影片）
  openPetDetail(activePet);
  renderSavingPets();
};

window.closePetModal = () => {
  const modal = document.getElementById("pet-interact-modal");
  modal.classList.add("hidden");

  // 清空動畫區，避免殘留
  const anim = document.getElementById("pet-animation");
  if (anim) anim.innerHTML = "";

  activePet = null;
};
window.cancelPet = window.closePetModal;
window.closePetDetailModal = window.closePetModal;

function playEvolveTransition(fromStage, toStage, onDone) {
  const el = document.getElementById("pet-animation");
  if (!el || !activePet) return;

  // showEvolveToast moved to feedPet

  const key = `${fromStage}-${toStage}`;
  const evolveFile = EVOLVE_ANIMATION_MAP[key] || "evolve.mp4";
  // 🔥 修正：pettype
  const petType = activePet.pettype || 'cat';
  const evolveSrc = `/static/assets/pets/${petType}/${evolveFile}`;


  el.innerHTML = renderPetVideo({
    src: evolveSrc,
    loop: false,
    size: "w-32 h-32"
  });

  const video = el.querySelector("video");
  video.onended = () => {
    const display = getPetDisplay(toStage, activePet.pet_type);
    el.innerHTML = renderPetVideo({
      src: display.src,
      loop: true,
      size: "w-28 h-28"
    });

    if (typeof onDone === "function") onDone();
  };
}