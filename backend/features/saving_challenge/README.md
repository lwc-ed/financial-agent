# saving_challenge

儲蓄挑戰：使用者建立存錢目標，追蹤進度（搭配寵物養成的動態頁面）。

- **對外接口（Blueprint）**：`saving_challenge_bp` →
  `POST /api/saving-challenge/create`、`GET /list`、`GET /wishlist`、`POST /feed`
- **主要檔案**：saving_challenge.py（route）
- **資料 / 模型**：saving_challenge_model.py（`SavingChallenge` ORM）
- **相依**：core.database；前端寵物動畫資產位於 backend/static/assets/pets/
