from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Annotated
import base_models
from database import engine, SessionLocal
from sqlalchemy.orm import Session

app = FastAPI()
base_models.Base.metadata.create_all(bind=engine)

class PostBase(BaseModel):
    title: str
    content: str
    user_id: int

class UserBase(BaseModel):
    username: str
    password: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




db_dependency = Annotated[Session, Depends(get_db)]


@app.post("/posts/", status_code=status.HTTP_201_CREATED)
async def create_post(post:PostBase, db: db_dependency):
    db_post = base_models.Post(**post.dict())
    db.add(db_post)
    db.commit()

@app.get("/posts/{post_id}", status_code=status.HTTP_200_OK)
async def read_post(post_id: int, db:db_dependency):
    post =db.query(base_models.Post).filter(base_models.Post.id == post_id).first()
    if post is None:
        raise HTTPException(status_code=404, detail='Page was not found')
    return post
    
@app.delete("/posts/{post_id}", status_code=status.HTTP_200_OK)
async def delete_post(post_id: int, db: db_dependency):
    db_post=db.query(base_models.Post).filter(base_models.Post.id ==post_id).first()
    if db_post is None:
        raise HTTPException(status_code=404, detail='Post was not found')
    db.delete(db_post)
    db.commit()


@app.post("/users/", status_code=status.HTTP_201_CREATED)
async def create_user(user:UserBase, db: db_dependency):
    db_user = base_models.User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    # 別回傳密碼到前端
    return {"id": db_user.id, "username": db_user.username}

@app.get("/users/{user_id}", status_code=status.HTTP_200_OK)
async def read_user(user_id: int, db: db_dependency):
    user = db.query(base_models.User).filter(base_models.User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail='User not found')
    return {"id": user.id, "username": user.username}
