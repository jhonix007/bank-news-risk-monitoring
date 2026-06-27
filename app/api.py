from __future__ import annotations

import csv
import io
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user, get_optional_user
from app.auth.security import create_access_token, hash_password, verify_password
from app.config import settings
from app.database.base import Base
from app.database.models import NewsBatch, NewsItem, PredictionResult, User
from app.database.session import engine, get_db
from app.routes import auth, billing, news, subscriptions
from app.services.news_file_service import create_news_batch_from_upload
from app.services.subscription_service import get_active_subscription, is_subscription_active
from app.services.task_service import enqueue_batch


app = FastAPI(title="Bank News Risk Monitoring")
APP_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=APP_DIR / "view" / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "view" / "templates")


@app.on_event("startup")
def startup() -> None:
    settings.ensure_dirs()
    Base.metadata.create_all(bind=engine)


app.include_router(auth.router)
app.include_router(billing.router)
app.include_router(subscriptions.router)
app.include_router(news.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def render(request: Request, template: str, context: dict | None = None) -> HTMLResponse:
    context = context or {}
    db = next(get_db())
    try:
        user = get_optional_user(request, db)
        context.update({"request": request, "current_user": user})
        return templates.TemplateResponse(template, context)
    finally:
        db.close()


def render_page(request: Request, template: str, user: User | None, context: dict | None = None, status_code: int = 200):
    context = context or {}
    context.update({"request": request, "current_user": user})
    return templates.TemplateResponse(template, context, status_code=status_code)


def get_web_user_or_redirect(request: Request, db: Session) -> User | RedirectResponse:
    user = get_optional_user(request, db)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return user


def wants_json(request: Request) -> bool:
    return "application/json" in request.headers.get("accept", "")


def get_page_user(request: Request, db: Session) -> User | RedirectResponse:
    user = get_optional_user(request, db)
    if user is None:
        if wants_json(request):
            raise HTTPException(status_code=401, detail="Требуется авторизация.")
        return RedirectResponse("/login", status_code=303)
    return user


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return render(request, "index.html")


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return render(request, "register.html")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return render(request, "login.html")


@app.post("/register", response_class=HTMLResponse)
def register_submit(
    request: Request,
    db: Session = Depends(get_db),
    email: str = Form(...),
    password: str = Form(...),
):
    normalized_email = email.strip().lower()
    if len(password) < 6:
        return render_page(
            request,
            "register.html",
            None,
            {"error": "Пароль должен содержать минимум 6 символов.", "email": normalized_email},
            status_code=400,
        )
    if db.scalar(select(User).where(User.email == normalized_email)):
        return render_page(
            request,
            "register.html",
            None,
            {"error": "Пользователь с таким email уже существует.", "email": normalized_email},
            status_code=400,
        )
    user = User(email=normalized_email, hashed_password=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    redirect = RedirectResponse("/dashboard", status_code=303)
    redirect.set_cookie("access_token", create_access_token(user.id), httponly=True)
    return redirect


@app.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    db: Session = Depends(get_db),
    email: str = Form(...),
    password: str = Form(...),
):
    normalized_email = email.strip().lower()
    user = db.scalar(select(User).where(User.email == normalized_email))
    if not user or not verify_password(password, user.hashed_password):
        return render_page(
            request,
            "login.html",
            None,
            {"error": "Неверный email или пароль.", "email": normalized_email},
            status_code=401,
        )
    redirect = RedirectResponse("/dashboard", status_code=303)
    redirect.set_cookie("access_token", create_access_token(user.id), httponly=True)
    return redirect


@app.get("/logout")
def logout():
    redirect = RedirectResponse("/", status_code=303)
    redirect.delete_cookie("access_token")
    return redirect


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: Session = Depends(get_db),
):
    user = get_web_user_or_redirect(request, db)
    if isinstance(user, RedirectResponse):
        return user
    batches = list(
        db.scalars(
            select(NewsBatch).where(NewsBatch.user_id == user.id).order_by(NewsBatch.created_at.desc()).limit(5)
        )
    )
    processed_news = db.scalar(
        select(func.count(PredictionResult.id)).where(PredictionResult.user_id == user.id)
    ) or 0
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "current_user": user,
            "subscription": get_active_subscription(db, user.id),
            "batches": batches,
            "batch_count": db.scalar(select(func.count(NewsBatch.id)).where(NewsBatch.user_id == user.id)) or 0,
            "processed_news": processed_news,
        },
    )


@app.get("/billing", response_class=HTMLResponse)
def billing_page(
    request: Request,
    db: Session = Depends(get_db),
):
    user = get_web_user_or_redirect(request, db)
    if isinstance(user, RedirectResponse):
        return user
    return templates.TemplateResponse(
        "billing.html",
        {
            "request": request,
            "current_user": user,
            "subscription": get_active_subscription(db, user.id),
            "subscription_price": settings.subscription_price,
            "error": request.query_params.get("error"),
        },
    )


@app.get("/upload", response_class=HTMLResponse)
def upload_page(
    request: Request,
    db: Session = Depends(get_db),
):
    user = get_web_user_or_redirect(request, db)
    if isinstance(user, RedirectResponse):
        return user
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "current_user": user,
            "has_subscription": is_subscription_active(db, user.id),
        },
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_submit(
    request: Request,
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
):
    user = get_web_user_or_redirect(request, db)
    if isinstance(user, RedirectResponse):
        return user
    context = {"request": request, "current_user": user, "has_subscription": is_subscription_active(db, user.id)}
    if not context["has_subscription"]:
        context["error"] = "Для загрузки новостей нужна активная подписка."
        return templates.TemplateResponse("upload.html", context, status_code=403)
    content = await file.read()
    try:
        batch = create_news_batch_from_upload(db, user, file, content)
        db.commit()
        try:
            enqueue_batch(batch.id)
        except RuntimeError as exc:
            batch.status = "failed"
            batch.error_message = str(exc)
            db.commit()
            context["error"] = str(exc)
            return templates.TemplateResponse("upload.html", context, status_code=503)
    except ValueError as exc:
        db.rollback()
        context["error"] = str(exc)
        return templates.TemplateResponse("upload.html", context, status_code=400)
    context["upload_result"] = {
        "saved_count": batch.total_items,
        "batch_id": batch.id,
        "status": batch.status,
    }
    context["batch"] = batch
    return templates.TemplateResponse("upload.html", context)


@app.get("/jobs")
def jobs_page(
    request: Request,
    db: Session = Depends(get_db),
):
    user = get_page_user(request, db)
    if isinstance(user, RedirectResponse):
        return user
    batches = list(db.scalars(select(NewsBatch).where(NewsBatch.user_id == user.id).order_by(NewsBatch.created_at.desc())))
    if wants_json(request):
        return [
            {
                "id": batch.id,
                "batch_id": batch.id,
                "original_filename": batch.original_filename,
                "filename": batch.original_filename,
                "status": batch.status,
                "total_items": batch.total_items,
                "processed_items": batch.processed_items,
                "created_at": batch.created_at,
                "processed_at": batch.processed_at,
                "error_message": batch.error_message,
            }
            for batch in batches
        ]
    return templates.TemplateResponse(
        "jobs.html",
        {"request": request, "current_user": user, "batches": batches},
    )


@app.get("/jobs/{batch_id}")
def job_page(
    batch_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    user = get_page_user(request, db)
    if isinstance(user, RedirectResponse):
        return user
    batch = db.get(NewsBatch, batch_id)
    if not batch or batch.user_id != user.id:
        raise HTTPException(status_code=404, detail="Загрузка не найдена.")
    if wants_json(request):
        return {
            "id": batch.id,
            "batch_id": batch.id,
            "original_filename": batch.original_filename,
            "filename": batch.original_filename,
            "status": batch.status,
            "total_items": batch.total_items,
            "processed_items": batch.processed_items,
            "created_at": batch.created_at,
            "processed_at": batch.processed_at,
            "error_message": batch.error_message,
        }
    return templates.TemplateResponse(
        "jobs.html",
        {"request": request, "current_user": user, "batches": [batch], "uploaded": request.query_params.get("uploaded")},
    )


@app.get("/results/{batch_id}")
def results_page(
    batch_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    user = get_page_user(request, db)
    if isinstance(user, RedirectResponse):
        return user
    batch = db.get(NewsBatch, batch_id)
    if not batch or batch.user_id != user.id:
        raise HTTPException(status_code=404, detail="Результаты не найдены.")
    rows = (
        db.query(NewsItem, PredictionResult)
        .join(PredictionResult, PredictionResult.news_item_id == NewsItem.id)
        .filter(NewsItem.batch_id == batch_id, NewsItem.user_id == user.id)
        .order_by(PredictionResult.risk_score.desc())
        .all()
    )
    if wants_json(request):
        return [
            {
                "title": item.title,
                "entity_norm": item.entity_norm,
                "text_fragment": item.text_fragment,
                "risk_score": pred.risk_score,
                "alert_flag": pred.alert_flag,
                "model_name": pred.model_name,
                "threshold": pred.threshold,
            }
            for item, pred in rows
        ]
    return templates.TemplateResponse(
        "results.html",
        {"request": request, "current_user": user, "batch": batch, "rows": rows},
    )


@app.get("/results/{batch_id}/download")
def download_results(
    batch_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    batch = db.get(NewsBatch, batch_id)
    if not batch or batch.user_id != user.id:
        raise HTTPException(status_code=404, detail="Результаты не найдены.")
    rows = (
        db.query(NewsItem, PredictionResult)
        .join(PredictionResult, PredictionResult.news_item_id == NewsItem.id)
        .filter(NewsItem.batch_id == batch_id, NewsItem.user_id == user.id)
        .order_by(PredictionResult.risk_score.desc())
        .all()
    )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["title", "entity_norm", "text_fragment", "risk_score", "alert_flag", "model_name", "threshold"])
    for item, pred in rows:
        writer.writerow([item.title, item.entity_norm, item.text_fragment, pred.risk_score, pred.alert_flag, pred.model_name, pred.threshold])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=results_batch_{batch_id}.csv"},
    )
