import os
import json
import time
import hmac
import base64
import sqlite3
import hashlib
import secrets
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

import cv2
from fastapi import FastAPI, Body, Depends, HTTPException, Request, Response, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import live_cam_trt  # owns CUDA + TensorRT


# ==========================================================
# App
# ==========================================================
app = FastAPI(
    title="Wood-Chip Monitor API",
    description=(
        "Backend API for the Wood-Chip Monitor edge-AI "
        "quality-monitoring system."
    ),
    version="0.1.0",
)


# ==========================================================
# Paths / DB
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")

DB_PATH = os.environ.get(
    "WOODCHIP_DB_PATH",
    os.path.join(DATA_DIR, "woodchip_app.sqlite3"),
)

# Browser dashboard
static_dir = os.path.join(BASE_DIR, "web")

# Session cookie name
SESSION_COOKIE = "woodchip_session"

# Background threads flags
_inference_started = False
_sampler_started = False


# ==========================================================
# Time helpers
# ==========================================================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_epoch() -> float:
    return time.time()


def _parse_iso_dt(s: str) -> Optional[datetime]:
    """
    Parse ISO datetime safely across Python versions (incl. Py3.6).
    Returns tz-aware UTC datetime when possible.
    """
    if not s:
        return None

    s2 = s.strip().replace("Z", "+00:00")

    # Python 3.7+ supports datetime.fromisoformat
    try:
        dt = datetime.fromisoformat(s2)  # may not exist on Py3.6
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass

    # Fallback: dateutil if available
    try:
        from dateutil.parser import isoparse  # type: ignore
        dt = isoparse(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass

    # Last resort: strip offset/micros; assume UTC
    try:
        base = s2
        if "+" in base:
            base = base.split("+", 1)[0]
        if "-" in base[19:]:  # handle negative offset like ...-05:00
            base = base[:19]
        if "." in base:
            base = base.split(".", 1)[0]
        dt = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# ==========================================================
# RBAC model (simple + extensible)
# ==========================================================
ROLE_PERMS = {
    # Operator / Staff: can view live + events
    "staff": {"view_live", "view_events"},

    # QA: can export + edit rules
    "quality_engineer": {"view_live", "view_events", "export_events", "edit_rules", "view_audit"},

    # Software engineer: can see devices status + basic audit
    "software_engineer": {"view_live", "view_events", "export_events", "view_devices", "view_audit", "edit_rules"},

    # Manager: can do most things
    "manager": {"view_live", "view_events", "export_events", "edit_rules", "view_audit", "view_devices"},

    # Admin: everything
    "admin": {"*"},
}


def has_perm(role: str, perm: str) -> bool:
    perms = ROLE_PERMS.get(role, set())
    return ("*" in perms) or (perm in perms)


# ==========================================================
# Password hashing (stdlib only)
# ==========================================================
def _pbkdf2_hash_password(password: str, salt_b64: Optional[str] = None) -> Dict[str, str]:
    """
    Store: algo, iters, salt_b64, hash_b64
    """
    if salt_b64 is None:
        salt = secrets.token_bytes(16)
        salt_b64 = base64.b64encode(salt).decode("utf-8")
    else:
        salt = base64.b64decode(salt_b64.encode("utf-8"))

    iters = 200_000
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters, dklen=32)
    hash_b64 = base64.b64encode(dk).decode("utf-8")
    return {"algo": "pbkdf2_sha256", "iters": str(iters), "salt_b64": salt_b64, "hash_b64": hash_b64}


def _verify_password(password: str, stored: Dict[str, str]) -> bool:
    if stored.get("algo") != "pbkdf2_sha256":
        return False
    salt_b64 = stored.get("salt_b64")
    iters = int(stored.get("iters", "200000"))
    expected_b64 = stored.get("hash_b64")
    if not salt_b64 or not expected_b64:
        return False
    salt = base64.b64decode(salt_b64.encode("utf-8"))
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters, dklen=32)
    got_b64 = base64.b64encode(dk).decode("utf-8")
    return hmac.compare_digest(got_b64, expected_b64)


# ==========================================================
# DB helpers
# ==========================================================
_DB_LOCK = threading.Lock()


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if os.path.dirname(DB_PATH) else None

    with _DB_LOCK:
        conn = db_connect()
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            display_name TEXT,
            role TEXT NOT NULL,
            pw_algo TEXT NOT NULL,
            pw_iters INTEGER NOT NULL,
            pw_salt_b64 TEXT NOT NULL,
            pw_hash_b64 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,               -- random token
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            user_email TEXT,
            action TEXT NOT NULL,
            details_json TEXT
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS quality_rules_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            created_by_email TEXT,
            reason TEXT,
            rules_json TEXT NOT NULL,
            applied INTEGER NOT NULL DEFAULT 0,
            applied_at TEXT
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,              -- ISO UTC
            ts_epoch REAL NOT NULL,
            device_id TEXT,
            alarm_active INTEGER,
            alarm_max_d_mm REAL,
            mean_d REAL,
            std_d REAL,
            units TEXT,
            moisture_mean_pred TEXT,
            payload_json TEXT              -- full snapshot for future-proofing
        );
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_ts_epoch ON events(ts_epoch);
        """)

        conn.commit()
        conn.close()


def audit_log(action: str, user_email: Optional[str] = None, details: Optional[dict] = None) -> None:
    rec = {
        "ts": utc_now_iso(),
        "user_email": user_email,
        "action": action,
        "details_json": json.dumps(details or {}, ensure_ascii=False),
    }
    with _DB_LOCK:
        conn = db_connect()
        conn.execute(
            "INSERT INTO audit_log (ts, user_email, action, details_json) VALUES (?, ?, ?, ?)",
            (rec["ts"], rec["user_email"], rec["action"], rec["details_json"]),
        )
        conn.commit()
        conn.close()


# ==========================================================
# Auth helpers
# ==========================================================
def _get_session_id_from_request(request: Request) -> Optional[str]:
    return request.cookies.get(SESSION_COOKIE)


def _get_user_by_session(session_id: str) -> Optional[dict]:
    with _DB_LOCK:
        conn = db_connect()
        row = conn.execute("""
            SELECT s.id as session_id, s.expires_at, u.*
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ?
        """, (session_id,)).fetchone()
        if not row:
            conn.close()
            return None

        # Expiry check (Python 3.6-safe)
        expires_at = row["expires_at"]
        exp_dt = _parse_iso_dt(expires_at)
        if exp_dt is None:
            conn.close()
            return None

        if exp_dt < datetime.now(timezone.utc):
            # delete expired
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            conn.close()
            return None

        # Update last_seen
        conn.execute("UPDATE sessions SET last_seen_at = ? WHERE id = ?", (utc_now_iso(), session_id))
        conn.commit()
        conn.close()

        return dict(row)


def get_current_user(request: Request) -> dict:
    sid = _get_session_id_from_request(request)
    if not sid:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = _get_user_by_session(sid)
    if not user or int(user.get("is_active", 0)) != 1:
        raise HTTPException(status_code=401, detail="Invalid session")

    return {
        "id": user["id"],
        "email": user["email"],
        "display_name": user.get("display_name"),
        "role": user["role"],
    }


def require_perm(perm: str):
    def _dep(user: dict = Depends(get_current_user)) -> dict:
        role = user.get("role", "")
        if not has_perm(role, perm):
            raise HTTPException(status_code=403, detail=f"Missing permission: {perm}")
        return user
    return _dep


# ==========================================================
# Bootstrap admin user (optional)
# ==========================================================
def ensure_bootstrap_admin() -> None:
    """
    If there are no users, create an admin user from env vars:
      WOODCHIP_ADMIN_EMAIL / WOODCHIP_ADMIN_PASSWORD
    """
    admin_email = os.environ.get("WOODCHIP_ADMIN_EMAIL")
    admin_password = os.environ.get("WOODCHIP_ADMIN_PASSWORD")

    with _DB_LOCK:
        conn = db_connect()
        n = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
        conn.close()

    if n and n > 0:
        return

    if not admin_email or not admin_password:
        # no users + no bootstrap credentials: leave system locked until user table is seeded
        audit_log("bootstrap_admin_missing_env", None, {"note": "No users exist but admin env vars not set."})
        return

    pw = _pbkdf2_hash_password(admin_password)
    with _DB_LOCK:
        conn = db_connect()
        conn.execute("""
            INSERT INTO users (email, display_name, role, pw_algo, pw_iters, pw_salt_b64, pw_hash_b64, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            admin_email.lower().strip(),
            "Admin",
            "admin",
            pw["algo"],
            int(pw["iters"]),
            pw["salt_b64"],
            pw["hash_b64"],
            utc_now_iso(),
        ))
        conn.commit()
        conn.close()

    audit_log("bootstrap_admin_created", admin_email, {})


# ==========================================================
# Inference + sampler startup
# ==========================================================
@app.on_event("startup")
def on_startup():
    global _inference_started, _sampler_started

    db_init()
    ensure_bootstrap_admin()

    # Start inference loop once (critical for CUDA stability)
    if not _inference_started:
        _inference_started = True
        t = threading.Thread(
            target=live_cam_trt.run_inference_loop,
            kwargs={"headless": True},
            daemon=True,
        )
        t.start()

    # Start event sampler once (writes 1 row/sec into events table)
    if not _sampler_started:
        _sampler_started = True
        t2 = threading.Thread(target=_event_sampler_loop, daemon=True)
        t2.start()


def _safe_get(d: Any, key: str, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _event_sampler_loop():
    """
    Periodically snapshot runtime telemetry -> events DB.
    This is intentionally *lightweight* and does not touch CUDA.
    """
    device_id = os.environ.get("WOODCHIP_DEVICE_ID", "device-1")

    while True:
        try:
            stats = getattr(live_cam_trt, "LATEST_STATS", None)
            moist = getattr(live_cam_trt, "LATEST_MOISTURE", None)
            health = getattr(live_cam_trt, "LATEST_HEALTH", None)

            ts_iso = utc_now_iso()
            ts_epoch = utc_now_epoch()

            alarm_active = int(bool(_safe_get(stats, "alarm_active", False))) if isinstance(stats, dict) else 0
            alarm_max_d_mm = _safe_get(stats, "alarm_max_d_mm", None) if isinstance(stats, dict) else None
            units = _safe_get(stats, "units", None) or _safe_get(stats, "UNITS", None)
            mean_d = _safe_get(stats, "mean", None) or _safe_get(stats, "mean_d", None) or _safe_get(stats, "mean_diameter", None)
            std_d = _safe_get(stats, "std", None) or _safe_get(stats, "std_d", None) or _safe_get(stats, "std_diameter", None)

            moisture_mean_pred = None
            if isinstance(moist, dict) and moist.get("ready"):
                moisture_mean_pred = moist.get("mean_pred")

            payload = {
                "stats": stats if isinstance(stats, dict) else None,
                "moisture": moist if isinstance(moist, dict) else None,
                "health": health if isinstance(health, dict) else None,
            }

            with _DB_LOCK:
                conn = db_connect()
                conn.execute("""
                    INSERT INTO events
                    (ts, ts_epoch, device_id, alarm_active, alarm_max_d_mm, mean_d, std_d, units, moisture_mean_pred, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ts_iso,
                    float(ts_epoch),
                    device_id,
                    int(alarm_active),
                    float(alarm_max_d_mm) if alarm_max_d_mm is not None else None,
                    float(mean_d) if mean_d is not None else None,
                    float(std_d) if std_d is not None else None,
                    str(units) if units is not None else None,
                    str(moisture_mean_pred) if moisture_mean_pred is not None else None,
                    json.dumps(payload, ensure_ascii=False),
                ))
                conn.commit()
                conn.close()

        except Exception as e:
            # Don't crash sampler
            audit_log("event_sampler_error", None, {"error": str(e)})

        time.sleep(1.0)


# ==========================================================
# Public health check (keep)
# ==========================================================
@app.get("/ping")
def ping():
    return {"status": "ok"}


# ==========================================================
# Auth API
# ==========================================================
@app.post("/api/auth/login")
def login(resp: Response, payload: dict = Body(...)):
    email = str(payload.get("email", "")).lower().strip()
    password = str(payload.get("password", ""))

    if not email or not password:
        raise HTTPException(status_code=400, detail="Missing email or password")

    with _DB_LOCK:
        conn = db_connect()
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

    if not row:
        audit_log("login_failed", email, {"reason": "no_such_user"})
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = dict(row)
    stored = {
        "algo": user["pw_algo"],
        "iters": str(user["pw_iters"]),
        "salt_b64": user["pw_salt_b64"],
        "hash_b64": user["pw_hash_b64"],
    }

    if int(user.get("is_active", 0)) != 1 or not _verify_password(password, stored):
        audit_log("login_failed", email, {"reason": "bad_password_or_inactive"})
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create session (default 7 days)
    sid = secrets.token_urlsafe(32)
    now_iso = utc_now_iso()
    expires_at_iso = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

    with _DB_LOCK:
        conn = db_connect()
        conn.execute("""
            INSERT INTO sessions (id, user_id, created_at, last_seen_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
        """, (sid, user["id"], now_iso, now_iso, expires_at_iso))
        conn.commit()
        conn.close()

    resp.set_cookie(
        key=SESSION_COOKIE,
        value=sid,
        httponly=True,
        samesite="lax",
        secure=False,  # set True if behind HTTPS
        max_age=7 * 24 * 3600,
        path="/",
    )

    audit_log("login_ok", email, {"role": user["role"]})

    return {
        "ok": True,
        "user": {"email": user["email"], "display_name": user.get("display_name"), "role": user["role"]},
    }


@app.post("/api/auth/logout")
def logout(resp: Response, user: dict = Depends(get_current_user), request: Request = None):
    sid = _get_session_id_from_request(request) if request is not None else None
    if sid:
        with _DB_LOCK:
            conn = db_connect()
            conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            conn.commit()
            conn.close()

    resp.delete_cookie(SESSION_COOKIE, path="/")
    audit_log("logout", user.get("email"), {})
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: dict = Depends(get_current_user)):
    return {"ok": True, "user": user}


# ==========================================================
# Admin: create user (invite-only)
# ==========================================================
@app.post("/api/admin/users")
def admin_create_user(
    payload: dict = Body(...),
    admin: dict = Depends(require_perm("manage_users")) if has_perm("admin", "*") else Depends(get_current_user),
):
    # Since our ROLE_PERMS uses "*" for admin, simplest is to require admin role:
    if admin.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    email = str(payload.get("email", "")).lower().strip()
    password = str(payload.get("password", "")).strip()
    role = str(payload.get("role", "")).strip() or "staff"
    display_name = str(payload.get("display_name", "")).strip() or None

    if role not in ROLE_PERMS:
        raise HTTPException(status_code=400, detail=f"Unknown role: {role}")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Missing email or password")

    pw = _pbkdf2_hash_password(password)

    with _DB_LOCK:
        conn = db_connect()
        try:
            conn.execute("""
                INSERT INTO users (email, display_name, role, pw_algo, pw_iters, pw_salt_b64, pw_hash_b64, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (email, display_name, role, pw["algo"], int(pw["iters"]), pw["salt_b64"], pw["hash_b64"], utc_now_iso()))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            raise HTTPException(status_code=409, detail="User already exists")
        conn.close()

    audit_log("user_created", admin.get("email"), {"email": email, "role": role})
    return {"ok": True, "email": email, "role": role}


# ==========================================================
# Protected API: live stats/frame/hist/moisture + device health
# ==========================================================
@app.get("/api/stats")
def get_stats(user: dict = Depends(require_perm("view_live"))):
    stats = getattr(live_cam_trt, "LATEST_STATS", None)
    if stats is None:
        return JSONResponse({"ready": False})
    return JSONResponse(stats)


@app.get("/api/frame")
def get_frame(user: dict = Depends(require_perm("view_live"))):
    frame = getattr(live_cam_trt, "LATEST_FRAME", None)
    if frame is None:
        return Response(status_code=204)

    ok, jpg = cv2.imencode(".jpg", frame)
    if not ok:
        return Response(status_code=500)

    return Response(content=jpg.tobytes(), media_type="image/jpeg")


@app.get("/api/hist")
def get_hist(user: dict = Depends(require_perm("view_live"))):
    hist = getattr(live_cam_trt, "HIST_DATA", None)
    if not hist:
        return JSONResponse({"ready": False})

    if isinstance(hist, dict) and hist.get("ready") and hist.get("mode") == "diameter":
        dia = hist.get("diameter")
        if not (isinstance(dia, dict) and "bins" in dia and "counts" in dia):
            safe = dict(hist)
            safe["ready"] = False
            safe["error"] = "Histogram payload missing diameter bins/counts."
            return JSONResponse(safe)

    return JSONResponse(hist)


@app.get("/api/moisture")
def get_moisture(user: dict = Depends(require_perm("view_live"))):
    moist = getattr(live_cam_trt, "LATEST_MOISTURE", None)
    if not moist:
        return JSONResponse({"ready": False})
    return JSONResponse(moist)


@app.get("/api/devices/status")
def device_status(user: dict = Depends(require_perm("view_devices"))):
    health = getattr(live_cam_trt, "LATEST_HEALTH", None)
    if not health:
        # fallback: at least tell if stats exist
        stats = getattr(live_cam_trt, "LATEST_STATS", None)
        return {"ready": False, "timestamp": utc_now_iso(), "stats_ready": bool(stats)}
    return health


# ==========================================================
# Quality Rules: current + versioning + apply
# ==========================================================
def _get_latest_rules_version() -> Optional[sqlite3.Row]:
    with _DB_LOCK:
        conn = db_connect()
        row = conn.execute("""
            SELECT * FROM quality_rules_versions
            ORDER BY version DESC
            LIMIT 1
        """).fetchone()
        conn.close()
        return row


def _insert_rules_version(rules: dict, created_by: Optional[str], reason: Optional[str], apply_now: bool) -> dict:
    last = _get_latest_rules_version()
    next_ver = int(last["version"]) + 1 if last else 1

    rec = {
        "version": next_ver,
        "created_at": utc_now_iso(),
        "created_by_email": created_by,
        "reason": reason,
        "rules_json": json.dumps(rules, ensure_ascii=False),
        "applied": 1 if apply_now else 0,
        "applied_at": utc_now_iso() if apply_now else None,
    }

    with _DB_LOCK:
        conn = db_connect()
        conn.execute("""
            INSERT INTO quality_rules_versions
            (version, created_at, created_by_email, reason, rules_json, applied, applied_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            rec["version"], rec["created_at"], rec["created_by_email"], rec["reason"],
            rec["rules_json"], rec["applied"], rec["applied_at"]
        ))
        conn.commit()
        conn.close()

    return rec


@app.get("/api/rules/current")
def rules_current(user: dict = Depends(require_perm("view_live"))):
    row = _get_latest_rules_version()
    if not row:
        # no saved versions yet -> return current live_cam config if available
        cfg = live_cam_trt.get_runtime_config() if hasattr(live_cam_trt, "get_runtime_config") else {}
        return {"ok": True, "source": "runtime", "version": None, "rules": cfg}
    return {
        "ok": True,
        "source": "db",
        "version": int(row["version"]),
        "created_at": row["created_at"],
        "created_by_email": row["created_by_email"],
        "reason": row["reason"],
        "applied": bool(row["applied"]),
        "applied_at": row["applied_at"],
        "rules": json.loads(row["rules_json"]),
    }


@app.get("/api/rules/versions")
def rules_versions(
    user: dict = Depends(require_perm("edit_rules")),
    limit: int = Query(50, ge=1, le=500),
):
    with _DB_LOCK:
        conn = db_connect()
        rows = conn.execute("""
            SELECT version, created_at, created_by_email, reason, applied, applied_at
            FROM quality_rules_versions
            ORDER BY version DESC
            LIMIT ?
        """, (int(limit),)).fetchall()
        conn.close()

    out = []
    for r in rows:
        out.append({
            "version": int(r["version"]),
            "created_at": r["created_at"],
            "created_by_email": r["created_by_email"],
            "reason": r["reason"],
            "applied": bool(r["applied"]),
            "applied_at": r["applied_at"],
        })
    return {"ok": True, "versions": out}


@app.post("/api/rules/update")
def rules_update(
    payload: dict = Body(...),
    user: dict = Depends(require_perm("edit_rules")),
):
    """
    Creates a new rules version and applies immediately (MVP).
    """
    rules = payload.get("rules")
    reason = payload.get("reason")

    if not isinstance(rules, dict):
        raise HTTPException(status_code=400, detail="payload.rules must be an object")

    # Apply to runtime config (keep contract compatible with your existing /api/config)
    if hasattr(live_cam_trt, "update_runtime_config"):
        live_cam_trt.update_runtime_config(
            conf_thr=rules.get("conf_thr"),
            nms_iou=rules.get("nms_iou"),
            alarm_threshold_mm=rules.get("alarm_threshold_mm"),
            alarm_enabled=rules.get("alarm_enabled"),
            ref_diam_mm=rules.get("ref_diam_mm"),
            histogram_mode=rules.get("histogram_mode"),
            moisture_enabled=rules.get("moisture_enabled"),
            moisture_topk=rules.get("moisture_topk"),
            moisture_every_n_frames=rules.get("moisture_every_n_frames"),
        )

    rec = _insert_rules_version(rules=rules, created_by=user.get("email"), reason=reason, apply_now=True)
    audit_log("rules_updated", user.get("email"), {"version": rec["version"], "reason": reason, "rules": rules})
    return {"ok": True, "version": rec["version"], "applied": True, "rules": rules}


@app.post("/api/rules/rollback")
def rules_rollback(
    payload: dict = Body(...),
    user: dict = Depends(require_perm("edit_rules")),
):
    """
    Roll back by creating a NEW version equal to a previous version's rules, then applying.
    """
    target_version = int(payload.get("version", 0))
    if target_version <= 0:
        raise HTTPException(status_code=400, detail="payload.version must be > 0")

    with _DB_LOCK:
        conn = db_connect()
        row = conn.execute("SELECT * FROM quality_rules_versions WHERE version = ?", (target_version,)).fetchone()
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Version not found")

    rules = json.loads(row["rules_json"])

    if hasattr(live_cam_trt, "update_runtime_config"):
        live_cam_trt.update_runtime_config(
            conf_thr=rules.get("conf_thr"),
            nms_iou=rules.get("nms_iou"),
            alarm_threshold_mm=rules.get("alarm_threshold_mm"),
            alarm_enabled=rules.get("alarm_enabled"),
            ref_diam_mm=rules.get("ref_diam_mm"),
            histogram_mode=rules.get("histogram_mode"),
            moisture_enabled=rules.get("moisture_enabled"),
            moisture_topk=rules.get("moisture_topk"),
            moisture_every_n_frames=rules.get("moisture_every_n_frames"),
        )

    rec = _insert_rules_version(rules=rules, created_by=user.get("email"), reason=f"rollback_to_v{target_version}", apply_now=True)
    audit_log("rules_rollback", user.get("email"), {"to_version": target_version, "new_version": rec["version"]})
    return {"ok": True, "rolled_back_to": target_version, "new_version": rec["version"], "rules": rules}


# ==========================================================
# Legacy runtime configuration endpoints (kept for compatibility)
# (Now protected by edit_rules)
# ==========================================================
@app.get("/api/config")
def get_config(user: dict = Depends(require_perm("edit_rules"))):
    if hasattr(live_cam_trt, "get_runtime_config"):
        return live_cam_trt.get_runtime_config()
    return {}


@app.post("/api/config")
def set_config(cfg: dict = Body(...), user: dict = Depends(require_perm("edit_rules"))):
    """
    Backward-compatible config API:
    - applies to runtime immediately
    - ALSO creates a rules version entry (so changes are versioned)
    """
    if hasattr(live_cam_trt, "update_runtime_config"):
        live_cam_trt.update_runtime_config(
            conf_thr=cfg.get("conf_thr"),
            nms_iou=cfg.get("nms_iou"),
            alarm_threshold_mm=cfg.get("alarm_threshold_mm"),
            alarm_enabled=cfg.get("alarm_enabled"),
            ref_diam_mm=cfg.get("ref_diam_mm"),
            histogram_mode=cfg.get("histogram_mode"),
            moisture_enabled=cfg.get("moisture_enabled"),
            moisture_topk=cfg.get("moisture_topk"),
            moisture_every_n_frames=cfg.get("moisture_every_n_frames"),
        )

    # Version it
    rec = _insert_rules_version(rules=cfg, created_by=user.get("email"), reason="legacy_api_config_update", apply_now=True)
    audit_log("config_updated", user.get("email"), {"version": rec["version"], "cfg": cfg})

    return live_cam_trt.get_runtime_config() if hasattr(live_cam_trt, "get_runtime_config") else {}


# ==========================================================
# Events API: list + export CSV
# ==========================================================
@app.get("/api/events")
def events_list(
    user: dict = Depends(require_perm("view_events")),
    start_epoch: Optional[float] = Query(None, description="UTC epoch seconds"),
    end_epoch: Optional[float] = Query(None, description="UTC epoch seconds"),
    device_id: Optional[str] = Query(None),
    alarm_only: bool = Query(False),
    limit: int = Query(200, ge=1, le=5000),
):
    q = "SELECT * FROM events WHERE 1=1"
    params: List[Any] = []

    if start_epoch is not None:
        q += " AND ts_epoch >= ?"
        params.append(float(start_epoch))
    if end_epoch is not None:
        q += " AND ts_epoch <= ?"
        params.append(float(end_epoch))
    if device_id:
        q += " AND device_id = ?"
        params.append(device_id)
    if alarm_only:
        q += " AND alarm_active = 1"

    q += " ORDER BY ts_epoch DESC LIMIT ?"
    params.append(int(limit))

    with _DB_LOCK:
        conn = db_connect()
        rows = conn.execute(q, tuple(params)).fetchall()
        conn.close()

    out = []
    for r in rows:
        out.append({
            "id": int(r["id"]),
            "ts": r["ts"],
            "ts_epoch": float(r["ts_epoch"]),
            "device_id": r["device_id"],
            "alarm_active": bool(r["alarm_active"]),
            "alarm_max_d_mm": r["alarm_max_d_mm"],
            "mean_d": r["mean_d"],
            "std_d": r["std_d"],
            "units": r["units"],
            "moisture_mean_pred": r["moisture_mean_pred"],
        })

    return {"ok": True, "events": out}


@app.get("/api/events/export.csv")
def events_export_csv(
    user: dict = Depends(require_perm("export_events")),
    start_epoch: Optional[float] = Query(None),
    end_epoch: Optional[float] = Query(None),
    device_id: Optional[str] = Query(None),
    alarm_only: bool = Query(False),
    limit: int = Query(10000, ge=1, le=200000),
):
    q = "SELECT * FROM events WHERE 1=1"
    params: List[Any] = []

    if start_epoch is not None:
        q += " AND ts_epoch >= ?"
        params.append(float(start_epoch))
    if end_epoch is not None:
        q += " AND ts_epoch <= ?"
        params.append(float(end_epoch))
    if device_id:
        q += " AND device_id = ?"
        params.append(device_id)
    if alarm_only:
        q += " AND alarm_active = 1"

    q += " ORDER BY ts_epoch ASC LIMIT ?"
    params.append(int(limit))

    with _DB_LOCK:
        conn = db_connect()
        rows = conn.execute(q, tuple(params)).fetchall()
        conn.close()

    audit_log("events_export_csv", user.get("email"), {
        "start_epoch": start_epoch, "end_epoch": end_epoch, "device_id": device_id,
        "alarm_only": alarm_only, "limit": limit, "rows": len(rows)
    })

    def gen():
        header = [
            "ts", "ts_epoch", "device_id",
            "alarm_active", "alarm_max_d_mm",
            "mean_d", "std_d", "units",
            "moisture_mean_pred",
        ]
        yield (",".join(header) + "\n").encode("utf-8")

        for r in rows:
            line = [
                str(r["ts"] or ""),
                str(r["ts_epoch"] or ""),
                str(r["device_id"] or ""),
                str(int(r["alarm_active"] or 0)),
                str(r["alarm_max_d_mm"] if r["alarm_max_d_mm"] is not None else ""),
                str(r["mean_d"] if r["mean_d"] is not None else ""),
                str(r["std_d"] if r["std_d"] is not None else ""),
                str(r["units"] or ""),
                str(r["moisture_mean_pred"] or ""),
            ]
            yield (",".join(line) + "\n").encode("utf-8")

    return StreamingResponse(
        gen(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="events_export.csv"'},
    )


# ==========================================================
# Audit API
# ==========================================================
@app.get("/api/audit")
def audit_list(
    user: dict = Depends(require_perm("view_audit")),
    limit: int = Query(200, ge=1, le=2000),
):
    with _DB_LOCK:
        conn = db_connect()
        rows = conn.execute("""
            SELECT id, ts, user_email, action, details_json
            FROM audit_log
            ORDER BY id DESC
            LIMIT ?
        """, (int(limit),)).fetchall()
        conn.close()

    out = []
    for r in rows:
        try:
            details = json.loads(r["details_json"]) if r["details_json"] else {}
        except Exception:
            details = {}
        out.append({
            "id": int(r["id"]),
            "ts": r["ts"],
            "user_email": r["user_email"],
            "action": r["action"],
            "details": details,
        })

    return {"ok": True, "items": out}


# ==========================================================
# Serve frontend (static)
# ==========================================================
if os.path.isdir(static_dir):
    app.mount(
        "/",
        StaticFiles(directory=static_dir, html=True),
        name="static",
    )


# ==========================================================
# Optional CLI entry
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.backend_app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # IMPORTANT for Jetson
    )
