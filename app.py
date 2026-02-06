# app.py
"""
Advanced Detection Service backend (single-file).

Features:
- Flask + Flask-SocketIO web app
- Built-in scheduler with per-user priority and per-user concurrency limits
- Multi-GPU allocation with CUDA_VISIBLE_DEVICES pinning per subprocess
- FIFO within same priority; scheduler will pick next runnable job (no head-of-line blocking)
- Safe ZIP handling and clear error reporting in DB + UI
- User auth (register/login) + admin pages & metrics
- Live socket updates: job_update, job_progress, system_metrics

Run:
    python3 app.py

Dependencies:
    pip install flask flask-socketio eventlet werkzeug psutil pynvml onnxruntime tqdm
"""
import os
import zipfile
import sqlite3
import uuid
import shutil
import json
import time
import threading
import subprocess
import datetime
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, session, abort
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import psutil

# Optional NVML for GPU enumeration
try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except Exception:
    PYNVML_AVAILABLE = False

# -------------------------
# Configuration & paths
# -------------------------
APP_ROOT = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
OUTPUT_FOLDER = os.path.join(APP_ROOT, "outputs")
DB_FILE = os.path.join(APP_ROOT, "history.db")
MODELS_DIR = os.path.join(APP_ROOT, "models")
LABELS_DIR = os.path.join(APP_ROOT, "labels")
DETECTION_SCRIPT = os.path.join(APP_ROOT, "detection_script.py")  # existing detection script

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
ALLOWED_EXT = ALLOWED_IMAGE_EXT | {".zip"}

# App
app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET", "replace-this-with-a-secret")
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")


# Scheduler parameters
DEFAULT_MAX_CONCURRENT_PER_USER = 2
CPU_SLOTS = 1  # number of simultaneous CPU-only tasks allowed

# -------------------------
# Database helpers & schema
# -------------------------
def init_db():
    # Create DB if missing and ensure schema. Use safe interpolation for DDL defaults.
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # users table with priority and max_concurrent default inserted as literal
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            is_admin INTEGER DEFAULT 0,
            priority INTEGER DEFAULT 0,
            max_concurrent INTEGER DEFAULT {DEFAULT_MAX_CONCURRENT_PER_USER}
        )
    """)

    # jobs table
    c.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            original_filename TEXT,
            stored_path TEXT,
            model TEXT,
            labels TEXT,
            threshold REAL,
            pad INTEGER,
            device_request TEXT,
            gpu_allocated TEXT,
            status TEXT,
            result_zip TEXT,
            log TEXT,
            error_msg TEXT,
            total_files INTEGER DEFAULT 0,
            processed INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            finished_at TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def db_execute(query, args=(), fetch=False):
    conn = sqlite3.connect(DB_FILE, timeout=30)
    c = conn.cursor()
    c.execute(query, args)
    if fetch:
        rows = c.fetchall()
        conn.close()
        return rows
    conn.commit()
    conn.close()

# initialize DB (create tables)
init_db()

# -------------------------
# Auth helpers
# -------------------------
def current_user():
    """
    Return a dict with user info or None.
    Templates expect a callable current_user(), so we inject the function into templates.
    """
    uid = session.get("user_id")
    if not uid:
        return None
    rows = db_execute("SELECT id, username, is_admin, priority, max_concurrent FROM users WHERE id=?", (uid,), fetch=True)
    if not rows:
        return None
    r = rows[0]
    return {"id": r[0], "username": r[1], "is_admin": bool(r[2]), "priority": int(r[3] or 0), "max_concurrent": int(r[4] or DEFAULT_MAX_CONCURRENT_PER_USER)}

# Inject current_user callable into Jinja templates so templates can call current_user()
@app.context_processor
def inject_current_user():
    return {"current_user": current_user}

def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not current_user():
            return redirect(url_for("login", next=request.path))
        return f(*a, **kw)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        u = current_user()
        if not u or not u.get("is_admin"):
            abort(403)
        return f(*a, **kw)
    return wrapper

# -------------------------
# GPU manager
# -------------------------
class GPUManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.gpus = []  # dicts: {index, name, free, task_id}
        self._probe()

    def _probe(self):
        with self.lock:
            self.gpus = []
            if not PYNVML_AVAILABLE:
                return
            try:
                count = pynvml.nvmlDeviceGetCount()
                for i in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode(errors="ignore")
                    self.gpus.append({"index": i, "name": name, "free": True, "task_id": None})
            except Exception:
                self.gpus = []

    def get_status(self):
        with self.lock:
            return [dict(g) for g in self.gpus]

    def acquire_any(self, task_id):
        with self.lock:
            for g in self.gpus:
                if g["free"]:
                    g["free"] = False
                    g["task_id"] = task_id
                    return g["index"]
        return None

    def release(self, index):
        with self.lock:
            for g in self.gpus:
                if g["index"] == index:
                    g["free"] = True
                    g["task_id"] = None

gpu_manager = GPUManager()

# CPU slot management
cpu_lock = threading.Lock()
cpu_in_use = 0
def acquire_cpu_slot():
    global cpu_in_use
    with cpu_lock:
        if cpu_in_use < CPU_SLOTS:
            cpu_in_use += 1
            return True
        return False

def release_cpu_slot():
    global cpu_in_use
    with cpu_lock:
        if cpu_in_use > 0:
            cpu_in_use -= 1

# -------------------------
# Scheduler (priority + per-user concurrency)
# -------------------------
class Scheduler:
    def __init__(self):
        self.lock = threading.Lock()
        self.queue = []  # list of dict entries: {job_id, created_at, user_id, priority, seq}
        self.running_counts = {}
        self.seq = 0

    def enqueue(self, job_id):
        row = db_execute("SELECT created_at, user_id FROM jobs WHERE id=?", (job_id,), fetch=True)
        if not row:
            return
        created_at, user_id = row[0]
        user_row = db_execute("SELECT priority FROM users WHERE id=?", (user_id,), fetch=True)
        priority = int(user_row[0][0] or 0) if user_row else 0
        with self.lock:
            self.seq += 1
            self.queue.append({"job_id": job_id, "created_at": created_at, "user_id": user_id, "priority": priority, "seq": self.seq})

    def remove(self, job_id):
        with self.lock:
            self.queue = [q for q in self.queue if q["job_id"] != job_id]

    def decrement_running(self, user_id):
        with self.lock:
            if user_id in self.running_counts:
                self.running_counts[user_id] = max(0, self.running_counts[user_id] - 1)

    def pop_next_runnable(self):
        with self.lock:
            if not self.queue:
                return None
            candidates = []
            gpu_free = any(g["free"] for g in gpu_manager.get_status())
            cpu_available = (cpu_in_use < CPU_SLOTS)
            for entry in self.queue:
                row = db_execute("SELECT device_request, user_id FROM jobs WHERE id=?", (entry["job_id"],), fetch=True)
                if not row:
                    continue
                device_request, user_id = row[0]
                user_row = db_execute("SELECT max_concurrent FROM users WHERE id=?", (user_id,), fetch=True)
                max_conc = int(user_row[0][0]) if user_row and user_row[0][0] is not None else DEFAULT_MAX_CONCURRENT_PER_USER
                running = self.running_counts.get(user_id, 0)
                if running >= max_conc:
                    continue
                wants_gpu = device_request in ("auto","cuda","tensorrt")
                gstat = gpu_manager.get_status()
                has_gpu_hardware = len(gstat) > 0
                if wants_gpu and has_gpu_hardware:
                    if not gpu_free:
                        continue
                else:
                    if not cpu_available:
                        continue
                candidates.append(entry)
            if not candidates:
                return None
            maxp = max(c["priority"] for c in candidates)
            top_priority_candidates = [c for c in candidates if c["priority"] == maxp]
            top_priority_candidates.sort(key=lambda x: x["seq"])
            chosen = top_priority_candidates[0]
            self.queue = [q for q in self.queue if q["job_id"] != chosen["job_id"]]
            self.running_counts[chosen["user_id"]] = self.running_counts.get(chosen["user_id"], 0) + 1
            return chosen["job_id"]

scheduler = Scheduler()

# -------------------------
# Worker loop
# -------------------------
def safe_extract_zip(zip_path, extract_to):
    files = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                norm = os.path.normpath(member)
                if norm.startswith(".."):
                    continue
            zf.extractall(extract_to)
        for root, _, filenames in os.walk(extract_to):
            for fn in sorted(filenames):
                ext = os.path.splitext(fn)[1].lower()
                if ext in ALLOWED_IMAGE_EXT:
                    files.append(os.path.join(root, fn))
        return files
    except zipfile.BadZipFile:
        raise RuntimeError("Invalid ZIP archive.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract ZIP: {e}")

def update_job(job_id, **fields):
    if not fields:
        return
    parts = ", ".join([f"{k}=?" for k in fields.keys()])
    args = list(fields.values()) + [job_id]
    db_execute(f"UPDATE jobs SET {parts} WHERE id=?", tuple(args))

def run_detection_process(job_id, allocated_gpu_index):
    row = db_execute("SELECT stored_path, model, labels, threshold, pad FROM jobs WHERE id=?", (job_id,), fetch=True)
    if not row:
        raise RuntimeError("Job record missing")
    stored_path, model_path, labels_path, threshold, pad = row[0]
    out_dir = os.path.join(OUTPUT_FOLDER, job_id)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tmp_extract = None
    work_input = stored_path
    if stored_path.lower().endswith(".zip"):
        tmp_extract = os.path.join(OUTPUT_FOLDER, f"{job_id}_input")
        os.makedirs(tmp_extract, exist_ok=True)
        files = safe_extract_zip(stored_path, tmp_extract)
        if not files:
            raise RuntimeError("No images found in ZIP.")
        work_input = tmp_extract
        update_job(job_id, total_files=len(files))

    else:
        if os.path.isdir(stored_path):
            count = 0
            for root, _, filenames in os.walk(stored_path):
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in ALLOWED_IMAGE_EXT:
                        count += 1
            if count == 0:
                raise RuntimeError("No images in provided input folder.")
            update_job(job_id, total_files=count)
        elif os.path.isfile(stored_path):
            if os.path.splitext(stored_path)[1].lower() not in ALLOWED_IMAGE_EXT:
                raise RuntimeError("Uploaded file unsupported.")
            update_job(job_id, total_files=1)
        else:
            raise RuntimeError("Input file not found on server.")

    device_arg = "cpu" if allocated_gpu_index is None else "cuda"
    cmd = [
        "python3", DETECTION_SCRIPT,
        "--model", model_path,
        "--labels", labels_path,
        "--input", work_input,
        "--output", out_dir,
        "--threshold", str(threshold),
        "--pad", str(pad),
        "--cuda", device_arg
    ]

    env = os.environ.copy()
    if allocated_gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(allocated_gpu_index)
    log_path = os.path.join(out_dir, "worker_log.txt")
    with open(log_path, "wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        progress_file = os.path.join(out_dir, "progress.json")
        last_meta = {"processed": 0, "total": 0}
        try:
            while True:
                ret = proc.poll()
                if os.path.exists(progress_file):
                    try:
                        with open(progress_file, "r") as pf:
                            data = json.load(pf)
                            processed = int(data.get("processed", 0))
                            total = int(data.get("total", 0))
                            percent = int(processed * 100 / total) if total else 0
                            meta = {"processed": processed, "total": total, "percent": percent}
                            if meta != last_meta:
                                last_meta = meta
                                update_job(job_id, processed=processed)
                                socketio.emit("job_progress", {"job_id": job_id, "progress": meta}, room=f"job_{job_id}")
                    except Exception:
                        pass
                st = db_execute("SELECT status FROM jobs WHERE id=?", (job_id,), fetch=True)[0][0]
                if st == "cancelled":
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    raise RuntimeError("Job cancelled by user")
                if ret is not None:
                    break
                time.sleep(0.8)
            if proc.returncode != 0:
                raise RuntimeError(f"Detection script failed (exit {proc.returncode}). See worker log.")
        finally:
            try:
                summary = os.path.join(out_dir, "detection_summary.txt")
                log_to_db = summary if os.path.exists(summary) else log_path
                update_job(job_id, log=log_to_db)
            except Exception:
                pass
            if tmp_extract and os.path.exists(tmp_extract):
                try:
                    shutil.rmtree(tmp_extract)
                except Exception:
                    pass

    result_zip = os.path.join(OUTPUT_FOLDER, f"{job_id}.zip")
    base = result_zip.replace(".zip", "")
    shutil.make_archive(base, "zip", out_dir)
    return result_zip, log_path

def worker_mainloop():
    while True:
        try:
            job_id = scheduler.pop_next_runnable()
            if not job_id:
                time.sleep(1.2)
                continue
            rows = db_execute("SELECT device_request, user_id FROM jobs WHERE id=?", (job_id,), fetch=True)
            if not rows:
                continue
            device_request, user_id = rows[0]
            allocated_gpu = None
            wants_gpu = device_request in ("auto","cuda","tensorrt")
            gstat = gpu_manager.get_status()
            has_gpu = len(gstat) > 0
            if wants_gpu and has_gpu:
                idx = gpu_manager.acquire_any(job_id)
                if idx is None:
                    scheduler.enqueue(job_id)
                    scheduler.decrement_running(user_id)
                    time.sleep(0.8)
                    continue
                allocated_gpu = idx
            else:
                if not acquire_cpu_slot():
                    scheduler.enqueue(job_id)
                    scheduler.decrement_running(user_id)
                    time.sleep(0.8)
                    continue

            update_job(job_id, status="running", gpu_allocated=str(allocated_gpu) if allocated_gpu is not None else None, started_at=datetime.datetime.utcnow().isoformat())
            socketio.emit("job_update", {"job_id": job_id, "status": "running"}, room=f"job_{job_id}")

            try:
                result_zip, logpath = run_detection_process(job_id, allocated_gpu)
                update_job(job_id, status="completed", result_zip=result_zip, finished_at=datetime.datetime.utcnow().isoformat())
                socketio.emit("job_update", {"job_id": job_id, "status": "completed"}, room=f"job_{job_id}")
            except Exception as e:
                err = str(e)
                update_job(job_id, status="failed", error_msg=err)
                socketio.emit("job_update", {"job_id": job_id, "status": "failed", "error": err}, room=f"job_{job_id}")
            finally:
                if allocated_gpu is not None:
                    gpu_manager.release(allocated_gpu)
                else:
                    release_cpu_slot()
                scheduler.decrement_running(user_id)
        except Exception as e:
            print("[worker] unexpected error:", e)
            time.sleep(1.0)

worker_thread = threading.Thread(target=worker_mainloop, daemon=True)
worker_thread.start()

# -------------------------
# Flask endpoints
# -------------------------
@app.route("/")
def index():
    u = current_user()
    models = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".onnx")]
    labels = [f for f in os.listdir(LABELS_DIR) if f.lower().endswith(".txt")]
    return render_template("upload.html", models=models, labels=labels, model_label_map={})

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Provide username and password.")
            return redirect(url_for("register"))
        hashed = generate_password_hash(password)
        try:
            db_execute("INSERT INTO users (username, password) VALUES (?,?)", (username, hashed))
            flash("Registered. Please log in.")
            return redirect(url_for("login"))
        except Exception:
            flash("Username already taken.")
            return redirect(url_for("register"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        rows = db_execute("SELECT id, password FROM users WHERE username=?", (username,), fetch=True)
        if not rows:
            flash("Invalid credentials.")
            return redirect(url_for("login"))
        uid, hashed = rows[0]
        if check_password_hash(hashed, password):
            session["user_id"] = uid
            flash("Logged in.")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials.")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out.")
    return redirect(url_for("index"))

@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    user = current_user()
    job_id = str(uuid.uuid4())
    stored_name = f"{job_id}_{filename}"
    saved_path = os.path.join(UPLOAD_FOLDER, stored_name)
    file.save(saved_path)

    if ext == ".zip":
        try:
            with zipfile.ZipFile(saved_path, "r") as _:
                pass
        except zipfile.BadZipFile:
            try:
                os.remove(saved_path)
            except Exception:
                pass
            flash("Uploaded ZIP is invalid.")
            return redirect(url_for("index"))

    model = request.form.get("model")
    labels = request.form.get("labels")
    threshold = float(request.form.get("threshold", 0.25))
    pad = int(request.form.get("pad", 10))
    device = request.form.get("cuda", "auto")

    model_path = os.path.join(MODELS_DIR, model) if model else None
    labels_path = os.path.join(LABELS_DIR, labels) if labels else None
    if not model_path or not os.path.exists(model_path) or not labels_path or not os.path.exists(labels_path):
        flash("Model or labels missing on server.")
        return redirect(url_for("index"))

    db_execute("""
        INSERT INTO jobs (id, user_id, original_filename, stored_path, model, labels, threshold, pad, device_request, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (job_id, user["id"], filename, saved_path, model_path, labels_path, threshold, pad, device, "queued"))

    scheduler.enqueue(job_id)
    flash("Job queued")
    return redirect(url_for("status_page", job_id=job_id))

@app.route("/status/<job_id>")
@login_required
def status_page(job_id):
    u = current_user()
    rows = db_execute("SELECT user_id FROM jobs WHERE id=?", (job_id,), fetch=True)
    if not rows:
        abort(404)
    owner_id = rows[0][0]
    if u["id"] != owner_id and not u["is_admin"]:
        abort(403)
    return render_template("status.html", job_id=job_id)

@app.route("/api/status/<job_id>")
@login_required
def api_status(job_id):
    u = current_user()
    rows = db_execute("SELECT id, user_id, original_filename, model, labels, threshold, pad, device_request, gpu_allocated, status, result_zip, log, error_msg, total_files, processed, created_at, started_at, finished_at FROM jobs WHERE id=?", (job_id,), fetch=True)
    if not rows:
        return jsonify({"error":"not found"}), 404
    job = dict(zip(["id","user_id","original_filename","model","labels","threshold","pad","device_request","gpu_allocated","status","result_zip","log","error_msg","total_files","processed","created_at","started_at","finished_at"], rows[0]))
    if u["id"] != job["user_id"] and not u["is_admin"]:
        return jsonify({"error":"forbidden"}), 403
    return jsonify(job)

@app.route("/history")
@login_required
def history_page():
    return render_template("history.html")

@app.route("/api/history")
@login_required
def api_history():
    u = current_user()
    if u["is_admin"]:
        rows = db_execute("SELECT id, original_filename, model, device_request, gpu_allocated, status, total_files, processed, created_at, user_id FROM jobs ORDER BY created_at DESC", fetch=True)
    else:
        rows = db_execute("SELECT id, original_filename, model, device_request, gpu_allocated, status, total_files, processed, created_at, user_id FROM jobs WHERE user_id=? ORDER BY created_at DESC", (u["id"],), fetch=True)
    jobs = []
    for r in rows:
        jobs.append(dict(zip(["id","original_filename","model","device_request","gpu_allocated","status","total_files","processed","created_at","user_id"], r)))
    return jsonify({"jobs": jobs})

@app.route("/download/<job_id>")
@login_required
def download(job_id):
    u = current_user()
    rows = db_execute("SELECT user_id, result_zip FROM jobs WHERE id=?", (job_id,), fetch=True)
    if not rows:
        return "Not found", 404
    owner_id, zip_path = rows[0]
    if u["id"] != owner_id and not u["is_admin"]:
        return "Forbidden", 403
    if not zip_path or not os.path.exists(zip_path):
        return "File not ready", 404
    return send_file(zip_path, as_attachment=True)

@app.route("/cancel/<job_id>", methods=["POST"])
@login_required
def cancel_job(job_id):
    u = current_user()
    rows = db_execute("SELECT user_id, status FROM jobs WHERE id=?", (job_id,), fetch=True)
    if not rows:
        return jsonify({"error":"not found"}), 404
    owner_id, status = rows[0]
    if u["id"] != owner_id and not u["is_admin"]:
        return jsonify({"error":"forbidden"}), 403
    if status in ("completed","failed","cancelled"):
        return jsonify({"ok": False, "reason": "cannot cancel finished job"})
    update_job(job_id, status="cancelled")
    scheduler.remove(job_id)
    socketio.emit("job_update", {"job_id": job_id, "status": "cancelled"}, room=f"job_{job_id}")
    return jsonify({"ok": True})

# Admin
@app.route("/admin")
@admin_required
def admin_page():
    return render_template("admin.html")

@app.route("/api/admin/metrics")
@admin_required
def api_admin_metrics():
    cpu = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    ram_pct = mem.percent
    total_today = db_execute("SELECT COUNT(*) FROM jobs WHERE DATE(created_at)=DATE('now')", fetch=True)[0][0]
    rows = db_execute("SELECT username, COUNT(jobs.id) FROM users LEFT JOIN jobs ON users.id=jobs.user_id AND DATE(jobs.created_at)=DATE('now') GROUP BY users.id", fetch=True)
    per_user = [{"username": r[0], "count": r[1]} for r in rows]
    gstat = gpu_manager.get_status()
    return jsonify({"cpu_percent": cpu, "ram_percent": ram_pct, "total_jobs_today": total_today, "per_user": per_user, "gpus": gstat})

@app.route("/api/log/<job_id>")
@login_required
def api_log(job_id):
    u = current_user()
    rows = db_execute("SELECT user_id, log FROM jobs WHERE id=?", (job_id,), fetch=True)
    if not rows:
        return "Not found", 404
    owner_id, logpath = rows[0]
    if u["id"] != owner_id and not u["is_admin"]:
        return "Forbidden", 403
    if not logpath:
        return "No log", 404
    if os.path.exists(logpath):
        try:
            with open(logpath, "r", errors="ignore") as f:
                return f.read(), 200, {"Content-Type":"text/plain; charset=utf-8"}
        except Exception:
            return "Cannot read log", 500
    return "Log missing", 404

# -------------------------
# Socket.IO events
# -------------------------
@socketio.on("join_job")
def on_join_job(data):
    job_id = data.get("job_id")
    join_room(f"job_{job_id}")
    rows = db_execute("SELECT status, processed, total_files FROM jobs WHERE id=?", (job_id,), fetch=True)
    if rows:
        status, processed, total = rows[0]
        emit("job_progress", {"job_id": job_id, "progress": {"processed": processed or 0, "total": total or 0, "percent": int(processed*100/total) if total else 0}})
        emit("job_update", {"job_id": job_id, "status": status})

@socketio.on("leave_job")
def on_leave_job(data):
    job_id = data.get("job_id")
    leave_room(f"job_{job_id}")

# Metrics broadcaster
def metrics_broadcaster():
    while True:
        try:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            gstat = gpu_manager.get_status()
            socketio.emit("system_metrics", {"cpu": cpu, "ram": mem.percent, "gpus": gstat})
        except Exception:
            pass
        time.sleep(2)

metrics_thread = threading.Thread(target=metrics_broadcaster, daemon=True)
metrics_thread.start()

# Ensure an admin exists (safe: will not duplicate if users table non-empty)
def ensure_admin():
    rows = db_execute("SELECT COUNT(*) FROM users", fetch=True)[0][0]
    if rows == 0:
        print("Creating default admin: username=admin password=admin")
        db_execute("INSERT INTO users (username, password, is_admin, priority, max_concurrent) VALUES (?,?,?,?,?)", ("admin", generate_password_hash("admin"), 1, 100, 10))
ensure_admin()

# -------------------------
# Start server
# -------------------------
if __name__ == "__main__":
    print("Starting Detection Service (single-command). Visit http://0.0.0.0:5002")
    
    
    socketio.run(app, host="0.0.0.0", port=5002)
