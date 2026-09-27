from flask import Flask, render_template, request, redirect, jsonify, g
from concurrent.futures import ThreadPoolExecutor
import sqlite3, requests
import socket
import urllib3.util.connection as urllib3_cn

# Ép Python chỉ phân giải IPv4 khi gọi API bên ngoài — một số mạng Windows có
# tuyến IPv6 bị "đen" (gói tin gửi đi nhưng không bao giờ có phản hồi), khiến
# `requests` treo tới hết timeout dù trình duyệt (dùng Happy Eyeballs) vẫn
# load bình thường. Xem thêm ghi chú tương tự trong upgrade_db.py.
def _force_ipv4():
    return socket.AF_INET
urllib3_cn.allowed_gai_family = _force_ipv4
import unicodedata, re
import eng_to_ipa
import os
import pandas as pd

import matplotlib
matplotlib.use('Agg') # Cấu hình để Flask không bị crash khi vẽ biểu đồ ngầm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

def clean_input(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', text).strip()

def remove_accents(text):
    text = text.lower()
    text = re.sub(r'[đĐ]', 'd', text)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.strip()

DB_PATH = "vocab.db"

def init_db():
    """Tạo bảng + vá schema cũ (migrate). CHỈ chạy 1 lần lúc khởi động app,
    không chạy lại ở mỗi request như code cũ."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vocab (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            en TEXT, vi TEXT, ipa TEXT DEFAULT '',
            definition TEXT DEFAULT '', example TEXT DEFAULT '',
            correct INTEGER DEFAULT 0, wrong INTEGER DEFAULT 0,
            is_graduated INTEGER DEFAULT 0,
            synonyms TEXT DEFAULT '',
            word_type TEXT DEFAULT '',
            interval INTEGER DEFAULT 0,
            ef REAL DEFAULT 2.5,
            repetitions INTEGER DEFAULT 0,
            next_review TEXT DEFAULT NULL
        )
    """)
    # Bọc lót cột tự động tránh lỗi DB cũ (idempotent, nhưng giờ chỉ chạy 1 lần)
    for col in ['ipa', 'definition', 'example', 'synonyms', 'word_type']:
        try: conn.execute(f"ALTER TABLE vocab ADD COLUMN {col} TEXT DEFAULT ''")
        except sqlite3.OperationalError: pass
    try: conn.execute("ALTER TABLE vocab ADD COLUMN is_graduated INTEGER DEFAULT 0")
    except sqlite3.OperationalError: pass

    # Tự động vá thêm các cột lưu dữ liệu SM-2
    try: conn.execute("ALTER TABLE vocab ADD COLUMN interval INTEGER DEFAULT 0")
    except sqlite3.OperationalError: pass
    try: conn.execute("ALTER TABLE vocab ADD COLUMN ef REAL DEFAULT 2.5")
    except sqlite3.OperationalError: pass
    try: conn.execute("ALTER TABLE vocab ADD COLUMN repetitions INTEGER DEFAULT 0")
    except sqlite3.OperationalError: pass

    try: conn.execute("ALTER TABLE vocab ADD COLUMN next_review TEXT DEFAULT NULL")
    except sqlite3.OperationalError: pass

    # Bảng Tracking hành vi học tập
    conn.execute("""
        CREATE TABLE IF NOT EXISTS learning_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vocab_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_correct INTEGER,
            response_time_seconds REAL,
            user_input TEXT
        )
    """)
    try: conn.execute("ALTER TABLE learning_logs ADD COLUMN user_input TEXT")
    except sqlite3.OperationalError: pass

    conn.commit()
    conn.close()


def get_db():
    """Trả về connection SQLite cho request hiện tại (lưu trong flask.g),
    KHÔNG chạy CREATE/ALTER TABLE nữa — việc đó đã do init_db() lo.
    Connection sẽ tự đóng khi request kết thúc, xem close_db() bên dưới."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    """Đóng connection sau mỗi request để tránh rò rỉ (trước đây connection
    không hề được đóng ở bất kỳ route nào)."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


init_db()  # Tạo bảng / migrate schema đúng 1 lần khi app khởi động (không phải mỗi request)

# ============================================================
# Tra loại từ (danh từ/động từ/tính từ...) HOÀN TOÀN OFFLINE bằng
# NLTK WordNet, KHÔNG phụ thuộc vào dictionaryapi.dev — API này có
# thể bị chặn ngầm bởi antivirus/firewall/ISP trên một số máy, rất
# khó chẩn đoán. WordNet chỉ cần tải 1 lần (~10MB) rồi chạy mãi mãi
# không cần mạng nữa, nên loại từ luôn hiện được dù mạng có vấn đề.
# ============================================================
import nltk
from nltk.corpus import wordnet as wn
try:
    wn.ensure_loaded()
except LookupError:
    nltk.download('wordnet')

WORDNET_POS_VI = {'n': 'danh từ', 'v': 'động từ', 'a': 'tính từ', 's': 'tính từ', 'r': 'trạng từ'}

FUNCTION_WORDS_VI = {
    'in': 'giới từ', 'on': 'giới từ', 'at': 'giới từ', 'by': 'giới từ', 'for': 'giới từ',
    'with': 'giới từ', 'about': 'giới từ', 'against': 'giới từ', 'between': 'giới từ',
    'into': 'giới từ', 'through': 'giới từ', 'during': 'giới từ', 'before': 'giới từ',
    'after': 'giới từ', 'above': 'giới từ', 'below': 'giới từ', 'to': 'giới từ',
    'from': 'giới từ', 'up': 'giới từ', 'down': 'giới từ', 'of': 'giới từ', 'off': 'giới từ',
    'over': 'giới từ', 'under': 'giới từ', 'since': 'giới từ', 'without': 'giới từ',
    'within': 'giới từ', 'along': 'giới từ', 'across': 'giới từ', 'behind': 'giới từ',
    'beyond': 'giới từ', 'near': 'giới từ', 'upon': 'giới từ', 'except': 'giới từ',
    'among': 'giới từ', 'toward': 'giới từ', 'towards': 'giới từ', 'onto': 'giới từ',
    'despite': 'giới từ', 'regarding': 'giới từ', 'concerning': 'giới từ',
    'and': 'liên từ', 'but': 'liên từ', 'or': 'liên từ', 'nor': 'liên từ', 'yet': 'liên từ',
    'so': 'liên từ', 'because': 'liên từ', 'although': 'liên từ', 'though': 'liên từ',
    'while': 'liên từ', 'if': 'liên từ', 'unless': 'liên từ', 'whereas': 'liên từ',
    'however': 'liên từ', 'therefore': 'liên từ', 'moreover': 'liên từ',
    'furthermore': 'liên từ', 'meanwhile': 'liên từ', 'whether': 'liên từ',
    'until': 'liên từ', 'than': 'liên từ',
    'i': 'đại từ', 'you': 'đại từ', 'he': 'đại từ', 'she': 'đại từ', 'it': 'đại từ',
    'we': 'đại từ', 'they': 'đại từ', 'me': 'đại từ', 'him': 'đại từ', 'her': 'đại từ',
    'us': 'đại từ', 'them': 'đại từ', 'this': 'đại từ', 'that': 'đại từ',
    'these': 'đại từ', 'those': 'đại từ', 'who': 'đại từ', 'whom': 'đại từ',
    'whose': 'đại từ', 'which': 'đại từ', 'what': 'đại từ', 'myself': 'đại từ',
    'yourself': 'đại từ', 'himself': 'đại từ', 'herself': 'đại từ', 'itself': 'đại từ',
    'ourselves': 'đại từ', 'themselves': 'đại từ', 'someone': 'đại từ',
    'anyone': 'đại từ', 'everyone': 'đại từ', 'nobody': 'đại từ',
    'something': 'đại từ', 'anything': 'đại từ', 'everything': 'đại từ', 'nothing': 'đại từ',
    'a': 'mạo từ', 'an': 'mạo từ', 'the': 'mạo từ',
    'my': 'từ hạn định', 'your': 'từ hạn định', 'his': 'từ hạn định',
    'its': 'từ hạn định', 'our': 'từ hạn định', 'their': 'từ hạn định',
    'some': 'từ hạn định', 'any': 'từ hạn định', 'no': 'từ hạn định',
    'every': 'từ hạn định', 'each': 'từ hạn định', 'either': 'từ hạn định',
    'neither': 'từ hạn định', 'much': 'từ hạn định', 'many': 'từ hạn định',
    'few': 'từ hạn định', 'little': 'từ hạn định', 'several': 'từ hạn định',
    'all': 'từ hạn định', 'both': 'từ hạn định',
    'oh': 'thán từ', 'wow': 'thán từ', 'hey': 'thán từ', 'alas': 'thán từ',
    'ouch': 'thán từ', 'oops': 'thán từ', 'hurray': 'thán từ', 'ah': 'thán từ',
    'um': 'thán từ', 'uh': 'thán từ', 'well': 'thán từ',
}

def get_word_type_offline(word):
    """Tra loại từ hoàn toàn offline (không gọi mạng): ưu tiên từ chức năng
    (danh sách cố định), sau đó tra WordNet cho từ nội dung."""
    w = word.strip().lower()
    if w in FUNCTION_WORDS_VI:
        return FUNCTION_WORDS_VI[w]
    synsets = wn.synsets(w)
    if synsets:
        return WORDNET_POS_VI.get(synsets[0].pos(), "")
    return ""

def get_word_details(word):
    word_clean = word.strip().lower()
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word_clean}"
    ipa, definition, example, synonyms = "/.../", "No description available.", "", ""
    word_type = get_word_type_offline(word_clean)  # offline, luôn chạy được dù API bên dưới có lỗi mạng
    try:
        res = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                main_data = data[0]
                if "phonetic" in main_data and main_data["phonetic"]: ipa = main_data["phonetic"]
                else:
                    for p in main_data.get("phonetics", []):
                        if "text" in p and p["text"]: 
                            ipa = p["text"]
                            break
                
                syn_list = []
                for meaning in main_data.get("meanings", []):
                    if meaning.get("synonyms"): syn_list.extend(meaning["synonyms"])
                    for df in meaning.get("definitions", []):
                        if "definition" in df and (definition == "No description available." or not definition):
                            definition = df["definition"]
                        if "example" in df and not example:
                            example = df["example"]
                        if df.get("synonyms"): syn_list.extend(df["synonyms"])
                
                if syn_list:
                    synonyms = ", ".join(list(set(syn_list))[:5])
                    
    except Exception: pass
    
    if ipa == "/.../":
        try:
            fallback_ipa = eng_to_ipa.convert(word_clean)
            if fallback_ipa and "*" not in fallback_ipa: ipa = f"/{fallback_ipa}/"
        except Exception: pass
    return ipa, definition, example, synonyms, word_type

def translate(word):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    # Nguồn 1: Google Translate (không chính thức) - thử tối đa 3 lần
    url = "https://translate.googleapis.com/translate_a/single"
    params = {"client": "gtx", "sl": "en", "tl": "vi", "dt": "t", "q": word}
    for attempt in range(3):
        try:
            res = requests.get(url, params=params, headers=headers, timeout=5)
            data = res.json()
            # Ghép đủ các đoạn dịch thay vì chỉ lấy đoạn đầu tiên
            meaning = "".join(chunk[0] for chunk in data[0] if chunk[0])
            if meaning.strip():
                return meaning.strip()
        except Exception:
            continue

    # Nguồn 2 (dự phòng): MyMemory API nếu Google không phản hồi được
    try:
        backup_url = "https://api.mymemory.translated.net/get"
        backup_params = {"q": word, "langpair": "en|vi"}
        res = requests.get(backup_url, params=backup_params, headers=headers, timeout=5)
        data = res.json()
        meaning = data.get("responseData", {}).get("translatedText", "")
        if meaning.strip():
            return meaning.strip()
    except Exception:
        pass

    return "Chưa rõ nghĩa"

@app.route("/", methods=["GET", "POST"])
def index():
    conn = get_db()
    cur = conn.cursor()
    add_message = None
    if request.method == "POST":
        word = clean_input(request.form.get("word", ""))
        if word:
            # Kiểm tra trùng (không phân biệt hoa/thường) trước khi thêm,
            # tránh tạo ra 2 dòng cùng 1 từ trong sổ.
            cur.execute("SELECT id FROM vocab WHERE LOWER(en) = LOWER(?)", (word,))
            existing = cur.fetchone()
            if existing:
                add_message = ("warn", f"“{word}” đã có trong sổ rồi, không thêm trùng nữa nhé.")
            else:
                # translate() và get_word_details() gọi 2 API khác nhau, không phụ
                # thuộc nhau. Trước đây chạy tuần tự nên tổng thời gian chờ = t1 + t2
                # (có thể tới ~15-20s nếu mạng chậm/API đầu tiên phải retry).
                # Chạy song song để tổng thời gian chờ chỉ còn max(t1, t2).
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_meaning = executor.submit(translate, word)
                    future_details = executor.submit(get_word_details, word)
                    meaning = future_meaning.result()
                    ipa, defn, ex, syn, word_type = future_details.result()
                cur.execute(
                    "INSERT INTO vocab (en, vi, ipa, definition, example, synonyms, word_type) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                    (word, meaning, ipa, defn, ex, syn, word_type)
                )
                conn.commit()
                add_message = ("ok", f"Đã thêm “{word}” vào sổ.")

    # Chỉ lấy những từ chưa tốt nghiệp
    cur.execute("""
        SELECT * FROM vocab 
        WHERE is_graduated = 0 
        AND (next_review IS NULL OR next_review <= datetime('now', 'localtime'))
    """)
    vocab = cur.fetchall()
    
    cur.execute("SELECT * FROM vocab WHERE is_graduated = 1")
    graduated = cur.fetchall()

    # Trước đây chạy lại đúng query của `vocab` lần thứ 3 chỉ để build quiz_data.
    # Giờ tái sử dụng luôn kết quả `vocab` đã fetch ở trên — cùng 1 dữ liệu, khỏi query lại.
    quiz_data = [
        {"id": w["id"], "en": w["en"], "vi": w["vi"], "ipa": w["ipa"] if w["ipa"] else "",
         "definition": w["definition"] if w["definition"] else "No description available.",
         "example": w["example"] if w["example"] else "", 
         "synonyms": w["synonyms"] if "synonyms" in w.keys() and w["synonyms"] else "",
         "word_type": w["word_type"] if "word_type" in w.keys() and w["word_type"] else "",
         "correct": w["correct"], "wrong": w["wrong"], "is_graduated": w["is_graduated"]}
        for w in vocab
    ]
    return render_template("index.html", vocab=vocab, graduated=graduated, quiz_data=quiz_data, add_message=add_message)

@app.route("/graduate/<int:id>", methods=["POST"])
def graduate(id):
    conn = get_db()
    conn.execute("UPDATE vocab SET is_graduated = 1 WHERE id=?", (id,))
    conn.commit()
    return redirect("/")

@app.route("/revive/<int:id>", methods=["POST"])
def revive(id):
    conn = get_db()
    conn.execute("UPDATE vocab SET is_graduated = 0, next_review = datetime('now', 'localtime') WHERE id=?", (id,))
    conn.commit()
    return redirect("/")

@app.route("/delete/<int:id>", methods=["POST"])
def delete(id):
    conn = get_db()
    # Xoá luôn log học tập gắn với từ này — nếu không, các dòng trong
    # learning_logs sẽ mồ côi (trỏ tới vocab_id không còn tồn tại) và
    # tích tụ mãi trong DB theo thời gian.
    conn.execute("DELETE FROM learning_logs WHERE vocab_id=?", (id,))
    conn.execute("DELETE FROM vocab WHERE id=?", (id,))
    conn.commit()
    return redirect("/")

@app.route("/edit/<int:id>", methods=["POST"])
def edit(id):
    vi = clean_input(request.form.get("vi", ""))
    conn = get_db()
    conn.execute("UPDATE vocab SET vi=? WHERE id=?", (vi, id))
    conn.commit()
    return redirect("/")

@app.route("/answer/<int:id>", methods=["POST"])
def answer(id):
    user_ans, correct_raw = clean_input(request.form.get("answer", "")), clean_input(request.form.get("correct", "")) 
    user_input = clean_input(request.form.get("user_input", ""))
    
    try: response_time = float(request.form.get("time", 0.0))
    except ValueError: response_time = 0.0

    conn = get_db()
    cur = conn.cursor()
    
    user_clean, raw_target_clean = remove_accents(user_ans), remove_accents(correct_raw)
    correct_options = [remove_accents(x.strip()) for x in re.split(r'[,;|]', correct_raw)]
    
    is_correct = 1 if user_clean in correct_options or (user_clean == raw_target_clean and raw_target_clean != "") else 0
    if is_correct: conn.execute("UPDATE vocab SET correct = correct + 1 WHERE id=?", (id,))
    else: conn.execute("UPDATE vocab SET wrong = wrong + 1 WHERE id=?", (id,))
    
    conn.execute("INSERT INTO learning_logs (vocab_id, is_correct, response_time_seconds, user_input) VALUES (?, ?, ?, ?)",
                 (id, is_correct, response_time, user_input))
                 
    cur.execute("SELECT vi, synonyms FROM vocab WHERE id=?", (id,))
    current_word = cur.fetchone()
    
    similar_words_in_db = []
    api_synonyms = ""
    
    if current_word:
        if current_word['vi']:
            vi_meaning = current_word['vi'].split(',')[0].strip()
            cur.execute("SELECT en FROM vocab WHERE vi LIKE ? AND id != ? LIMIT 3", (f"%{vi_meaning}%", id))
            similar_words_in_db = [row['en'] for row in cur.fetchall()]
        
        api_synonyms = current_word['synonyms'] if current_word['synonyms'] else ""

    conn.commit()
    
    return jsonify({
        "status": "success",
        "is_correct": is_correct,
        "db_synonyms": similar_words_in_db,
        "api_synonyms": api_synonyms
    }), 200

@app.route("/sm2_rate/<int:id>", methods=["POST"])
def sm2_rate(id):
    try:
        rating = int(request.form.get("rating", 2))
    except ValueError:
        rating = 2

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT interval, ef, repetitions FROM vocab WHERE id=?", (id,))
    word = cur.fetchone()
    
    if not word:
        return jsonify({"status": "error", "message": "Word not found"}), 404

    interval = word["interval"] if word["interval"] is not None else 0
    ef = word["ef"] if word["ef"] is not None else 2.5
    repetitions = word["repetitions"] if word["repetitions"] is not None else 0

    if repetitions == 0:
        if rating == 0:   # Again
            interval, repetitions = 0, 0
        elif rating == 1: # Hard
            interval, repetitions = 1, 1
        elif rating == 2: # Good -> Lên 3 ngày
            interval, repetitions = 3, 1
        elif rating == 3: # Easy -> Lên 7 ngày
            interval, ef, repetitions = 7, ef + 0.15, 1
    else:
        if rating == 0:   # Again
            interval, ef, repetitions = 1, max(1.3, ef - 0.2), 0
        elif rating == 1: # Hard
            interval = max(1, int(interval * 1.2))
            ef = max(1.3, ef - 0.15)
            repetitions += 1
        elif rating == 2: # Good
            interval = int(interval * ef)
            repetitions += 1
        elif rating == 3: # Easy
            interval = int(interval * ef * 1.3)
            ef += 0.15
            repetitions += 1

    # Thực hiện lưu thông số và ĐẨY NGÀY HẸN GẶP LẠI lên SQL dựa vào số ngày (`interval`) vừa tính
    conn.execute("""
        UPDATE vocab 
        SET interval=?, ef=?, repetitions=?, 
            next_review = datetime('now', 'localtime', '+' || ? || ' days') 
        WHERE id=?
    """, (interval, ef, repetitions, interval, id))
    conn.commit()
    
    return jsonify({"status": "success", "new_interval": interval}), 200

# ==================== PHẦN DATA ANALYST & MACHINE LEARNING ====================

# Cache đơn giản trong bộ nhớ: chỉ train lại RandomForest + vẽ lại biểu đồ
# khi số lượng log thay đổi so với lần train gần nhất, thay vì mỗi lần
# ai đó mở trang /analytics là train lại từ đầu.
_ml_cache = {"log_count": None, "ml_accuracy": 0, "ml_features": {}}


def _get_ml_results(df_logs, df_vocab, log_count):
    """Trả kết quả ML đã cache nếu log_count không đổi kể từ lần train trước.
    Chỉ thực sự train (và ghi lại biểu đồ) khi có log mới."""
    if _ml_cache["log_count"] == log_count:
        return _ml_cache["ml_accuracy"], _ml_cache["ml_features"]

    ml_accuracy = 0
    ml_features = {}

    if len(df_logs) >= 10:
        df = df_logs.merge(df_vocab, left_on='vocab_id', right_on='id')
        df['word_len'] = df['en'].str.len()

        os.makedirs("static/charts", exist_ok=True)
        plt.figure(figsize=(9, 4))
        sns.histplot(data=df, x='response_time_seconds', hue='is_correct', kde=True, multiple="stack", palette="Set2")
        plt.title("Phan Bo Thoi Gian Phan Xa (Giay)")
        plt.xlabel("Thoi gian phan xa (giay)")
        plt.ylabel("So lan tuong tac")
        plt.tight_layout()
        plt.savefig("static/charts/response_chart.png")
        plt.close()

        df['attempt_index'] = df.groupby('vocab_id').cumcount() + 1
        X = df[['response_time_seconds', 'word_len', 'attempt_index']]
        y = df['is_correct']

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            ml_accuracy = round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)
            importances = model.feature_importances_
            ml_features = {
                "Tốc độ phản xạ": round(importances[0] * 100, 2),
                "Độ dài của từ vựng": round(importances[1] * 100, 2),
                "Số lần lặp lại từ vựng": round(importances[2] * 100, 2)
            }
        except Exception:
            pass

    _ml_cache["log_count"] = log_count
    _ml_cache["ml_accuracy"] = ml_accuracy
    _ml_cache["ml_features"] = ml_features
    return ml_accuracy, ml_features


@app.route("/analytics")
def analytics():
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM learning_logs")
    log_count = cur.fetchone()[0]
    if log_count == 0:
        empty_kpis = {"total_reviews": 0, "accuracy_rate": 0, "avg_time": 0}
        return render_template("analytics.html", kpis=empty_kpis, hard_words=[], peak_hours=[], ml_accuracy=0, ml_features={}, log_count=0)

    cur.execute("""
        SELECT COUNT(*) as total_reviews,
               ROUND(SUM(is_correct) * 100.0 / COUNT(*), 1) as accuracy_rate,
               ROUND(AVG(response_time_seconds), 2) as avg_time
        FROM learning_logs
    """)
    kpis = cur.fetchone()
    
    cur.execute("""
        SELECT v.en, v.vi, COUNT(*) as total_attempts, 
               (COUNT(*) - SUM(l.is_correct)) as wrong_attempts,
               ROUND(SUM(l.is_correct) * 100.0 / COUNT(*), 1) as word_accuracy
        FROM learning_logs l JOIN vocab v ON l.vocab_id = v.id
        GROUP BY l.vocab_id HAVING total_attempts >= 3
        ORDER BY word_accuracy ASC, total_attempts DESC LIMIT 5
    """)
    hard_words = cur.fetchall()
    
    cur.execute("""
        SELECT strftime('%H', timestamp, '+7 hours') as study_hour, COUNT(*) as total_clicks
        FROM learning_logs GROUP BY study_hour ORDER BY total_clicks DESC LIMIT 3
    """)
    peak_hours = cur.fetchall()

    df_logs = pd.read_sql_query("SELECT * FROM learning_logs", conn)
    df_vocab = pd.read_sql_query("SELECT id, en FROM vocab", conn)

    # Chỉ train lại RandomForest + vẽ lại biểu đồ khi log_count thay đổi
    # so với lần request trước (xem _get_ml_results ở trên).
    ml_accuracy, ml_features = _get_ml_results(df_logs, df_vocab, log_count)

    return render_template("analytics.html", kpis=kpis, hard_words=hard_words, peak_hours=peak_hours, 
                           ml_accuracy=ml_accuracy, ml_features=ml_features, log_count=log_count)

if __name__ == "__main__":

    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)