from flask import Flask, render_template, request, redirect, jsonify
import sqlite3, requests
import unicodedata, re
import eng_to_ipa
import os
import pandas as pd

# Thư viện phục vụ Data Analyst & Machine Learning
import matplotlib
matplotlib.use('Agg') # Cấu hình để Flask không bị crash khi vẽ biểu đồ ngầm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ==================== PHẦN DEV (XỬ LÝ NGHIỆP VỤ APP) ====================
def clean_input(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', text).strip()

def remove_accents(text):
    text = text.lower()
    text = re.sub(r'[đĐ]', 'd', text)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.strip()

def get_db():
    conn = sqlite3.connect("vocab.db")
    conn.row_factory = sqlite3.Row
    
    # Đã sửa lại DEFAULT của next_review thành NULL
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vocab (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            en TEXT, vi TEXT, ipa TEXT DEFAULT '',
            definition TEXT DEFAULT '', example TEXT DEFAULT '',
            correct INTEGER DEFAULT 0, wrong INTEGER DEFAULT 0,
            is_graduated INTEGER DEFAULT 0,
            synonyms TEXT DEFAULT '',
            interval INTEGER DEFAULT 0,
            ef REAL DEFAULT 2.5,
            repetitions INTEGER DEFAULT 0,
            next_review TEXT DEFAULT NULL
        )
    """)
    # Bọc lót cột tự động tránh lỗi DB cũ
    for col in ['ipa', 'definition', 'example', 'synonyms']:
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
    
    # SỬA Ở ĐÂY: SQLite không cho dùng hàm động (datetime) trong DEFAULT của ALTER TABLE, dùng NULL là an toàn nhất.
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
    return conn

def get_word_details(word):
    word_clean = word.strip().lower()
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word_clean}"
    ipa, definition, example, synonyms = "/.../", "No description available.", "", ""
    try:
        res = requests.get(url, timeout=3)
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
    return ipa, definition, example, synonyms

def translate(word):
    url = "https://translate.googleapis.com/translate_a/single"
    params = {"client": "gtx", "sl": "en", "tl": "vi", "dt": "t", "q": word}
    try: return requests.get(url, params=params).json()[0][0][0]
    except Exception: return "Chưa rõ nghĩa"

@app.route("/", methods=["GET", "POST"])
def index():
    conn = get_db()
    cur = conn.cursor()
    if request.method == "POST":
        word = clean_input(request.form.get("word", ""))
        if word:
            meaning = translate(word)
            ipa, defn, ex, syn = get_word_details(word)
            cur.execute(
                "INSERT INTO vocab (en, vi, ipa, definition, example, synonyms) VALUES (?, ?, ?, ?, ?, ?)", 
                (word, meaning, ipa, defn, ex, syn)
            )
            conn.commit()

    # THAY ĐỔI LOGIC: Chỉ lấy những từ chưa tốt nghiệp VÀ đã đến hạn ôn tập (hoặc từ mới thêm)
    cur.execute("""
        SELECT * FROM vocab 
        WHERE is_graduated = 0 
        AND (next_review IS NULL OR next_review <= datetime('now', 'localtime'))
    """)
    vocab = cur.fetchall()
    
    cur.execute("SELECT * FROM vocab WHERE is_graduated = 1")
    graduated = cur.fetchall()
    
    # quiz_data cũng lọc tương tự để khi bấm F5 list Quiz không chứa từ chưa đến hạn
    cur.execute("""
        SELECT * FROM vocab 
        WHERE is_graduated = 0 
        AND (next_review IS NULL OR next_review <= datetime('now', 'localtime'))
    """)
    quiz_data = [
        {"id": w["id"], "en": w["en"], "vi": w["vi"], "ipa": w["ipa"] if w["ipa"] else "",
         "definition": w["definition"] if w["definition"] else "No description available.",
         "example": w["example"] if w["example"] else "", 
         "synonyms": w["synonyms"] if "synonyms" in w.keys() and w["synonyms"] else "",
         "correct": w["correct"], "wrong": w["wrong"], "is_graduated": w["is_graduated"]}
        for w in cur.fetchall()
    ]
    return render_template("index.html", vocab=vocab, graduated=graduated, quiz_data=quiz_data)

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

# THAY ĐỔI LOGIC: Cập nhật ngày tiếp tục xuất hiện dựa vào số ngày của interval
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
                "Tốc độ phản xạ của ông": round(importances[0] * 100, 2),
                "Độ dài của từ vựng": round(importances[1] * 100, 2),
                "Số lần lặp lại từ vựng": round(importances[2] * 100, 2)
            }
        except Exception:
            pass

    return render_template("analytics.html", kpis=kpis, hard_words=hard_words, peak_hours=peak_hours, 
                           ml_accuracy=ml_accuracy, ml_features=ml_features, log_count=log_count)

if __name__ == "__main__":
    app.run(debug=True)