import sqlite3
import nltk
from nltk.corpus import wordnet as wn


try:
    wn.ensure_loaded()
except LookupError:
    print("Đang tải từ điển WordNet lần đầu (chỉ tải 1 lần duy nhất)...")
    nltk.download('wordnet')

WORDNET_POS_VI = {
    'n': 'danh từ',
    'v': 'động từ',
    'a': 'tính từ',
    's': 'tính từ',   # adjective vệ tinh, gộp chung với tính từ cho đơn giản
    'r': 'trạng từ',
}

# WordNet chỉ chứa "từ nội dung" (danh/động/tính/trạng từ), không có các
# "từ chức năng" (giới từ, liên từ, đại từ, mạo từ...). Tập này là hữu hạn
# và cố định trong tiếng Anh nên liệt kê thủ công cho chắc.
FUNCTION_WORDS_VI = {
    # Giới từ
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

    # Liên từ
    'and': 'liên từ', 'but': 'liên từ', 'or': 'liên từ', 'nor': 'liên từ', 'yet': 'liên từ',
    'so': 'liên từ', 'because': 'liên từ', 'although': 'liên từ', 'though': 'liên từ',
    'while': 'liên từ', 'if': 'liên từ', 'unless': 'liên từ', 'whereas': 'liên từ',
    'however': 'liên từ', 'therefore': 'liên từ', 'moreover': 'liên từ',
    'furthermore': 'liên từ', 'meanwhile': 'liên từ', 'whether': 'liên từ',
    'until': 'liên từ', 'than': 'liên từ',

    # Đại từ
    'i': 'đại từ', 'you': 'đại từ', 'he': 'đại từ', 'she': 'đại từ', 'it': 'đại từ',
    'we': 'đại từ', 'they': 'đại từ', 'me': 'đại từ', 'him': 'đại từ', 'her': 'đại từ',
    'us': 'đại từ', 'them': 'đại từ', 'this': 'đại từ', 'that': 'đại từ',
    'these': 'đại từ', 'those': 'đại từ', 'who': 'đại từ', 'whom': 'đại từ',
    'whose': 'đại từ', 'which': 'đại từ', 'what': 'đại từ', 'myself': 'đại từ',
    'yourself': 'đại từ', 'himself': 'đại từ', 'herself': 'đại từ', 'itself': 'đại từ',
    'ourselves': 'đại từ', 'themselves': 'đại từ', 'someone': 'đại từ',
    'anyone': 'đại từ', 'everyone': 'đại từ', 'nobody': 'đại từ',
    'something': 'đại từ', 'anything': 'đại từ', 'everything': 'đại từ', 'nothing': 'đại từ',

    # Mạo từ / từ hạn định
    'a': 'mạo từ', 'an': 'mạo từ', 'the': 'mạo từ',
    'my': 'từ hạn định', 'your': 'từ hạn định', 'his': 'từ hạn định',
    'its': 'từ hạn định', 'our': 'từ hạn định', 'their': 'từ hạn định',
    'some': 'từ hạn định', 'any': 'từ hạn định', 'no': 'từ hạn định',
    'every': 'từ hạn định', 'each': 'từ hạn định', 'either': 'từ hạn định',
    'neither': 'từ hạn định', 'much': 'từ hạn định', 'many': 'từ hạn định',
    'few': 'từ hạn định', 'little': 'từ hạn định', 'several': 'từ hạn định',
    'all': 'từ hạn định', 'both': 'từ hạn định',

    # Thán từ
    'oh': 'thán từ', 'wow': 'thán từ', 'hey': 'thán từ', 'alas': 'thán từ',
    'ouch': 'thán từ', 'oops': 'thán từ', 'hurray': 'thán từ', 'ah': 'thán từ',
    'um': 'thán từ', 'uh': 'thán từ', 'well': 'thán từ',
}


def get_word_type_offline(word):
    """Tra loại từ hoàn toàn offline: ưu tiên từ chức năng (danh sách cố định),
    sau đó tra WordNet cho từ nội dung. Không cần mạng, không bao giờ timeout."""
    w = word.strip().lower()
    if w in FUNCTION_WORDS_VI:
        return FUNCTION_WORDS_VI[w]
    synsets = wn.synsets(w)
    if synsets:
        return WORDNET_POS_VI.get(synsets[0].pos(), "")
    return ""


def upgrade_database():
    try:
        # Kết nối tới database hiện tại
        conn = sqlite3.connect('vocab.db')
        cursor = conn.cursor()
        
        # Thêm cột user_input vào bảng learning_logs
        cursor.execute("ALTER TABLE learning_logs ADD COLUMN user_input TEXT;")
        
        conn.commit()
        print("Đã thêm cột 'user_input' thành công! Không bị mất dữ liệu cũ.")
        
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("Cột 'user_input' đã tồn tại, không cần thêm nữa.")
        else:
            print(f"Lỗi: {e}")
    finally:
        conn.close()


def upgrade_word_type_column():
    """Thêm cột word_type (danh từ/động từ/tính từ...) vào bảng vocab nếu chưa có."""
    conn = sqlite3.connect('vocab.db')
    try:
        conn.execute("ALTER TABLE vocab ADD COLUMN word_type TEXT DEFAULT '';")
        conn.commit()
        print("Đã thêm cột 'word_type' thành công!")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("Cột 'word_type' đã tồn tại, không cần thêm nữa.")
        else:
            print(f"Lỗi: {e}")
    finally:
        conn.close()


def backfill_word_types():
    """Bổ sung loại từ cho những từ đã có sẵn trong sổ nhưng chưa có word_type.
    Chạy hoàn toàn offline (WordNet) — không cần mạng, không timeout, chạy rất nhanh."""
    conn = sqlite3.connect('vocab.db')
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, en FROM vocab WHERE word_type IS NULL OR word_type = ''"
    ).fetchall()

    total = len(rows)
    if total == 0:
        print("Tất cả các từ đã có loại từ (word_type). Không cần bổ sung gì thêm.")
        conn.close()
        return

    print(f"Tìm thấy {total} từ chưa có loại từ. Bắt đầu tra cứu offline...")
    updated, skipped = 0, 0
    for i, row in enumerate(rows, start=1):
        word_type = get_word_type_offline(row["en"])
        if word_type:
            conn.execute("UPDATE vocab SET word_type = ? WHERE id = ?", (word_type, row["id"]))
            updated += 1
            print(f"[{i}/{total}] {row['en']} -> {word_type}")
        else:
            skipped += 1
            print(f"[{i}/{total}] {row['en']} -> (không tra được, có thể là cụm từ hoặc từ hiếm)")

    conn.commit()
    conn.close()
    print(f"Hoàn tất: đã cập nhật {updated} từ, bỏ qua {skipped} từ không tra được.")


if __name__ == "__main__":
    upgrade_database()
    upgrade_word_type_column()
    backfill_word_types()