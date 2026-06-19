import sqlite3

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

if __name__ == "__main__":
    upgrade_database()