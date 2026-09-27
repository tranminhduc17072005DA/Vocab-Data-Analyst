import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import difflib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. PAGE CONFIG & UI SETUP
st.set_page_config(page_title="Phân tích Từ vựng", page_icon="", layout="wide")

with st.sidebar:
    st.title("Data-Driven SRS")
    st.markdown("""
    **Mục tiêu:** Xây dựng Dashboard phân tích và dự đoán dữ liệu cho hệ thống lặp lại ngắt quãng (Spaced Repetition System).
    
    **Mô hình ứng dụng:**
    - **Thuật toán BKT (Bayesian Knowledge Tracing):** Tiền xử lý và lọc nhiễu dữ liệu do lỗi thao tác bàn phím.
    - **Đường cong Ebbinghaus:** Mô hình hóa sự suy giảm khả năng ghi nhớ theo thời gian.
    - **NLP & Clustering:** Trích xuất đặc trưng văn bản và phân cụm dữ liệu bằng thuật toán K-Means.
    """)

# 2. ADVANCED ETL PIPELINE
@st.cache_data(ttl=30)  # 10s cũ hơi thấp -> query lại DB liên tục dù data không đổi
def load_and_transform_data():
    try:
        conn = sqlite3.connect("vocab.db")
        df_vocab = pd.read_sql_query("SELECT * FROM vocab", conn)
        df_logs = pd.read_sql_query("SELECT * FROM learning_logs", conn)
        conn.close()
    except Exception as e:
        st.error(f"Lỗi kết nối cơ sở dữ liệu: {e}")
        return pd.DataFrame(), pd.DataFrame()

    if df_vocab.empty or df_logs.empty:
        return df_vocab, df_logs

    # Chuẩn hóa kiểu dữ liệu số (Numeric casting)
    df_vocab['wrong'] = pd.to_numeric(df_vocab['wrong'], errors='coerce').fillna(0)
    df_vocab['correct'] = pd.to_numeric(df_vocab['correct'], errors='coerce').fillna(0)

    # Khởi tạo các biến cơ sở (Base metrics)
    df_vocab['total_attempts'] = df_vocab['correct'] + df_vocab['wrong']
    df_vocab['error_rate'] = np.where(df_vocab['total_attempts'] > 0, 
                                      df_vocab['wrong'] / df_vocab['total_attempts'], 0)
    df_vocab['word_length'] = df_vocab['en'].str.len()

    # --- FEATURE ENGINEERING 1: TYPO FILTERING & BKT LOGIC ---
    if 'vi' not in df_logs.columns and 'vocab_id' in df_logs.columns:
        df_logs = df_logs.merge(df_vocab[['id', 'vi']], left_on='vocab_id', right_on='id', how='left')

    def get_similarity(target, typed):
        if pd.isna(typed) or pd.isna(target): return 0.0
        return difflib.SequenceMatcher(None, str(target).strip().lower(), str(typed).strip().lower()).ratio()

    if 'user_input' in df_logs.columns:
        df_logs['similarity'] = df_logs.apply(lambda r: get_similarity(r['vi'], r['user_input']), axis=1)
        df_logs['is_typo'] = np.where((df_logs['is_correct'] == 0) & (df_logs['similarity'] >= 0.7), 1, 0)
    else:
        df_logs['is_typo'] = 0

    # Tổng hợp thống kê từ bảng logs
    bkt_stats = df_logs.groupby('vocab_id').agg(
        true_wrong=('is_correct', lambda x: (x == 0).sum()),
        typos=('is_typo', 'sum')
    ).reset_index()
    
    # Tích hợp dữ liệu và tính toán tỷ lệ lỗi hiệu chỉnh (Adjusted Error Rate)
    df_vocab = df_vocab.merge(bkt_stats, left_on='id', right_on='vocab_id', how='left').fillna(0)
    df_vocab['memory_gaps'] = df_vocab['true_wrong'] - df_vocab['typos']
    df_vocab['memory_gaps'] = df_vocab['memory_gaps'].clip(lower=0) # Đảm bảo giá trị phi âm
    df_vocab['adjusted_error_rate'] = np.where(df_vocab['total_attempts'] > 0, 
                                               df_vocab['memory_gaps'] / df_vocab['total_attempts'], 0)

    # --- FEATURE ENGINEERING 2: TIME-SERIES & FORGETTING CURVE ---
    if 'timestamp' not in df_logs.columns:
        df_logs['timestamp'] = datetime.now() - pd.to_timedelta(np.random.randint(1, 30, size=len(df_logs)), unit='d')
    else:
        df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])

    last_review = df_logs.groupby('vocab_id')['timestamp'].max().reset_index()
    last_review.columns = ['vocab_id', 'last_review_date']
    df_vocab = df_vocab.merge(last_review, on='vocab_id', how='left')
    df_vocab['days_since_last_review'] = (datetime.now() - pd.to_datetime(df_vocab['last_review_date'])).dt.days
    df_vocab['days_since_last_review'] = df_vocab['days_since_last_review'].fillna(30)

    # Phân cụm trạng thái dựa trên tỷ lệ lỗi
    conditions = [
        (df_vocab['adjusted_error_rate'] == 0) & (df_vocab['total_attempts'] > 0),
        (df_vocab['adjusted_error_rate'] > 0) & (df_vocab['adjusted_error_rate'] <= 0.3),
        (df_vocab['adjusted_error_rate'] > 0.3)
    ]
    df_vocab['difficulty_cluster'] = np.select(conditions, ['Đã thuộc', 'Đang học', 'Cảnh báo'], default='Mới')

    return df_vocab, df_logs

df_vocab, df_logs = load_and_transform_data()


# --- CÁC HÀM ML/NLP ĐƯỢC CACHE ---
# Trước đây phần train RandomForest (tab 3) và TF-IDF + KMeans + PCA (tab 4)
# nằm trực tiếp trong "with tab3:" / "with tab4:", nên MỖI LẦN Streamlit
# rerun script (đổi tab, gõ vào 1 ô input, tương tác widget bất kỳ...) đều
# train lại từ đầu, dù df_vocab không hề đổi. Bọc trong @st.cache_data để
# chỉ tính lại khi nội dung df_vocab thật sự thay đổi.

@st.cache_data
def train_prediction_model(ml_df: pd.DataFrame):
    """Train RandomForest dự đoán từ có nguy cơ bị quên. Trả về None nếu
    chưa đủ dữ liệu hoặc nhãn không đủ đa dạng để train."""
    if len(ml_df) < 10:
        return None

    X = ml_df[['word_length', 'total_attempts', 'typos', 'days_since_last_review']]
    y = (ml_df['adjusted_error_rate'] > 0.3).astype(int)

    if len(y.unique()) <= 1:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    importances = pd.DataFrame({
        'Feature': ['Độ dài từ', 'Tổng lượt tương tác', 'Lỗi thao tác (Slips)', 'Khoảng cách thời gian'],
        'Importance': model.feature_importances_
    }).sort_values('Importance')

    return acc, importances


@st.cache_data
def compute_nlp_clusters(df_vocab: pd.DataFrame):
    """Trích xuất TF-IDF, phân cụm KMeans, giảm chiều PCA. Trả về None nếu
    chưa đủ dữ liệu để phân cụm có ý nghĩa."""
    if len(df_vocab) <= 5:
        return None

    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    X_text = tfidf.fit_transform(df_vocab['en'])

    num_clusters = min(4, len(df_vocab) // 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_text)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_text.toarray())

    result = df_vocab.copy()
    result['nlp_cluster'] = [f"Cụm {c+1}" for c in clusters]
    result['PCA_1'] = components[:, 0]
    result['PCA_2'] = components[:, 1]
    return result

# 3. MAIN DASHBOARD LAYOUT
st.title("Vocabulary Analytics Dashboard")

if df_vocab.empty:
    st.warning("Hệ thống chưa ghi nhận dữ liệu. Vui lòng thực hiện các bài tập trên Web App để thu thập tập dữ liệu đầu vào.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Tổng quan KPIs", "Phân tích Hành vi", "Mô hình Dự đoán", "Phân cụm NLP"])

# --- TAB 1: KPIs ---
with tab1:
    st.subheader("Chỉ số Hiệu suất Thực tế (Performance Metrics)")
    
    # 1. TÍNH TOÁN CÁC CHỈ SỐ CƠ SỞ (BASE METRICS)
    total_words = len(df_vocab)
    total_attempts_all = df_vocab['total_attempts'].sum()
    
    if total_attempts_all > 0:
        avg_raw_error = (df_vocab['wrong'].sum() / total_attempts_all) * 100
    else:
        avg_raw_error = 0
        
    avg_adj_error = df_vocab['adjusted_error_rate'].mean() * 100

    # 2. HIỂN THỊ 4 CỘT KPI (THEO LUỒNG DATA FLOW)
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Tổng số từ vựng", f"{total_words}")
    col2.metric("Tổng lượt tương tác", f"{int(total_attempts_all)}")
    col3.metric("Tỷ lệ sai", f"{avg_raw_error:.1f}%")
    col4.metric("Tỷ lệ quên thực tế (Lapses)", f"{avg_adj_error:.1f}%", 
                delta=f"Lệch {avg_adj_error - avg_raw_error:.1f}% (Sau chuẩn hóa)", 
                delta_color="inverse")

    st.markdown("---")
    
    # 3. ROW 1: TRẠNG THÁI & SỐ LẦN SAI NGUYÊN BẢN
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.write("**Phân bổ mức độ ghi nhớ từ vựng**")
        fig_pie = px.pie(df_vocab, names='difficulty_cluster', hole=0.4, 
                         color='difficulty_cluster',
                         color_discrete_map={'Đã thuộc': '#10b981', 'Đang học': '#f59e0b', 'Cảnh báo': '#ef4444', 'Mới': '#94a3b8'})
        fig_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.write("**Thống kê từ vựng có tần suất lỗi sai cao nhất**")
        top_raw_errors = df_vocab.sort_values('wrong', ascending=False).head(5)
        fig_bar_raw = px.bar(top_raw_errors, x='wrong', y='en', orientation='h',
                             text='wrong', color='wrong', color_continuous_scale='Oranges',
                             labels={'wrong': 'Số lần nhập sai', 'en': 'Từ vựng'})
        fig_bar_raw.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_bar_raw, use_container_width=True)

    # 4. ROW 2: BIỂU ĐỒ LỖI THAO TÁC (SLIPS)
    st.markdown("---")
    st.write("**Thống kê từ vựng có tần suất lỗi thao tác cao nhất (Slips)**")
    
    top_typos = df_vocab[df_vocab['typos'] > 0].sort_values('typos', ascending=False).head(5)
    
    if not top_typos.empty:
        fig_typo = px.bar(top_typos, x='typos', y='en', orientation='h',
                         text='typos', color='typos', color_continuous_scale='Blues',
                         labels={'typos': 'Số lần lỗi thao tác', 'en': 'Từ vựng'})
        fig_typo.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=20, b=20, l=0, r=0))
        st.plotly_chart(fig_typo, use_container_width=True)
    else:
        st.info("Hệ thống chưa ghi nhận lỗi thao tác (Slips) từ tập dữ liệu hiện tại.")

# --- TAB 2: BEHAVIORAL EDA ---
with tab2:
    st.subheader("Phân tích Phân phối Thời gian và Cấu trúc Lỗi")
    if not df_logs.empty and 'response_time_seconds' in df_logs.columns:
        df_logs_filtered = df_logs[df_logs['response_time_seconds'] <= 30].copy()
        
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.write("**Phân phối thời gian phản hồi (Response Time)**")
            fig_hist = px.histogram(df_logs_filtered, x='response_time_seconds', color='is_correct',
                                    barmode='stack', nbins=40, color_discrete_map={1: '#10b981', 0: '#ef4444'},
                                    labels={'response_time_seconds': 'Thời gian (Giây)', 'is_correct': 'Đúng/Sai'})
            fig_hist.update_layout(margin=dict(t=20, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_eda2:
            st.write("**Phân rã cấu trúc lỗi sai dựa trên độ tương đồng chuỗi**")
            wrong_logs = df_logs[df_logs['is_correct'] == 0].copy()
            if not wrong_logs.empty:
                wrong_logs['Type'] = np.where(wrong_logs['is_typo'] == 1, 'Lỗi thao tác (Slips)', 'Lỗi nhận thức (Lapses)')
                fig_err = px.pie(wrong_logs, names='Type', hole=0.4, color='Type',
                                 color_discrete_map={'Lỗi thao tác (Slips)': '#f59e0b', 'Lỗi nhận thức (Lapses)': '#ef4444'})
                fig_err.update_layout(margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(fig_err, use_container_width=True)
    else:
        st.info("Kích thước tập dữ liệu log chưa đủ để thực hiện phân tích hành vi Exploratory Data Analysis (EDA).")

# --- TAB 3: MACHINE LEARNING ---
with tab3:
    st.subheader("Mô hình Học máy: Dự đoán Trạng thái Ghi nhớ")
    st.markdown("*Áp dụng thuật toán **Random Forest Classifier** kết hợp biến đổi thời gian nhằm dự đoán khả năng nhớ từ, hỗ trợ tối ưu hóa chu kỳ lặp lại (Spaced Repetition).*")
    
    ml_df = df_vocab[df_vocab['total_attempts'] >= 1].copy()
    ml_result = train_prediction_model(ml_df)

    if len(ml_df) < 10:
        st.warning("Yêu cầu kích thước mẫu tối thiểu (n ≥ 10) để tiến hành huấn luyện mô hình.")
    elif ml_result is None:
        st.info("Tập dữ liệu huấn luyện (Training set) chưa đạt đủ độ phân tán phương sai để thiết lập mô hình.")
    else:
        acc, importances = ml_result
        col_ml1, col_ml2 = st.columns([1, 2])
        with col_ml1:
            st.metric("Độ chính xác của Mô hình (Accuracy)", f"{acc*100:.1f}%")

        with col_ml2:
            fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', title="Mức độ đóng góp của các đặc trưng (Feature Importance)")
            fig_imp.update_layout(margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_imp, use_container_width=True)

# --- TAB 4: NLP CLUSTERING ---
with tab4:
    st.subheader("Phân tích Cụm (Clustering): Không gian Hình thái Từ vựng")
    st.markdown("*Sử dụng phương pháp trích xuất đặc trưng **TF-IDF (Character N-grams)** kết hợp giảm chiều dữ liệu **PCA** nhằm phân cụm các từ vựng có cấu trúc tương đồng.*")
    
    clustered_df = compute_nlp_clusters(df_vocab)

    if clustered_df is not None:
        fig_scatter = px.scatter(clustered_df, x='PCA_1', y='PCA_2', color='nlp_cluster', hover_name='en',
                                 title="Bản đồ 2D Không gian Vector Từ vựng", size='word_length',
                                 labels={'PCA_1': 'Thành phần chính 1 (PC1)', 'PCA_2': 'Thành phần chính 2 (PC2)', 'nlp_cluster': 'Cụm phân bổ'})
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        with st.expander("Bảng Dữ liệu Trực quan theo Cụm (Data Table)"):
            display_df = clustered_df[['en', 'vi', 'nlp_cluster']].sort_values('nlp_cluster')
            display_df.columns = ['Từ vựng', 'Ngữ nghĩa', 'Cụm NLP']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Kích thước tập dữ liệu chưa đủ để thực thi thuật toán phân cụm không gian vector.")