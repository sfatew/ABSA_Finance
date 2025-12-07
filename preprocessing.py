import pandas as pd
import ast
from sklearn.model_selection import train_test_split

# ==========================================
# CẤU HÌNH
# ==========================================
INPUT_FILE = 'SEntFiN-v1.1.csv'  # Tên file dữ liệu gốc của bạn
TEST_SIZE = 0.2             # Tỉ lệ chia tập test (20%)
RANDOM_STATE = 42           # Cố định để kết quả chia luôn giống nhau

def transform_to_instruct_format(df, output_filename):
    """
    Hàm biến đổi DataFrame từ format SentFin sang format InstructABSA
    """
    new_data = []
    
    for index, row in df.iterrows():
        try:
            # 1. Xử lý cột Decisions: "{'SpiceJet': 'neutral'}" -> Dictionary
            # Lưu ý: Dữ liệu thực tế có thể bị lỗi format, cần try-except
            if isinstance(row['Decisions'], str):
                decisions_dict = ast.literal_eval(row['Decisions'])
            else:
                decisions_dict = row['Decisions']
            
            # 2. Tạo list aspectTerms chuẩn format repo
            aspect_terms = []
            if isinstance(decisions_dict, dict):
                for entity, sentiment in decisions_dict.items():
                    # Format: [{'term': 'ABC', 'polarity': 'positive'}]
                    aspect_terms.append({
                        'term': entity, 
                        'polarity': sentiment.lower() # Chuyển về chữ thường
                    })
            
            # Nếu dòng này không có thực thể nào thì bỏ qua
            if not aspect_terms:
                continue

            # 3. Tạo dòng dữ liệu mới
            new_row = {
                'sentenceId': f"{row['S No.']}:1",  # Giả lập ID
                'raw_text': row['Title'],           # Lấy nội dung câu
                'aspectTerms': str(aspect_terms),   # Convert list về string để lưu csv
                'aspectCategories': "[{'category': 'noaspectcategory', 'polarity': 'none'}]" # Cột giả
            }
            new_data.append(new_row)
            
        except Exception as e:
            # In ra dòng lỗi để debug nếu cần
            # print(f"Lỗi dòng {index}: {e}")
            continue

    # 4. Lưu ra file CSV
    out_df = pd.DataFrame(new_data)
    out_df.to_csv(output_filename, index=False)
    print(f"-> Đã tạo file format InstructABSA: {output_filename} ({len(out_df)} dòng)")

# ==========================================
# THỰC THI
# ==========================================
def main():
    print("1. Đang đọc dữ liệu gốc...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file '{INPUT_FILE}'. Hãy upload file này lên Colab trước.")
        return

    print(f"   Tổng số dòng: {len(df)}")

    # BƯỚC 1: CHIA TÁCH DỮ LIỆU (SPLIT)
    print("\n2. Đang chia tập Train/Test (Fixed Random Seed)...")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Lưu file RAW (Dùng cho Baseline SVM/BERT)
    train_df.to_csv('sentfin_train_raw.csv', index=False)
    test_df.to_csv('sentfin_test_raw.csv', index=False)
    print(f"-> Đã lưu file raw cho nhóm khác: 'sentfin_train_raw.csv' & 'sentfin_test_raw.csv'")

    # BƯỚC 2: BIẾN ĐỔI FORMAT (TRANSFORM)
    print("\n3. Đang convert sang format InstructABSA...")
    transform_to_instruct_format(train_df, 'instruct_train.csv')
    transform_to_instruct_format(test_df, 'instruct_test.csv')

    print("\nHOÀN TẤT! Bạn có thể dùng 2 file 'instruct_train.csv' và 'instruct_test.csv' để finetune.")

if __name__ == "__main__":
    main()