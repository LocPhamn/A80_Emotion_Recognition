import pandas as pd
import matplotlib.pyplot as plt
import os

# Đường dẫn tới file CSV
csv_path = r"C:\Users\ADMIN\Downloads\checkpoints\content\checkpoints\training_metrics.csv"

# Kiểm tra file tồn tại
if not os.path.exists(csv_path):
    print(f"File không tồn tại: {csv_path}")
    exit()

# Đọc dữ liệu từ CSV
df = pd.read_csv(csv_path)

# Kiểm tra các cột cần thiết
required_columns = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
for col in required_columns:
    if col not in df.columns:
        print(f"Cột '{col}' không tồn tại trong CSV")
        exit()

# Tạo figure với 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Vẽ biểu đồ Loss
ax1.plot(df.index, df['train_loss'], label='Train Loss', marker='o', linewidth=2)
ax1.plot(df.index, df['val_loss'], label='Val Loss', marker='s', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Vẽ biểu đồ Accuracy
ax2.plot(df.index, df['train_acc'], label='Train Accuracy', marker='o', linewidth=2)
ax2.plot(df.index, df['val_acc'], label='Val Accuracy', marker='s', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Điều chỉnh layout
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()

# Lưu biểu đồ (tùy chọn)
output_path = r"C:\Users\ADMIN\Downloads\checkpoints\content\checkpoints\training_metrics_plot.png"
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Biểu đồ đã được lưu tại: {output_path}")
