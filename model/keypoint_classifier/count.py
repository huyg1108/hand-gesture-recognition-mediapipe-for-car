import pandas as pd

df = pd.read_csv("keypoint.csv", header=None)

# Đếm số lượng mẫu theo từng nhãn (giá trị ở cột 0)
class_counts = df[0].value_counts().sort_index()

print("Số mẫu theo từng class:")
print(class_counts)
