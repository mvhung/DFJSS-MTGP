import matplotlib.pyplot as plt
import numpy as np

# Danh sách các đối thủ và tỷ lệ thắng/tổng số trận tương ứng
env = ['env_1', 'env_2', 'env_3', 'env_4']
ccgp = [206, 200, 512,  1136]
mtgp = [190,  222 ,440,  798]

x = np.arange(len(env))  # Vị trí của các nhãn trên trục x
width = 0.35  # Độ rộng của các cột

fig, ax = plt.subplots()
rects1 = ax.bar(x, ccgp, width, label='số thao tác')
# rects2 = ax.bar(x + width/2, mtgp, width, label='MTGP')


# Thêm một số thuộc tính văn bản cho các nhãn, tiêu đề và tùy chỉnh trục x
ax.set_xlabel('Môi trường')
ax.set_ylabel('Số thao tác')
ax.set_title('Biểu đồ liệt kê số thao tác')
ax.set_xticks(x)
ax.set_xticklabels(env)
ax.legend()

# Hiển thị giá trị trên mỗi cột
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

plt.show()
