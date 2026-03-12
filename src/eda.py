import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("TkAgg")

# 读取数据
df = pd.read_csv("../data/train.csv")

print(df.head())
print(df.info())
print(df.describe())

# 检查缺失值
print(df.isnull().sum())

# 生存人数分布
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# 性别 vs 生存
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

# 舱位 vs 生存
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

# 年龄分布
sns.histplot(df['Age'], bins=30)
plt.title("Age Distribution")
plt.show()

# 相关性热图
corr = df.corr(numeric_only=True)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()