# FinLLM-FOMC

## 目录结构

```plaintext
FOMC_Sentiment_Project/
│
├── data/                          # 存放原始数据和处理后的数据
│   ├── raw/                       # 原始数据（FOMC 会议记录）
│   ├── processed/                 # 预处理后的数据
│
├── notebooks/                     # Jupyter Notebooks 存放目录
│   ├── data_analysis.ipynb        # 数据探索和初步分析
│   └── sentiment_analysis.ipynb   # 情感分析实验
│
├── scripts/                       # Python 脚本文件
│   ├── preprocess.py              # 数据预处理脚本
│   ├── sentiment_score.py         # 计算情感得分的脚本
│   ├── visualize_results.py       # 可视化情感分析结果的脚本
│
├── models/                        # 存储训练的模型和模型参数
│   ├── sentiment_model.pkl        # 情感分析模型文件
│
├── logs/                          # 存放日志文件
│   └── sentiment_analysis.log     # 运行分析的日志记录
│
├── config/                        # 配置文件目录
│   └── config.yaml                # 配置文件，用于存储路径、参数等信息
│
├── requirements.txt               # 项目依赖库列表
├── README.md                      # 项目简介和使用说明
├── .gitignore                     # 忽略不必要上传到 Git 的文件和目录
└── main.py                        # 项目主入口
