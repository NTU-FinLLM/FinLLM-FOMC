# FinLLM-FOMC

FOMC_Sentiment_Project/<br>
│<br>
├── data/                          # 存放原始数据和处理后的数据<br>
│   ├── raw/                       # 原始数据（FOMC 会议记录）<br>
│   ├── processed/                 # 预处理后的数据<br>
│
├── notebooks/                     # Jupyter Notebooks 存放目录<br>
│   ├── data_analysis.ipynb        # 数据探索和初步分析<br>
│   └── sentiment_analysis.ipynb   # 情感分析实验<br>
│<br>
├── scripts/                       # Python 脚本文件<br>
│   ├── preprocess.py              # 数据预处理脚本<br>
│   ├── sentiment_score.py         # 计算情感得分的脚本<br>
│   ├── visualize_results.py       # 可视化情感分析结果的脚本<br>
│<br>
├── models/                        # 存储训练的模型和模型参数<br>
│   ├── sentiment_model.pkl        # 情感分析模型文件<br>
│<br>
├── logs/                          # 存放日志文件<br>
│   └── sentiment_analysis.log     # 运行分析的日志记录<br>
│<br>
├── config/                        # 配置文件目录<br>
│   └── config.yaml                # 配置文件，用于存储路径、参数等信息<br>
│<br>
├── requirements.txt               # 项目依赖库列表<br>
├── README.md                      # 项目简介和使用说明<br>
├── .gitignore                     # 忽略不必要上传到Git的文件和目录<br>
└── main.py                        # 项目主入口<br>
