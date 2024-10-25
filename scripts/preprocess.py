import pandas as pd
import spacy
import pandas as pd
import re
from tqdm import tqdm

NER = spacy.load("en_core_web_lg") # 创建NER类用来分类
file_path = "~/PycharmProjects/pythonProject1/FOMC_Minutes.csv"

# 定义一个去除Minutes文章"__________"后面的无用内容的函数
def remove_text_after_underscores(text):
    return re.sub(r'__*.*', '', text, flags=re.DOTALL)

data = pd.read_csv(file_path)

data['Minutes_cleaned'] = data['Minutes'].apply(remove_text_after_underscores) # 去除Minutes文章后面无用内容

for index, row in tqdm(data.iterrows()): # 遍历data数据每一行
    text = row['Minutes_cleaned'] # 提取每一行需要进一步处理的内容
    paragraphs = text.split('\n') # 将要处理的内容按照段落拆分

    kept_paragraphs = []
    entity_density_mark = []
    paragraphs_mark = []
    total_tokens_mark = []
    named_entities_mark = []

    for paragraph in paragraphs: # 遍历每一个段落
        named_entities=0
        doc = NER(paragraph) # 对每一段进行NER分类
        total_tokens = len(doc)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG']: # 计算每一段中被分类为"PERSON"和"ORG"的token数量
                words_in_ent = [token for token in ent]
                named_entities += len(words_in_ent)

        if total_tokens > 0:
            entity_density = named_entities / total_tokens # 计算每一段被分类为"PERSON"和"ORG"的token数量占总数量占比
        else:
            entity_density = 0
        entity_density_mark.append(entity_density)
        paragraphs_mark.append(paragraph)
        named_entities_mark.append(named_entities)
        total_tokens_mark.append(total_tokens)

        if entity_density < 0.2: # 对于占比小于20%的段落进行保留
            kept_paragraphs.append(paragraph)

    final_text = '\n'.join(kept_paragraphs)
    data.at[index, 'Minutes_cleaned'] = final_text

data.to_csv('FOMC_Minutes_cleaned.csv', index=False, encoding='utf-8')


# 读取两个文件
file1 = pd.read_csv('FOMC_Minutes_cleaned.csv')
file2 = pd.read_csv('FOMC_Implementation_Note.csv')
file3 = pd.read_csv('FOMC_Statement.csv')

# 第一次合并：合并 file1 和 file2
merged_data = pd.merge(file1[['Date', 'Minutes_cleaned']], file2[['Date', 'Implementation Note']], on='Date', how='outer')

# 第二次合并：将上次的合并结果与 file3 合并
merged_data = pd.merge(merged_data, file3[['Date', 'Statement']], on='Date', how='outer')

# 保存合并后的数据到新的CSV文件
merged_data.to_csv('merged_FMOC.csv', index=False, encoding='utf-8')

print("文件已成功合并并保存到 'merged_FMOC.csv'")