from peft import PeftModel
import pandas as pd
import re
import pandas as pd
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)




if __name__ == "__main__":
    MODEL_NAME = "AdaptLLM/finance-chat"
    PEFT_MODEL_LOCAL_CHECKPOINT = "./outputs/peft-training-checkpoint"
    PEFT_MODEL_ADAPTER_ID = "anhtranhong/finance-chat_fingpt-fiqa_qa_v2" 

    peft_model = PeftModel.from_pretrained(model, PEFT_MODEL_ADAPTER_ID, is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(
        PEFT_MODEL_ADAPTER_ID,
        padding_side="left",
    )



    # Load the CSV file
    file_path = '/usr1/home/s124mdg41_08/FinLLM-FOMC/data/processed/Merged_FMOC.csv'
    df = pd.read_csv(file_path)

    # Function to split each note by paragraphs and remove paragraphs with less than 50 characters
    def split_into_paragraphs(row):
        date = row['Date']
        paragraphs = str(row['Minutes_cleaned']).split('\n')
        # Only keep paragraphs that are non-empty and have at least 50 characters (ignoring spaces)
        return [(date, para.strip()) for para in paragraphs if para.strip() and len(para.strip()) >= 40]

    # Apply the function to each row in the DataFrame
    split_data = [item for idx, row in df.iterrows() for item in split_into_paragraphs(row)]

    # Create a new DataFrame with the split paragraphs
    new_df = pd.DataFrame(split_data, columns=['Date', 'Minutes_cleaned'])

    # 删除 'Minutes_cleaned' 列中为空字符串或者'NaN' 的行
    new_df = new_df[new_df['Minutes_cleaned'].notna() & (new_df['Minutes_cleaned'] != '')]

    # 输出前50行并输出行数
    print(new_df.head(50))
    print(len(new_df))





    # 初始化一个空列表来保存每一行的结果
    results = []

    # 遍历 new_df 的每一行
    for idx, row in new_df.iterrows():
        content = row['Minutes_cleaned']  # 获取 'Minutes_cleaned' 列的内容
        date_value = row['Date']  # 获取 'Date' 列的值

        our_system_prompt = """You're an expert on sentiment analysis in economic texts"""

        user_input = f"""Please analyze the economic sentiment of the following content and provide a score from 0 to 10, where 0 represents extremely negative sentiment, 5 is neutral, and 10 is extremely positive. Analyze only the economic-related aspects in each paragraph. If you can't judge the emotional content, judge it as 5.
            Content: {content}
            <requirement>Only show number of one score, no more text.</requirement>
        """
        
        # 构建最终的 LLaMA2 prompt
        prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{user_input} [/INST]"

        # 生成情感分数
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

        outputs = peft_model.generate(
            input_ids=inputs,
            do_sample=True,
            max_new_tokens=1024,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            num_return_sequences=1,
        )[0]

        # 解码并提取分数
        answer_start = int(inputs.shape[-1])
        pred = tokenizer.decode(outputs[answer_start:], skip_special_tokens=True)


    # 使用 re.search() 获取第一个匹配的数字
        score = re.search(r'\d+', pred)
        print(score)

        # 处理提取的分数
        if score:
            score_value = int(score[0])  # 提取第一个匹配的分数
        else:
            score_value = None  # 如果没有找到分数，设置为 None

        # 将结果与 Date 合并，存入结果列表
        results.append({
            'Date': date_value,
            'Sentiment_Score': score_value
        })
        

    # 将结果列表转换为 DataFrame
    result_df = pd.DataFrame(results)


    csv_file_path = '/usr1/home/s124mdg41_08/FinLLM-FOMC/data/sentiment_score/sentiment_analysis_results.csv'
    result_df.to_csv(csv_file_path, index=False)

    print(f"Results saved to {csv_file_path}")



