import pandas as pd
from typing import List, Tuple

def define_train_predict_qa_data(
    question_answer_pair: Tuple[str]
) -> pd.DataFrame:
    
    question, answer = question_answer_pair
    question = "[CLS]" + question + "[SEP]"
    answer = answer + "[SEP]"
    data = [
        {
            "train": question + answer[:i],
            "predict": answer[i]
        }
        for i in range(answer.__len__())
    ]
    return pd.DataFrame.from_records(data)