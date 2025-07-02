import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import json
import os
import time
from WM.model import LLM


class AspectMapTrainer:
    def __init__(self, json_path):
        self.filepath = json_path
        if not os.path.exists(self.filepath):
            # 默认值写初始化
            default_map = {
                "quality": ["quality", "material", "build", "solid", "design", "durable", "finish"],
                "component": ["chip", "battery"],
                "performance": ["performance", "speed", "lag", "slow", "fast", "responsive"],
                "display": ["screen", "display", "resolution", "brightness", "touch"],
                "price": ["price", "value", "cost", "cheap", "expensive", "worth"],
                "service": ["service", "support", "warranty", "helpful", "replacement", "customer service"],
                "others": ['love', 'like', 'nice', 'great']
            }
            # 转成set对象
            self.maps = {a: set(ws) for a, ws in default_map.items()}

        else:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 保证set类型
                self.maps = {k: set(v) for k, v in data.items()}

        # 3. 初始化VADER
        nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()
        self.kw2aspect = self.make_kw2aspect()
        self.update_time = 1
        self.update_frequency = 50
        self.change = True

    def save_aspect_map(self):
        # 存盘需转list
        save_map = {k: list(v) for k, v in self.maps.items()}
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(save_map, f, ensure_ascii=False, indent=2)
        print('Successfully update map file at:', self.filepath)

    # 2. 快速反查索引 - 关键词到aspect的映射
    def make_kw2aspect(self):
        kw2aspect = {}
        for aspect, words in self.maps.items():
            for w in words:
                kw2aspect[w] = aspect
        return kw2aspect

    def is_in(self, phrase):
        for x in self.maps.keys():
            if phrase in self.maps[x]:
                return True
        return False

    # 4. 新版短语归属分类（知识库优先 + LLM补充 + 旧逻辑兜底）
    def classify_phrase(self, phrase):
        phrase_lower = phrase.lower()
        # print(phrase_lower)
        # 1. 快查知识库
        if self.is_in(phrase_lower):
            return
        # 2. 大模型
        aspect_from_llm = (LLM.Add_New_Member(phrase, {k: list(v) for k, v in self.maps.items()})).lower()
        if isinstance(aspect_from_llm, str) and aspect_from_llm in self.maps:
            # 知识库实时更新
            self.maps[aspect_from_llm].add(phrase_lower)
            self.update_time += 1
            self.change = True
            print(self.update_time, self.update_frequency)
            return
            # TODO:考虑是否添加词向量辅助分类？
        return

    def train(self, df):
        """
        主流程：批量统计
        """
        siz = len(df)
        try:
            for idx, row in df.iterrows():
                review = str(row["reviewText"])
                if idx % 2000 == 0:
                    print('Current:', idx + 1, '/', siz)
                chunks = self.chunk_phrases(review)
                for phrase in chunks:
                    self.classify_phrase(phrase)
                    if self.update_time % self.update_frequency == 0 and self.change:
                        self.change = False
                        self.save_aspect_map()
        except Exception:
            self.save_aspect_map()
            return
        self.save_aspect_map()

    def chunk_phrases(self, review: str) -> list[str]:
        """
        1. 将常见断句标点替换成"."
        2. 连续的多个点合并为单个"."
        3. 用空格切分，每个短语不含空格
        """
        # 1. 替换断句标点为 .
        review = re.sub(r"[。！？?!；;]", ".", review)
        # 2. 合并连续多个 . 为一个 .
        review = re.sub(r"\.{2,}", ".", review)
        # 3. 按空格切分，剔除空项
        phrases = [s.strip() for s in review.strip().split() if s.strip()]
        return phrases


def main(input_csv):
    """
    读取csv，统计各维度情感，输出结构化内容
    """
    # 路径配置
    KB_JSON = "aspect_map.json"
    AMT = AspectMapTrainer(KB_JSON)
    df = pd.read_csv(input_csv)
    assert 'reviewText' in df.columns, 'CSV必须含reviewText列'
    AMT.train(df)


if __name__ == "__main__":
    now_str = time.strftime("%Y%m%d_%H%M")
    INPUT_CSV = 'data/logs/top/top5pct_reviews_20250701_0004.csv'
    main(INPUT_CSV)
