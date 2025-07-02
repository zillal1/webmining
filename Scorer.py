import pandas as pd
import numpy as np
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os
import time
import json
import math


class EmotionalAnalysisHelper:
    def __init__(self, output_json,aspect_map_path):
        # 1. 维度设计与关键词mapping（如需针对你们产品可更换）
        self.output_json = output_json
        with open(aspect_map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 保证set类型
            self.ASPECT_MAP = {k: set(v) for k, v in data.items() if k != 'others'}
        self.ALL_ASPECTS = list(self.ASPECT_MAP.keys())

        # 2. 初始化VADER
        # NOTE:第一次运行需要运行下方被注释代码
        # nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()

    def extract_aspect_phrases(self, review):
        """
        针对一条评论, 返回该文本内涉及到的aspect及其短语窗口
        """
        aspects_found = {}
        for aspect, keywords in self.ASPECT_MAP.items():
            for kw in keywords:
                # 忽略大小写，找所有出现位置
                pat = r'\b%s\b' % re.escape(kw)
                matches = list(re.finditer(pat, review.lower()))
                for m in matches:
                    left = max(0, m.start() - 80)
                    right = min(len(review), m.end() + 80)
                    snippet = review[left:right].replace('\n', ' ').strip()
                    if aspect not in aspects_found:
                        aspects_found[aspect] = []
                    aspects_found[aspect].append((kw, snippet))
        return aspects_found

    def split_into_clauses(self, text: str) -> list[str]:
        """
        将文本按“句末符号序列”切成子句。
        可识别组合与重复：??!!、?!、...、…… 等。
        """
        _SENT_END_RE = re.compile(r'[.!?。！？…]+')
        clauses = _SENT_END_RE.split(text)
        return [c.strip() for c in clauses if c.strip()]

    def analyze_aspects(self, df: pd.DataFrame):
        """
        子句级情感 + review 级去重 + spam & helpful 加权
        """
        aspect_stats = {a: {"count": 0,
                            "sentiment_sum": 0.0,
                            "weight_sum": 0.0,
                            "examples": []}
                        for a in self.ALL_ASPECTS}

        for idx, row in df.iterrows():
            review = str(row["reviewText"])
            clauses = self.split_into_clauses(review)

            # ——— Weights Computation ———
            spam_p = float(row.get("spam_proba", 0.5))  # 缺失视为 0.5
            '''
            Quality weight, if a comment is more likely a good comment, it will have a higher weight.
            '''
            quality_w = max(0.0, min(1.0, 1.0 - spam_p))  # clamp to [0,1]
            '''
            popularity weight: the numerical formula is:
            w = exp(α*(Good-Down))
            We choose to use exponential function because we want to demonstrate that if one comment gains
            sufficient supports, it will have a predominant advantage. Since we assume that bad comment has
            been filtered, the remaining comments with higher supports must be great comments that can provide
            a lot of useful information.  
            '''
            up, down = eval(row['helpful'])  # 容错
            popularity_w = math.exp(
                0.008 * ((up or 0) - (
                        down or 0)))  # If the number of "UP" is greatly large, it will dominate the judging
            '''
            OverallWeight=PopularityWeight * QualityWeight
            '''
            weight_review = quality_w * popularity_w  # ≥0

            emotional_w = float(row.get("overall"))
            # 当前 review 中命中的 aspect -> [(sent, snippet), ...]
            aspects_in_review = {}

            for clause in clauses:
                '''
                The final formula of sentiment score:
                s = SentimentOfWord+0.1*OverallRating-0.15
                Why is 0.15?
                If we regard 3 stars as a neutral comment, it should have no emotional bias, which means we need
                to set the emotional weight to 0.
                Also, consider a bad comment might contain some good parts of a product, so the emotional weight CANNOT 
                completely determine whether one aspect is good or not. So we set a relatively less weight 0.05 instead of 0.2
                '''
                sent_score = self.sia.polarity_scores(clause)['compound'] + 0.05 * emotional_w - 0.15
                clause_aspects = self.extract_aspect_phrases(clause)
                for aspect, hits in clause_aspects.items():
                    aspects_in_review.setdefault(aspect, []).append(
                        (sent_score, clause)  # 直接用子句
                    )

            # ——— 写入全局统计 ———
            for aspect, lst in aspects_in_review.items():
                aspect_stats[aspect]["count"] += 1  # 仍按 review 计次

                sent_mean_this_review = np.mean([s for s, _ in lst])
                aspect_stats[aspect]["sentiment_sum"] += sent_mean_this_review * weight_review
                aspect_stats[aspect]["weight_sum"] += weight_review

                # 代表子句：本 review 中情感绝对值最大的那一句
                best_sent, best_snip = max(lst, key=lambda x: abs(x[0]))
                if abs(best_sent) > 0.3:  # 跳过中性
                    aspect_stats[aspect]["examples"].append({
                        "snippet": best_snip,
                        "sentiment": round(best_sent, 3),
                        "weight": weight_review,
                        "review_id": row.get("review_id", idx)
                    })

        # ——— 汇总输出 ———
        results = []
        for aspect, stat in aspect_stats.items():
            if stat["count"] == 0 or stat["weight_sum"] == 0:
                continue

            avg_sent = stat["sentiment_sum"] / stat["weight_sum"]

            # filter neutral samples
            pos_cands = [e for e in stat["examples"] if e["sentiment"] > 0.3]
            neg_cands = [e for e in stat["examples"] if e["sentiment"] < -0.3]

            # score = weight * SentimentScore
            pos_example_snip = max(pos_cands, key=lambda e: e["weight"] * e["sentiment"])[
                "snippet"] if pos_cands else ""
            neg_example_snip = min(neg_cands, key=lambda e: e["weight"] * e["sentiment"])[
                "snippet"] if neg_cands else ""

            results.append({
                "aspect": aspect,
                "n_mention": stat["count"],
                "avg_sentiment": round(avg_sent, 3),
                "pos_example": pos_example_snip,
                "neg_example": neg_example_snip
            })

        return results

    def Do_Analysis(self, input_csv):
        """
        Read csv-> sentimental analysis->Output structural content.
        """
        df = pd.read_csv(input_csv)
        assert 'reviewText' in df.columns, 'CSV must contain reviewText column!'

        aspects_summary = self.analyze_aspects(df)
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(aspects_summary, f, ensure_ascii=False, indent=2)
        print(f"The analysis result has been saved in: {self.output_json}")


# 示例用法
if __name__ == "__main__":
    now_str = time.strftime("%Y%m%d_%H%M")
    INPUT_CSV = 'data/logs/top/top_reviews_20250701_2341.csv'
    ASPECT_MAP_JSON = 'aspect_map.json'
    OUTPUT_JSON = f"data/logs/sentiment/sentiment_result_{now_str}.json"
    Ana = EmotionalAnalysisHelper(OUTPUT_JSON,ASPECT_MAP_JSON)
    Ana.Do_Analysis(INPUT_CSV)

