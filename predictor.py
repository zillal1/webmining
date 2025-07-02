import torch, numpy as np, pandas as pd, time
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from WM.model.nn import TextCNN
from WM.utils.review import ReviewSet


class PredictorHelper:
    def __init__(self, CKPT_PATH, INPUT_CSV, strategy=0):
        # ------------ path & device -------------
        self.CKPT_PATH = CKPT_PATH  # 用你的新模型名
        self.INPUT_CSV = INPUT_CSV
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # ------------------------------------- #

        # ========== model loader ==========
        self.ckpt = torch.load(CKPT_PATH, map_location=self.DEVICE)
        self.vocab = self.ckpt["vocab"]
        self.PAD_IDX = self.ckpt["pad_idx"]
        self.MAX_LEN = self.ckpt["max_len"]
        self.THR = self.ckpt.get("threshold", 0.5)  # 兼容老模型
        self.EMB_DIM = self.ckpt.get("emb_dim", 128)  # 新版如保存维度则直接用，否则128

        self.tokenizer = get_tokenizer("basic_english")

        self.model = TextCNN(len(self.vocab), self.EMB_DIM, self.PAD_IDX).to(self.DEVICE)
        self.model.load_state_dict(self.ckpt["model_state"])
        self.model.eval()

        try:
            assert 100 >= strategy >= 0
            self.strategy = strategy
        except AssertionError:
            print('''The strategy should be a non-negative number, 0 represents save all the reviews that has a proba < threshold, 
            while other positive numbers n denote saving the top-n reviews that's least likely to be a reviewer spam.  
            ''')
            self.strategy = 0

    def text_to_tensor(self, text):
        ids = self.vocab(self.tokenizer(text)[:self.MAX_LEN])
        ids += [self.PAD_IDX] * (self.MAX_LEN - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def inference(self):
        # ========== inference ==========
        df = pd.read_csv(self.INPUT_CSV).dropna(subset=["reviewText"])
        df = df[1200000:1201000]  # TODO:Demo用，正式跑全体可删除

        ds = ReviewSet(df["reviewText"].astype(str).tolist(), item_func=self.text_to_tensor)
        dl = DataLoader(ds, batch_size=128)

        proba = []
        with torch.no_grad():
            for batch in dl:
                proba.extend(self.model(batch.to(self.DEVICE)).cpu().numpy())
        df["spam_proba"] = proba

        # ========== save model according to time =============
        now_str = time.strftime("%Y%m%d_%H%M")

        if self.strategy == 0:
            out = f"data/logs/top/top_reviews_{now_str}.csv"
            df_best = df[df['spam_proba'] < self.THR]
            df_best.to_csv(out, index=False)
            print(f"The predictive positive comment has been saved at: {out}. Threshold: {self.THR:.4f}")
        else:
            # ========== save best n% only ==========
            out = f"data/logs/top/topNpct_reviews_{now_str}.csv"
            pct_n = np.percentile(df["spam_proba"], self.strategy)
            best_mask = df["spam_proba"] <= pct_n
            df_best = df[best_mask]
            print(
                f"The best {self.strategy}% have been saved at {out} ({len(df_best)}/{len(df)} rows, proba ≤{pct_n:.4f}).")


if __name__ == '__main__':
    ph = PredictorHelper(CKPT_PATH='model/saved_models/spam_cnn_20250630_1758_100w.pt',
                         INPUT_CSV='data/training_data/amazon_spam_train.csv')
    ph.inference()
