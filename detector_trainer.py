# train_spam_detector.py
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from WM.model.nn import TextCNN
from WM.utils.review import ReviewDataset

import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


class DetectorTrainer:
    def __init__(self, config: dict, csv_path, model_path):
        # ---------------- parameters ---------------- #
        self.best_thr = 0
        self.VOCAB_SIZE = config['VOCAB_SIZE']
        self.MAX_LEN = config['MAX_LEN']
        self.EMB_DIM = config['EMB_DIM']
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.EPOCHS = config['EPOCHS']
        self.LR = config['LR']
        self.DEVICE = config['DEVICE']
        # -------------1.read data-------------------- #
        self._read_data(csv_path)
        # -------------2.build vocabulary table--------#
        # 2. 构建词表
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = build_vocab_from_iterator(self._yield_tokens(self.train_text),
                                               max_tokens=self.VOCAB_SIZE,
                                               specials=["<pad>", "<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.PAD_IDX = self.vocab["<pad>"]
        # -----------------3.Build Dataset-----------------#
        self._dataset_constructor()
        # ------------------4.Initialize nn model ----------#
        self._initialize_nn_model()
        self.save_path = model_path

    def _read_data(self, csv_path):
        df = pd.read_csv(csv_path).dropna(subset=["reviewText", "label"])
        print(len(df))
        df = df[:1000]
        self.train_text, self.test_text, self.y_train, self.y_test = train_test_split(
            df["reviewText"].astype(str),
            df["label"].astype(int).values,
            test_size=0.2,
            random_state=42,
            stratify=df["label"]
        )

    def _yield_tokens(self, text_series):
        for txt in text_series:
            yield self.tokenizer(txt)

    def text_to_tensor(self, text):
        tokens = self.tokenizer(text)[:self.MAX_LEN]
        ids = self.vocab(tokens)
        if len(ids) < self.MAX_LEN:
            ids += [self.PAD_IDX] * (self.MAX_LEN - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def _dataset_constructor(self):
        # 3. 自定义 Dataset
        self.train_ds = ReviewDataset(self.train_text, self.y_train, self.text_to_tensor)
        self.test_ds = ReviewDataset(self.test_text, self.y_test, self.text_to_tensor)
        self.train_dl = DataLoader(self.train_ds, batch_size=self.BATCH_SIZE, shuffle=True)
        self.test_dl = DataLoader(self.test_ds, batch_size=self.BATCH_SIZE)

    def _initialize_nn_model(self):
        # 4. CNN 模型
        self.model = TextCNN(len(self.vocab), self.EMB_DIM, self.PAD_IDX).to(self.DEVICE)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)

    def train(self):
        # 5. 训练循环
        for epoch in range(1, self.EPOCHS + 1):
            self.model.train()
            total_loss = 0
            for x, y in self.train_dl:
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(x)
            print(f"Epoch {epoch}/{self.EPOCHS}  Train Loss: {total_loss / len(self.train_ds):.4f}")

        self.best_thr, best_f1 = self.find_best_threshold(self.model, self.test_dl, self.DEVICE)
        print(f"Best F1={best_f1:.4f} @ threshold={self.best_thr:.4f}")
        self.save_checkpoint()

    # ========= ① 计算最佳阈值 =========
    def find_best_threshold(self, model, dataloader, device="cpu"):
        """
        在 dataloader 上扫描所有阈值，返回使 F1 分数最大的阈值。
        你也可以把 metric 参数改写成自定义逻辑。
        返回: best_thr (float), best_f1 (float)
        """
        model.eval()
        probs, labels = [], []

        with torch.no_grad():
            for x, y in dataloader:
                probs.extend(model(x.to(device)).cpu().numpy())
                labels.extend(y.numpy())

        probs = np.asarray(probs)
        labels = np.asarray(labels)

        # Precision-Recall 曲线
        prec, rec, thr = precision_recall_curve(labels, probs)
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)

        idx = f1.argmax()
        best_thr = float(thr[idx])
        best_f1 = float(f1[idx])
        return best_thr, best_f1

    # ========= save model and threshold =========
    def save_checkpoint(self):
        """
        把模型、词表、padding 索引、max_len、阈值统一保存
        """
        # 7. 保存模型 + 词表
        now_str = time.strftime("%Y%m%d_%H%M")
        SAVE_PATH = f"{self.save_path}_{now_str}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "vocab": self.vocab,
                "pad_idx": self.PAD_IDX,
                "max_len": self.MAX_LEN,
                "threshold": self.best_thr
            },
            SAVE_PATH
        )
        print(f"Checkpoint saved to {SAVE_PATH} (threshold = {self.best_thr:.4f})")

    def eval(self):
        # the path to save results with time
        now_str = time.strftime("%Y%m%d_%H%M%S")
        pict_dir = os.path.join("result/pict", now_str)
        os.makedirs(pict_dir, exist_ok=True)

        self.model.eval()
        all_pred, all_true, all_prob = [], [], []
        with torch.no_grad():
            for x, y in self.test_dl:
                out = self.model(x.to(self.DEVICE)).cpu().numpy()
                prob = out.flatten()
                pred = (prob > self.best_thr).astype(int)
                all_pred.extend(pred.tolist())
                all_true.extend(y.numpy().tolist())
                all_prob.extend(prob.tolist())

        print(classification_report(all_true, all_pred))

        # Confusion Matrix
        cm = confusion_matrix(all_true, all_pred)
        cmd = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots()
        cmd.plot(ax=ax)
        plt.title("Confusion Matrix")
        fig.savefig(os.path.join(pict_dir, "confusion_matrix.png"))
        plt.close(fig)

        # ROC Curve
        fpr, tpr, _ = roc_curve(all_true, all_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        fig.savefig(os.path.join(pict_dir, "roc_curve.png"))
        plt.close(fig)

        # PR Curve
        prec, recall, _ = precision_recall_curve(all_true, all_prob)
        ap_score = average_precision_score(all_true, all_prob)
        fig, ax = plt.subplots()
        ax.plot(recall, prec, label="AP={:.2f}".format(ap_score))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc='lower left')
        fig.savefig(os.path.join(pict_dir, "pr_curve.png"))
        plt.close(fig)

        print(f"\n[Info] Pictures saved to {pict_dir}/")


def run():
    CONFIG = {
        'VOCAB_SIZE': 20_000,
        'MAX_LEN': 150,
        'EMB_DIM': 128,
        'BATCH_SIZE': 64,
        'EPOCHS': 12,
        'LR': 1e-3,
        'DEVICE': "cuda" if torch.cuda.is_available() else "cpu"
    }
    csv_path = 'data/training_data/reviews.csv'
    model_path = 'model/saved_models/spam_cnn'
    trainer = DetectorTrainer(CONFIG, csv_path, model_path)
    trainer.train()
    trainer.eval()


if __name__ == '__main__':
    run()
