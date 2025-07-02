import pandas as pd
import sys
import json


def jsonlines_to_csv(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"转换完成，结果保存为 {csv_path}")


if __name__ == "__main__":
    # 用法：python jsonlines_to_csv.py input.json output.csv
    if len(sys.argv) != 3:
        print("用法: python jsonlines_to_csv.py input.json output.csv")
    else:
        jsonlines_to_csv(sys.argv[1], sys.argv[2])
