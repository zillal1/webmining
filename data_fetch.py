# amazon_to_kaggle_format.py
"""
将 Amazon 评论抓取并保存为 Kaggle 数据集相同结构
用法：python data_fetch.py B07FZ8S74R Toys_and_Games 600
"""

import csv, sys, time, random, re, requests
from datetime import datetime
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
HELPFUL_RE = re.compile(r"(\d[\d,]*)")  # 提取 “123 people found this helpful”


def parse_votes(helpful_text: str):
    """
    Amazon 页面只给正向 helpful；这里按 Kaggle 结构返回 [down, up]
    """
    if not helpful_text:
        return [0, 0]
    m = HELPFUL_RE.search(helpful_text)
    up = int(m.group(1).replace(",", "")) if m else 0
    return [0, up]  # down 票无法获取，填 0


def normalise_date(raw_date: str):
    """
    Amazon: 'Reviewed in the United States on July 13, 2014'
    输出 MM.DD.YYYY
    """
    try:
        month_day_year = raw_date.rsplit(" on ", 1)[-1]  # July 13, 2014
        d = datetime.strptime(month_day_year, "%B %d, %Y")
        return d.strftime("%m.%d.%Y")
    except Exception:
        return ""


def crawl_one_asin(asin, category, limit):
    url_tpl = f"https://www.amazon.com/product-reviews/{asin}/?pageNumber={{}}&sortBy=recent"
    print(url_tpl)
    page, collected = 1, 0
    with open("kaggle_format_reviews.csv", "w", newline="", encoding="utf8") as f:
        fieldnames = ["reviewerID", "reviewerName", "votes-down/up",
                      "reviewText", "rating", "summary",
                      "reviewTime", "category", "class"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while collected < limit:
            print(f"[{asin}] page {page}")
            r = requests.get(url_tpl.format(page), headers=HEADERS, timeout=15)
            if r.status_code != 200:
                print("Blocked or finished.")
                break
            soup = BeautifulSoup(r.text, "lxml")
            blocks = soup.select("div[data-hook='review']")
            if not blocks:
                break
            for b in blocks:
                collected += 1
                review_id = b.get("id") or f"{asin}_{page}_{collected}"
                name_tag = b.select_one("span.a-profile-name")
                title_tag = b.select_one("a[data-hook='review-title']")
                rating_tag = b.select_one("i[data-hook='review-star-rating']")
                body_tag = b.select_one("span[data-hook='review-body']")
                date_tag = b.select_one("span[data-hook='review-date']")
                helpful_tag = b.select_one("span[data-hook='helpful-vote-statement']")

                row = {
                    "reviewerID": review_id,
                    "reviewerName": (name_tag.text.strip() if name_tag else ""),
                    "votes-down/up": parse_votes(helpful_tag.text.strip() if helpful_tag else ""),
                    "reviewText": (body_tag.get_text(" ", strip=True) if body_tag else ""),
                    "rating": int(float(rating_tag.text.split()[0])) if rating_tag else "",
                    "summary": (title_tag.text.strip() if title_tag else ""),
                    "reviewTime": normalise_date(date_tag.text.strip() if date_tag else ""),
                    "category": category,
                    "class": -1  # 未标注
                }
                writer.writerow(row)
                if collected >= limit:
                    break
            page += 1
            time.sleep(random.uniform(1, 3))


if __name__ == "__main__":
    asin = sys.argv[1]  # e.g. B07FZ8S74R
    category = sys.argv[2]  # e.g. Toys_and_Games
    limit_num = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    crawl_one_asin(asin, category, limit_num)
    print("Done! 输出文件: kaggle_format_reviews.csv")
