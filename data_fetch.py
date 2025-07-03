import time
import json
import hashlib
import random
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")

service = Service(executable_path="C:/Users/ISCREAM/AppData/Local/Google/Chrome/Application/chrome.exe")  # ← 换成你的实际路径
driver = webdriver.Chrome(options=chrome_options)


def get_asin_from_url(url):
    if "/dp/" in url:
        return url.split("/dp/")[1].split('/')[0]
    elif "/product/" in url:
        return url.split("/product/")[1].split('/')[0]
    return ''


def generate_oid(text):
    return hashlib.md5(text.encode()).hexdigest()[:24]


def parse_helpful(text):
    # 例："23 people found this helpful"
    try:
        num = int(text.strip().split()[0])
        return [num, num]
    except:
        return [0, 0]


def parse_unix_time(time_str):
    # 亚马逊英文格式："Reviewed in the United States on March 7, 2013"
    # 输出：unix时间戳
    try:
        # 只取日期部分
        if "on" in time_str:
            date_part = time_str.split("on")[-1].strip()
        else:
            date_part = time_str.strip()
        dt = datetime.strptime(date_part, "%B %d, %Y")
        return int(time.mktime(dt.timetuple())), date_part
    except:
        return 0, ""


def review_to_json(review_elem, asin, category):
    # 提取用户名
    reviewer_name = review_elem.find_element(By.CLASS_NAME, 'a-profile-name').text.strip()
    # 评论正文
    review_text = review_elem.find_element(By.CLASS_NAME, 'review-text-content').text.strip()
    # 评论标题
    summary = review_elem.find_element(By.CLASS_NAME, 'review-title-content').text.strip()
    # 评分: "5.0 out of 5 stars"
    ov = review_elem.find_element(By.CLASS_NAME, 'review-rating').text.strip()
    overall = float(ov.split(' ')[0])
    # 时间
    review_time_str = review_elem.find_element(By.CLASS_NAME, 'review-date').text.strip()
    unixReviewTime, reviewTimeFmt = parse_unix_time(review_time_str)
    # 有用数
    try:
        helpful_txt = review_elem.find_element(By.CLASS_NAME, 'cr-vote-text').text.strip()
        helpful = parse_helpful(helpful_txt)
    except:
        helpful = [0, 0]
    # 伪造reviewerID与_id
    reviewerID = generate_oid(reviewer_name + asin + review_time_str + str(random.random()))
    oid = generate_oid(review_text + reviewer_name)
    # 整理输出
    return {
        "_id": {"$oid": oid},
        "reviewerID": reviewerID,
        "asin": asin,
        "reviewerName": reviewer_name,
        "helpful": helpful,
        "reviewText": review_text,
        "overall": overall,
        "summary": summary,
        "unixReviewTime": unixReviewTime,
        "reviewTime": reviewTimeFmt,
        "category": category
    }


def get_amazon_reviews(url, category):
    asin = get_asin_from_url(url)
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    time.sleep(2)

    # 跳转到所有评论页面
    try:
        all_reviews_link = driver.find_element(By.PARTIAL_LINK_TEXT, 'See all reviews')
        all_reviews_link.click()
        time.sleep(2)
    except:
        print("未找到进入评论区的链接，可能本页面已是评论页")

    results = []
    while True:
        time.sleep(1.5)
        reviews = driver.find_elements(By.XPATH, '//div[contains(@id,"customer_review-")]')
        for review_elem in reviews:
            try:
                dct = review_to_json(review_elem, asin, category)
                results.append(dct)
            except Exception as ex:
                print(f"解析评论失败，跳过。{str(ex)}")
        # 翻页
        try:
            next_btn = driver.find_element(By.CLASS_NAME, 'a-last')
            if 'a-disabled' in next_btn.get_attribute('class'):
                break
            next_btn.click()
        except:
            break
    driver.quit()
    return results


if __name__ == "__main__":
    # 输入示例
    url = input("请输入亚马逊商品链接：\n").strip()
    category = input("请输入商品分类(如Cell_Phones_and_Accessories)：\n").strip()
    reviews = get_amazon_reviews(url, category)
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    print("已保存所有评论至output.json")
