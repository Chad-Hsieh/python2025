import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def fetch_html():
    url = "https://water.taiwanstat.com/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

def parse_data(html):
    soup = BeautifulSoup(html, "html.parser")
    reservoirs = soup.find_all("div", class_="reservoir")

    names = []
    values = []

    for res in reservoirs:
        name_tag = res.find("h3")
        name = name_tag.text.strip() if name_tag else "未知水庫"

        value_tag = res.find("div", class_="volumn")
        if value_tag:
            h5_tag = value_tag.find("h5")
            if h5_tag:
                value_text = h5_tag.text.strip()
                if "有效蓄水量：" in value_text:
                    value = value_text.split("有效蓄水量：")[1].replace("萬立方公尺", "").replace(",", "").strip()
                elif "萬立方公尺" in value_text:
                    value = value_text.replace("萬立方公尺", "").replace(",", "").strip()
                else:
                    value = "N/A"
            else:
                value = "N/A"
        else:
            value = "N/A"

        names.append(name)
        values.append(value)

    return names, values

def plot_data(names, values):
    try:
        values = [float(v) if v != "N/A" else 0 for v in values]
    except ValueError:
        values = [0] * len(values)

    plt.figure(figsize=(10, 6))
    plt.bar(names, values, color='skyblue')
    plt.ylabel("有效蓄水量（萬立方公尺）")
    plt.xlabel("水庫名稱")
    plt.title("台灣各水庫有效蓄水量")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    html = fetch_html()
    names, values = parse_data(html)
    plot_data(names, values)

if __name__ == "__main__":
    main()

"""
完成時間順序
012
003 023
002
006
026
025 022 010 016 020 009 018 027
014
024 我
015
019 021
004
013 011
005
"""