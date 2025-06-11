with open(page_file_name, encoding="utf-8") as file:
        src = file.read()
soup = BeautifulSoup(src, "lxml")
news = soup.find("div", class_="post")
try:
    link = news.find("h2", class_="first_child").find("a").get("href")
    if not link.startswith("https://"):
        link = 'https://www.hse.ru' + link
except:
    link = ""
try:
    news_short_content = news.find("p", class_="first_child").find_next_sibling("p").text.strip()
except:
    news_short_content = ""