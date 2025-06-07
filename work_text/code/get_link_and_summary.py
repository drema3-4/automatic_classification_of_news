# получение html кода страницы из файла
with open(page_file_name, encoding="utf-8") as file:
        src = file.read()
# преобразование html кода в классы
soup = BeautifulSoup(src, "lxml")
# переход к содержимому новости, которое находится
# в теге div с классом post
news = soup.find("div", class_="post")
try:
    # получение текста ссылки из соответствующего тега
    link = news.find("h2", class_="first_child").find("a").get("href")
    # не все ссылки в теге сохранены полностью, данный
    # код добавляет обрезанную часть
    if not link.startswith("https://"):
        link = 'https://www.hse.ru' + link
except:
    link = ""
try:
    # получение краткого описания новости из соответствующего тега
    news_short_content = news.find("p", class_="first_child").find_next_sibling("p").text.strip()
except:
    news_short_content = ""