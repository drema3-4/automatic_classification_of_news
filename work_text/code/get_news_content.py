def __parse_news__(url: str) -> str:
    # получаем html код страницы по ссылке на новость
    news_file_name = "news.html"
    __getPage__(url, news_file_name)
    # и сразу загружаем его из файла
    with open(news_file_name, encoding="utf-8") as file:
        src = file.read()
    # преобразуем html код к классам и сразу получаем всё текстовое содержание
    # новости. Это возможно так как весь контент новости содержится
    # в теге post__text
    content = BeautifulSoup(src, "lxml").find("div", class_="main").find(
        "div", class_="post__text"
    ).text.strip()
    # возвращаем полученное содержание новости в виде строки
    return content