def __parse_page__(page_file_name: str, news_container: pd.DataFrame) -> None:
    for i in range(10):
        try:
            if link.startswith("https://www.hse.ru/news/"):
                news_content = __parse_news__(link)
        except:
            news_content = ""
        if len(
            news_day + news_month + news_year + news_name + news_short_content +
            news_content
        ) > 0:
            news_container.loc[len(news_container.index)] = [
                link, news_date, news_name, news_short_content, news_content]
        news = news.find_next_sibling("div", class_="post")