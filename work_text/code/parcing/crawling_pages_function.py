def __crawling_pages__(start: int, end: int, news_container: pd.DataFrame, num_of_thread: int) -> pd.DataFrame:
    page_file_name = "page.html"
    for i in range(start, end + 1):
        try:
            __getPage__("https://www.hse.ru/news/page{0}.html".format(i), page_file_name)
            __parse_page__(page_file_name, news_container)
        except:
            continue