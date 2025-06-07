def __parse_news__(url: str) -> str:
    # �������� html ��� �������� �� ������ �� �������
    news_file_name = "news.html"
    __getPage__(url, news_file_name)
    # � ����� ��������� ��� �� �����
    with open(news_file_name, encoding="utf-8") as file:
        src = file.read()
    # ����������� html ��� � ������� � ����� �������� �� ��������� ����������
    # �������. ��� �������� ��� ��� ���� ������� ������� ����������
    # � ���� post__text
    content = BeautifulSoup(src, "lxml").find("div", class_="main").find(
        "div", class_="post__text"
    ).text.strip()
    # ���������� ���������� ���������� ������� � ���� ������
    return content