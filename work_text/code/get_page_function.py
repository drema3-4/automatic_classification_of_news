def __getPage__(url: str, file_name: str) -> None:
    # ��������� html ���� �������� � ������� ���������� requests
    r = requests.get(url=url)
    # ���������� ����������� ���� � ��������� ����
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(r.text)