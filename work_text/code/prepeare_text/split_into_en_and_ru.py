def __first_is_en__(self, cell: str) -> bool:
    index_first_en = re.search(r"[a-zA-Z]", cell)
    index_first_ru = re.search(r"[à-ÿÀ-ß]", cell)
    return True if index_first_en and ( not (index_first_ru) or
        index_first_en.start() < index_first_ru.start() ) else False
def __split_into_en_and_ru__(self, cell: str) -> list[(bool, str)]:
    parts = []
    is_en = self.__first_is_en__(cell)
    part = ""
    for symb in cell:
        if is_en == (symb in string.ascii_letters) or not (symb.isalpha()):
            part += symb
        else:
            parts.append((is_en, part))
            part = symb
            is_en = not (is_en)
    if part:
        parts.append((is_en, part))
    return parts