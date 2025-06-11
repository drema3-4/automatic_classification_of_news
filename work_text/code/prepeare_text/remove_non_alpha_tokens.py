def __count_letters_in_token__(self, token: str) -> int:
    num_letters = 0
    for symb in token:
        if ("a" <= symb and symb <= "z") or ("A" <= symb and symb <= "Z"):
            num_letters += 1
        if ("à" <= symb and symb <= "ÿ") or ("À" <= symb and symb <= "ß"):
            num_letters += 1
    return num_letters
def __strip_non_letters__(self, text: str) -> str:
    return re.sub(r"^[^a-zA-Zà-ÿÀ-ß¸¨]+|[^a-zA-Zà-ÿÀ-ß¸¨]+$", "", text)
def __processing_token__(self, token: str) -> str:
    new_token = self.__strip_non_letters__(token)
    return new_token if (self.__count_letters_in_token__(new_token) +
                            1.0) / (len(new_token) + 1.0) >= 0.5 else ""