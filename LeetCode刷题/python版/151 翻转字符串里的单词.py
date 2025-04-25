def reverseWords(s: str) -> str:
    # 去除多余空格
    s = s.strip()
    words = []
    word = ''
    for char in s:
        if char != ' ':
            word += char
        else:
            if word:  # 遇到空格且当前单词非空
                words.append(word)
                word = ''
    if word:  # 添加最后一个单词
        words.append(word)
    # 反转并拼接
    return ' '.join(reversed(words))