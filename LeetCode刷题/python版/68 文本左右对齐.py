def fullJustify(words: list[str], maxWidth: int) -> list[str]:
    res, line, line_length = [], [], 0

    for word in words:
        # 当前行能否加入新单词？
        if line_length + len(word) + len(line) <= maxWidth:
            line.append(word)
            line_length += len(word)
        else:
            # 处理非最后一行
            if len(line) == 1:
                res.append(line[0] + ' ' * (maxWidth - line_length))
            else:
                total_spaces = maxWidth - line_length
                space_per_gap, extra = divmod(total_spaces, len(line) - 1)
                # 构造当前行
                line_str = line[0]
                for i in range(1, len(line)):
                    line_str += ' ' * (space_per_gap + (1 if i <= extra else 0))
                    line_str += line[i]
                res.append(line_str)
            # 重置当前行
            line, line_length = [word], len(word)

    # 处理最后一行（左对齐）
    last_line = ' '.join(line)
    res.append(last_line + ' ' * (maxWidth - len(last_line)))
    return res