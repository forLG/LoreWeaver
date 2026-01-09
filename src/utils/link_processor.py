import re
import json

class LinkProcessor:
    # 定义不视为“跳转链接”的格式化标签
    FORMATTING_TAGS = {
        'b', 'i', 'u', 's', 'c', 'color', 'note', 'bold', 'italic', 
        'strikethrough', 'underline', 'comic', 'book', 'filter'
    }

    @staticmethod
    def parse_and_clean(text: str):
        """
        解析文本：
        1. 移除标签语法，保留显示文本用于阅读。
        2. 提取有意义的跳转链接 (links)，并去除重复项。
        
        Returns:
            cleaned_text (str): 清洗后的文本
            links (list): 提取出的链接对象列表
        """
        if not isinstance(text, str):
            return "", []

        extracted_links = []

        # 辅助函数：处理单个正则匹配
        def replace_tag(match):
            full_tag = match.group(0)  # 例如: {@area map 2|021|x}
            tag_type = match.group(1)  # 例如: area
            content = match.group(2)   # 例如: map 2|021|x

            # 处理管道符分隔的属性
            parts = content.split('|')
            display_text = parts[0]    # 第一部分永远是显示文本
            attributes = parts[1:] if len(parts) > 1 else []

            # 如果不是纯格式化标签，则记录为链接
            if tag_type not in LinkProcessor.FORMATTING_TAGS:
                extracted_links.append({
                    "text": display_text,
                    "tag": tag_type,
                    "attrs": attributes
                })

            return display_text

        # 正则：匹配 {@tag content}
        # [^{}]+ 确保匹配最内层的标签，处理嵌套情况
        pattern = re.compile(r'{@(\w+)\s+([^{}]+)}')

        current_text = text
        # 循环处理直到没有标签为止（处理嵌套标签，如 {@note {@b bold} text}）
        while True:
            new_text, count = pattern.subn(replace_tag, current_text)
            if count == 0:
                break
            current_text = new_text

        # --- 去重逻辑 ---
        # 使用 JSON 序列化作为 key 来去重字典列表
        unique_links = []
        seen = set()
        for link in extracted_links:
            # 将字典转换为不可变的 JSON 字符串以便存入 set
            # sort_keys=True 确保属性顺序不同但内容相同的字典被视为相同（虽然这里顺序通常固定）
            link_signature = json.dumps(link, sort_keys=True)
            if link_signature not in seen:
                seen.add(link_signature)
                unique_links.append(link)

        return current_text, unique_links