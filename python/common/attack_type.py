import random
import string


# 对生成的文本进行篡改攻击
def str_change(original_string):
    # 生成一个随机索引
    random_index = random.randint(1, len(original_string) - 1)
    # 生成一个随机字符
    random_char = random.choice(string.ascii_letters)
    # 将随机字符替换原始字符串中的随机索引位置的字符
    new_string = original_string[:random_index] + random_char + original_string[random_index + 1:]
    # print(f"原始字符串: {original_string}")
    # print(f"随机改变一个字符后的字符串: {new_string}")
    return new_string

# 对生成的文本进行删除攻击
def str_delete(original_string):
    # 检查字符串长度是否大于1
    if len(original_string) <= 1:
        return original_string
    # 随机选择一个要删除的字符的索引，范围是1到len(s)-1
    remove_index = random.randint(1, len(original_string) - 1)
    # 构建新的字符串，删除选中的字符
    new_string = original_string[:remove_index] + original_string[remove_index + 1:]
    
    return new_string

# 对生成的文本进行换位攻击
def str_swap(s):
    # 将字符串转换为列表，因为字符串是不可变的
    s_list = list(s)
    
    # 获取字符串的长度
    length = len(s_list)
    
    # 随机选择两个不同的位置
    pos1, pos2 = random.sample(range(length), 2)
    
    # 调换这两个位置的字符
    s_list[pos1], s_list[pos2] = s_list[pos2], s_list[pos1]
    
    # 将列表转换回字符串并返回
    return ''.join(s_list)


