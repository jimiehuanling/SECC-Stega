import random
import string
import numpy as np
# 二进制转十进制
def binary_to_decimal(binary_string):
    decimal_number = int(binary_string, 2)
    return decimal_number

# 十进制转二进制
def decimal_to_binary(decimal_number):
    binary_string  = format(decimal_number, '05b')
    return binary_string

# 奇校验编码
def parity_check_ji(bit_string):
    ones_count = bit_string.count('1')
    if ones_count % 2 == 0:
        return bit_string + '1'  
    else:
        return bit_string + '0'  
    
# 偶校验编码
def parity_check_ou(bit_string):
    ones_count = bit_string.count('1')
    if ones_count % 2 == 0:
        return bit_string + '0'  
    else:
        return bit_string + '1'  
    
# 偶校验
def check_parity(bit_string):
    # 计算前4位的1的个数
    num_ones = bit_string[:-1].count('1')
    # 获取偶校验位
    parity_bit = bit_string[-1]
    
    if parity_bit == '0':
        return num_ones % 2 == 0
    elif parity_bit == '1':
        return num_ones % 2 == 1
    else:
        return False
    
def validate_parity_codes(bit_strings):
    # 对每个字符串进行偶校验
    useful_str = []
    indexofstr = []
    for i, str in enumerate(bit_strings):
        if check_parity(str):
            useful_str.append(str)
            indexofstr.append(i)
    return useful_str, indexofstr

    
# 读取LT编码结果并进行奇偶校验
def loadandparity(file_path='newbits.txt'):
    # 读取LT编码结果
    with open(file_path, 'r') as file:
        data = file.read()
        data = data.replace('\n','')
        matrix = [data[i:i+4] for i in range(0, len(data), 4)]    
        
        parity_matrix = []
        # 添加奇偶校验位，这里采用的是偶校验
        for bits in matrix:
            new_bits = parity_check_ou(bits)
            parity_matrix.append(new_bits)
        
        return parity_matrix

# 对生成的文本进行篡改攻击
def attack(original_string):
    # 生成一个随机索引
    random_index = random.randint(1, len(original_string) - 1)
    # 生成一个随机字符
    random_char = random.choice(string.ascii_letters)

    # 将随机字符替换原始字符串中的随机索引位置的字符
    new_string = original_string[:random_index] + random_char + original_string[random_index + 1:]
    # print(f"原始字符串: {original_string}")
    # print(f"随机改变一个字符后的字符串: {new_string}")
    return new_string


# 针对numpy.array的数组元素查找
def find_char_index(char_array, target_char):
    # 使用 numpy.where 查找目标字符的索引
    result = np.where(char_array == target_char)[0]
    if result.size > 0:
        return result[0]
    else:
        return 1


def save_bit_strings_to_file(bit_strings, file_name):
    with open(file_name, 'w') as file:
        for bit_string in bit_strings:
            file.write(bit_string + '\n')

def save_numeric_array_to_file(numeric_array, file_name):
    with open(file_name, 'w') as file:
        for number in numeric_array:
            file.write(f"{number}\n")

def save_string_to_file(string, file_name):
    with open(file_name, 'w') as file:
        file.write(string)

def count_different_characters(str1, str2):
    # 检查两个字符串是否长度相等
    if len(str1) != len(str2):
        raise ValueError("字符串长度不相等")
    
    # 计算相同位置上不同字符的数量
    count = sum(1 for a, b in zip(str1, str2) if a != b)
    
    return count

if __name__ == '__main__':
    
    
    
    pass
