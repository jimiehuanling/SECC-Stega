SECC-Stega的代码分为matlab部分和python部分。

matlab部分负责实现秘密信息的编码和解码。embeding.m负责编码，extracting_Adg.m、extracting_Bin.m、extracting_Flc.m、extracting_Vlc.m负责解码。

python部分负责实现隐写的嵌入和提取。子目录common下是定义的一些通用函数，control group下是对照组：no-adg.py、no-Bin.py、no-flc.py、no-vlc.py，experimental group下是实验组:adg.py、Bin-LM.py、rnn-stega-flc.py、rnn-stega-vlc.py，train model下是模型训练函数：train_model_adg.py、train_model_bin.py、train_model_flc.py、train_model_vlc.py。

以下是整个隐写流程演示（以adg算法为例）：
1、使用embeding.m生成LT编码后的秘密信息。程序运行结束之后会产生newbits.txt、index.txt文件。
2、使用adg.py（实验组）读取newbits.txt文件模拟隐写的嵌入和提取过程。程序运行结束后会产生经过三种文本攻击后提取到的秘密信息以及对应的index，即adg-changed.txt、adg-changed-index.txt、adg-deleted.txt、adg-deleted-index.txt、adg-swapped.txt、adg-swapped-index.txt文件。
3、使用no-adg.py（对照组）读取newbits.txt文件模拟隐写的嵌入和提取过程。程序运行结束后会输出经过三种文本攻击后提取到的秘密信息的误码率：ber1、ber2、ber3。
4、使用extracting_Adg.m读取adg-changed.txt、adg-changed-index.txt、adg-deleted.txt、adg-deleted-index.txt、adg-swapped.txt、adg-swapped-index.txt文件对编码信息进行解码还原出原始的秘密信息。程序运行结束后会输出经过三种文本攻击后提取到的秘密信息的误码率：BERating、BERating1、BERating2。
