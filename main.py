# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import jieba

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode:"+"/".join(seg_list))

fr = open("./data/80w.txt", 'r')

arrayOfLines = fr.readlines()

for line in arrayOfLines:
    line = line.strip()
    line = line.split('\t')
    print(line)
#      if line[1] == '1':
#         writeStr(line[2],'rubbishMsg.txt')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
