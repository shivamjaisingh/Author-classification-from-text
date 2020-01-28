import pandas as pd
import scipy as sp
import nltk
import math
import csv

count = 1
for i in nltk.corpus.gutenberg.fileids():
    print(i.split("-")[0])



print(math.floor((len((nltk.corpus.gutenberg.raw('carroll-alice.txt')).split(" ")))/150))


#print(nltk.corpus.gutenberg.raw('carroll-alice.txt'))

sent_array = []


# for i in range(0, math.floor((len((nltk.corpus.gutenberg.raw('carroll-alice.txt')).split(" ")))/150)):

i = 0
n = 150
t1 = ()

while i <= math.floor(len((nltk.corpus.gutenberg.raw('carroll-alice.txt')).split(" "))):
    x = (nltk.corpus.gutenberg.raw('carroll-alice.txt').split()[i: i+150])
    x1 = " ".join(x)
    t1 = (x1, "Lewis")
    sent_array.append(t1)
    i = i+150

for r in sent_array:
    print(r)


# print(sent_array[0][0])


with open('file.csv', 'w') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['Sentence', 'Author'])
    for row in sent_array:
        csv_out.writerow(row)





# print(26443%150)
# def stringy(string, n):
#     str_size = len(string)
#     if str_size % n != 0:
#         print("Invalid Input: String size is not divisible by n")
#         return
#     part_size = str_size / n
#     k = 0
#     for i in string:
#         if k % part_size == 0:
#             print("\n")
#         print(i)
#         k += 1


#stringy(nltk.corpus.gutenberg.raw('carroll-alice.txt'),200)