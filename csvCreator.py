import nltk
import math
import csv
import numpy as np

sent_array = []
for book_name in nltk.corpus.gutenberg.fileids():
    i = 0
    n = 150
    t1 = ()
    while i <= math.floor(len((nltk.corpus.gutenberg.raw(book_name)).split(" "))):
        x = (nltk.corpus.gutenberg.raw(book_name).split()[i: i + n])
        x1 = " ".join(x)
        t1 = (x1, book_name.split("-")[0])
        sent_array.append(t1)
        i = i + n

with open('file.csv', 'w') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['Sentence', 'Author'])
    csv_out.writerows(sent_array)


np.random.shuffle(sent_array)
