'''
Hierarchical Agglomerative Clustering to identify unique news items
'''
import re
import string
import pandas as pd
import csv

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords



with open('../data/output-2.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

def fp_steps(text):
    title = text.strip().lower()

    remove_spl_char_regex = re.compile('[%s]' % re.escape(string.punctuation)) # regex to remove special characters
    remove_num = re.compile('[\d]+')

    title_splchar_removed = remove_spl_char_regex.sub(" ", title)
    title_number_removed = remove_num.sub("", title_splchar_removed)
    words = title_number_removed.split()
    filter_stop_words = [w for w in words if not w in stopwords.words('english')]
    ps = PorterStemmer()
    stemed = [ps.stem(w) for w in filter_stop_words]
    return sorted(stemed)

def fingerprint(text):
    fp = " ".join(fp_steps(text))
    return fp


DISTANCE = 20   # Reduce
cluster = {}    # Reduce
cid = 0         # Reduce

f = open('../data/output-clusters.txt', 'w')

def levenshtein_distance(source, target):  # Reduce
    if source == target:
        return 0


    # Prepare a matrix
    slen, tlen = len(source), len(target)
    dist = [[0 for i in range(tlen+1)] for x in range(slen+1)]
    for i in range(slen+1):
        dist[i][0] = i
    for j in range(tlen+1):
        dist[0][j] = j

    # Counting distance, here is my function
    for i in range(slen):
        for j in range(tlen):
            cost = 0 if source[i] == target[j] else 1
            dist[i+1][j+1] = min(
                            dist[i][j+1] + 1,   # deletion
                            dist[i+1][j] + 1,   # insertion
                            dist[i][j] + cost   # substitution
                        )
    return dist[-1][-1]



list_stemmed_line=[]
list_line=[]

df = pd.read_csv('Data-OpsRisk.csv')

for index, row in df.iterrows():
    line = row['Content']
    list_stemmed_line.append(fingerprint(line))
    list_line.append(line.strip())

# for line in sys.stdin:
#     #print("%s\t%s" % (fingerprint(line),line.strip()))
#     list_stemmed_line.append(fingerprint(line))
#     list_line.append(line.strip())

# Merge the stemmed and unstemmed line together with a tab separator
list_merged=[]
for i in range(0, len(list_stemmed_line)):
    #print(list_stemmed_line[i], list_line[i])
    merged_with_tab = str(list_stemmed_line[i])+'\t'+str(list_line[i])
    list_merged.append(merged_with_tab)

list_merged = sorted(list_merged)


i=0                         # Reduce
for line in list_merged:    # Reduce
    cols = line.strip().split("\t")
    if i == 0:
        cluster[cid] = []
        cluster[cid].append(cols)
    else:
        last = cluster[cid][-1]
        if levenshtein_distance(last[0], cols[0]) <= DISTANCE:
            cluster[cid].append(cols)
        else:
            cid += 1
            cluster[cid] = []
            cluster[cid].append(cols)
    i+=1


for k, v in cluster.items():  # Reduce
    print("Cluster # ", k)
    f.write("Cluster #  "+"\t"+str(k))
    for entry in v:
        headline = entry[1]
        print(entry[1])
        f.write(entry[1])
        new_df = df.loc[df['Content'] == headline]
        date = new_df['Date'].values
        print(date)

        with open('../data/output-2.csv', 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            csv_writer.writerow([date, headline])


        break
    print('\n')
    f.write('\n')
f.close()





