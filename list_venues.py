import csv
from collections import Counter

counts = Counter()
with open('EuroCrypt_citations_matched.csv', 'r') as f:
    reader = csv.reader(f, escapechar='\\')
    header = next(reader)
    print("Headers:", header)
    for row in reader:
        if len(row) >= len(header):
            venue_matched = row[header.index('venue_matched')]
            counts[venue_matched] += 1

for venue, count in counts.most_common(50):
    print(f"{count:4d} {venue}")