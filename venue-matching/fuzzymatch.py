import sys
import csv
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def main():

    writer = csv.writer(sys.stdout)

    dblp_venues = []
    with open('dblp-labels.csv', 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            dblp_venues.append(row[0])

    with open('mega-spreadsheet-clean.csv', 'r') as infile:
        reader = csv.reader(infile)
        next(reader)  
        for row in reader:
            url = row[7]
            venue = row[4]

            if venue in dblp_venues:
                writer.writerow(row)
                continue

            if 'arxiv' in url:
                writer.writerow(row)
                continue

            best_match, score = process.extractOne(venue, dblp_venues, scorer=fuzz.token_sort_ratio)

            if score >= 85:
                #print(best_match, " | ", venue)
                row[4] = best_match
                writer.writerow(row)

if __name__ == "__main__":
    main()
