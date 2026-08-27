import csv
import math
from venueGroupings import *

# Global Variables Hard Coded For Checks
nonAcademicCitations = [
    "ePrint",
    "web",
    "arXiv",
    "GitHub",
    "Technological Standards and Reports"
]

targetApplicationMapping = {
    "OPI" : ['CNP', 'AUT', '1-BLK', 'ML', 'NWT', 'DAD'],
    "PDI" : ['CC', 'DCC', 'HWM'],
    "GOV" : ['DPI', 'SMG', 'DCC', 'SPG', 'PH', 'NIST', 'VOT', 'GVE', 'LE', 'NGO'],
    "PRD" : ['ADD', 'WEB', 'MSG', 'LOC', 'PWD', 'DLD', 'RS', 'SE', 'BMD'],
    "PEP" : ['JUR', 'ACT', 'MIN', 'PGTS', 'IPS'],
    'FIN' : ['6-BLK', 'BNK', 'CUR', 'ASC'],
    'CRY' : ['CRY'],
    'PQC' : ['PQC'],
}


# Helper function for setting up CSV, returns in 2D list.
def parseCSV(csvInput : str):
    data = csv.reader(open(csvInput, "r", encoding="utf-8"))
    data = list(data)
    data.pop(0)
    return data


# Data filtering, takes check str and location to check.
def filterData(inputData, comparisonStr : str, location : int):
    outputData = []
    for citation in inputData:
        if citation[location] == comparisonStr:
            outputData.append(citation)
    return outputData


def filterDataInclusiveList(inputData, comparisonList, location: int):
    outputData = []
    for citation in inputData:
        for comparisonStr in comparisonList:
            if comparisonStr in citation[location]:
                outputData.append(citation)
    return outputData


# Takes a dictionary with venue and count pairs. Filters data into groupings based on our grouping.
def filterToGroups(inputData, onlyAcademic = False):
    outputData = {}
    total = 0
    for groupingKey in venueGroupings.keys():
        outputData[groupingKey] = 0
    for venue in inputData.keys():
        failure = True
        count = 0
        for groupingKey in venueGroupings.keys():
            if venue in venueGroupings[groupingKey]:
                failure = False
                if not (onlyAcademic == False or not (groupingKey == "Web" or groupingKey == "Non-Research Paper")):
                    continue
                outputData[groupingKey] += inputData[venue]
                count += 1
                total += inputData[venue]
        if failure:
            print("FAILED TO FIND:", venue)
    return outputData, total


# Input 2D array formated to CSV data, counts the venues within the data and returns a dict with them.
def countVenues(inputData):
    venueCounter = {}
    for citation in inputData:
        if citation[7] in venueCounter:
            venueCounter[citation[7]] += 1
        else:
            venueCounter[citation[7]] = 1
    return venueCounter


# Takes a source and 2D array formated to our CSV data, calculates the probability squared.
def calculateConcentration(inputData, label):
    countOfLabelFromPaper = dict()
    totalOfLabel = 0
    for citation in inputData:
        if citation[7] == label:
            totalOfLabel += 1
            if citation[0] in countOfLabelFromPaper:
                countOfLabelFromPaper[citation[0]] += 1
            else:
                countOfLabelFromPaper[citation[0]] = 1
    totalProbSquared = 0
    for key in countOfLabelFromPaper:
        totalProbSquared += (countOfLabelFromPaper[key]/totalOfLabel)**2
    return totalProbSquared, countOfLabelFromPaper


# Takes a dictionary of key and number pairs. Orders based on largest numbers and returns top n and the total count.
def topNVenues(inputData, n, filterNonAcademic = False):
    dataSorted = [(v, k) for k, v in inputData.items()]
    dataSorted.sort(reverse=True)
    topNData = {}
    count = 0
    totalCount = 0
    for v,k in dataSorted:
        if (not filterNonAcademic or k not in nonAcademicCitations) and count < n:
            topNData[k] = v
            totalCount += v
            count += 1
        elif not filterNonAcademic or k not in nonAcademicCitations:
            totalCount += v
    return topNData, totalCount


# Takes a dict with labels and counts, outputs a Shannon's Entropy calculation, NOT normalized.
def calculateShannonsEntropy(inputData, base=2, normalize = True, countWeb = False):
    counts = {
        label: count
        for label, count in inputData.items()
        if count > 0 and (label != "web" or countWeb)
    }
    total = sum(counts.values())
    entropy = 0
    countOfLabels = 0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p, base)
            countOfLabels += 1
    if normalize:
        entropy /= math.log(countOfLabels, base)
    return entropy


# Diction printer to ease translation
def printDictionary(inputDict, n = 15):
    venuePrint = [(v, k) for k, v in inputDict.items()]
    venuePrint.sort(reverse=True)
    count = 1
    for v, k in venuePrint:
        print(count, "%s: %d" % (k, v))
        count += 1
        if count > n+1:
            break


# Methods for combining the analysis given a dataset.
def topNcitations(inputData, n, keyword):
    print("\n\nTop {0} Venues for {1}\n -------------------------".format(n, keyword))
    venueCounter = countVenues(inputData)
    entropy = calculateShannonsEntropy(venueCounter, 2, True)
    topVenues, total = topNVenues(venueCounter, n)
    printDictionary(topVenues, n)
    print("Total citations and entropy:", total, entropy)

def topNCitationsAcademicOnly(inputData, n, keyword):
    print("\n\nTop {0} Academic Citations for {1}\n -------------------------".format(n, keyword))
    venueCounter = countVenues(inputData)
    entropy = calculateShannonsEntropy(venueCounter, 2, True)
    topVenues, total = topNVenues(venueCounter, n, True)
    printDictionary(topVenues, n)
    print("Total citations and entropy:", total, entropy)

def doGroupedAnalysis(inputData, keyword):
    print("\n\nGrouped Analysis for citations in {0}\n----------------------------".format(keyword))
    venueCounter = countVenues(inputData)
    entropy = calculateShannonsEntropy(venueCounter, 2, True)
    groupedData, total = filterToGroups(venueCounter)
    printDictionary(groupedData)
    print("Total citations and entropy:", total, entropy)


def main():
    combinedData = parseCSV("csv/Combined_citations_matched.csv")
    combinedAnalysis = False
    VenueAnalysis = False
    applicationEngagementAnalysis = True
    targetApplicationAnalysis = False
    groupedAnalysis = False
    labelConcentration = True

    # Top 30 venues for all citations
    if combinedAnalysis:
        topNcitations(combinedData, 30, "Combined")
        topNCitationsAcademicOnly(combinedData, 15, "Combined")
        if groupedAnalysis:
            doGroupedAnalysis(combinedData, "Combined")

    # Top 12 sources and venues for each conference
    if VenueAnalysis:
        CryptoData = filterData(combinedData, "Crypto", 1)
        EuroCryptData = filterData(combinedData, "Euro", 1)
        USENIXData = filterData(combinedData, "USENIX", 1)
        SPData = filterData(combinedData, "Oakland", 1)

        topNcitations(CryptoData, 55, "Crypto")
        topNCitationsAcademicOnly(CryptoData, 12, "Crypto")
        topNcitations(EuroCryptData, 55, "EuroCrypt")
        topNCitationsAcademicOnly(EuroCryptData, 12, "EuroCrypt")
        topNcitations(USENIXData, 35, "USENIX")
        topNCitationsAcademicOnly(USENIXData, 12, "USENIX")
        topNcitations(SPData, 35, keyword="Oakland")
        topNCitationsAcademicOnly(SPData, 12, keyword="Oakland")

        if groupedAnalysis:
            doGroupedAnalysis(CryptoData, "Crypto")
            doGroupedAnalysis(EuroCryptData, "EuroCrypt")
            doGroupedAnalysis(USENIXData, "USENIX Data")
            doGroupedAnalysis(SPData, "SPData")

    # Top 24 Sources for each AA Level
    if applicationEngagementAnalysis:
        AE1Data = filterData(combinedData, "1", 2)
        AE2Data = filterData(combinedData, "2", 2)
        AE3Data = filterData(combinedData, "3", 2)
        AE4Data = filterData(combinedData, "4", 2)

        topNcitations(AE1Data, 45, "Application Agnostic")
        topNcitations(AE2Data, 24, "Application Gesturing")
        topNcitations(AE3Data, 24, "Application Aware")
        topNcitations(AE4Data, 35, "Application Motivated")

        if labelConcentration:
            valueAE1, countDictAE1 = calculateConcentration(AE1Data, 'web')
            valueAE2, countDictAE2 = calculateConcentration(AE2Data, 'web')
            valueAE3, countDictAE3 = calculateConcentration(AE3Data, 'web')
            valueAE4, countDictAE4 = calculateConcentration(AE4Data, 'web')
            print("Concentration of each AE level in order:", valueAE1, valueAE2, valueAE3, valueAE4)

    # Top 12 Sources for Each Top Level TA Group
    if targetApplicationAnalysis:
        openDigitalInfraData = filterDataInclusiveList(combinedData, targetApplicationMapping['OPI'], 3)
        proprietaryDigitalInfraData = filterDataInclusiveList(combinedData, targetApplicationMapping['PDI'], 3)
        governanceData = filterDataInclusiveList(combinedData, targetApplicationMapping['GOV'], 3)
        productData = filterDataInclusiveList(combinedData, targetApplicationMapping['PRD'], 3)
        peopleData = filterDataInclusiveList(combinedData, targetApplicationMapping['PEP'],3)
        financeData = filterDataInclusiveList(combinedData, targetApplicationMapping['FIN'],3)

        topNcitations(openDigitalInfraData, 12, "Open Digital Infra")
        topNcitations(proprietaryDigitalInfraData, 12, "Proprietary Digital Infra")
        topNcitations(governanceData, 12, "Governance ")
        topNcitations(productData, 12, "Products")
        topNcitations(peopleData, 12, "People and Communities")
        topNcitations(financeData, 12, "Finance")

if __name__ == "__main__":
    main()