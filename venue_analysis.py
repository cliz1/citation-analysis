import csv


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
    "PRD" : ['ADD', 'WEB', 'MSG', 'LOC', 'BNK', 'PWD', 'DLD', 'RS', 'SE'],
    "PEP" : ['JUR', 'ACT', 'MIN', 'PGTS', 'IPS'],
    'FIN' : ['6-BLK', 'BNK', 'CUR', 'ASC'],
    'CRY' : ['CRY'],
    'PQC' : ['PQC'],
    'BMD' : ['BMD']
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


# Input 2D array formated to CSV data, counts the venues within the data and returns a dict with them.
def countVenues(inputData):
    venueCounter = {}
    for citation in inputData:
        if citation[7] in venueCounter:
            venueCounter[citation[7]] += 1
        else:
            venueCounter[citation[7]] = 1
    return venueCounter


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
    topVenues, total = topNVenues(venueCounter, n)
    printDictionary(topVenues, n)
    print("Total citations:", total)

def topNCitationsAcademicOnly(inputData, n, keyword):
    print("\n\nTop {0} Academic Citations for {1}\n -------------------------".format(n, keyword))
    venueCounter = countVenues(inputData)
    topVenues, total = topNVenues(venueCounter, n, True)
    printDictionary(topVenues, n)
    print("Total citations:", total)



def main():
    combinedData = parseCSV("csv/Combined_citations_matched.csv")
    combinedAnalysis = False
    VenueAnalysis = False
    applicationEngagementAnalysis = True
    targetApplicationAnalysis = False

    # Top 30 venues for all citations
    if combinedAnalysis:
        topNcitations(combinedData, 30, "Combined")
        topNCitationsAcademicOnly(combinedData, 15, "Combined")

    # Top 12 sources and venues for each conference
    if VenueAnalysis:
        CryptoData = filterData(combinedData, "Crypto", 1)
        EuroCryptData = filterData(combinedData, "Euro", 1)
        USENIXData = filterData(combinedData, "USENIX", 1)
        SPData = filterData(combinedData, "Oakland", 1)

        topNcitations(CryptoData, 12, "Crypto")
        topNCitationsAcademicOnly(CryptoData, 12, "Crypto")
        topNcitations(EuroCryptData, 12, "EuroCrypt")
        topNCitationsAcademicOnly(EuroCryptData, 12, "EuroCrypt")
        topNcitations(USENIXData, 12, "USENIX")
        topNCitationsAcademicOnly(USENIXData, 12, "USENIX")
        topNcitations(SPData, 12, keyword="Oakland")
        topNCitationsAcademicOnly(SPData, 12, keyword="Oakland")


    # Top 24 Sources for each AA Level
    if applicationEngagementAnalysis:
        AE1Data = filterData(combinedData, "1", 2)
        AE2Data = filterData(combinedData, "2", 2)
        AE3Data = filterData(combinedData, "3", 2)
        AE4Data = filterData(combinedData, "4", 2)

        topNcitations(AE1Data, 24, "Application Agnostic")
        topNcitations(AE2Data, 24, "Application Gesturing")
        topNcitations(AE3Data, 24, "Application Aware")
        topNcitations(AE4Data, 24, "Application Motivated")

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