import pandas as pd
from scholarly import scholarly

# Load your combined DataFrame (if not already loaded)
df = pd.read_csv("combined_papers.csv")
print(f"Loaded {len(df)} papers from 'combined_papers.csv'.")

# Define a list to keep track of failed rows
failed_rows = []

def extract_scholarly_metadata(title, index):
    if pd.isna(title):
        print(f"Skipping NaN title at index {index}.")
        return {
            "authors": "N/A",
            "abstract": "N/A",
            "venue": "N/A",
            "publish_date": "N/A",
            "url": "N/A",
            "citations": 0,
            "error": "NaN title"
        }

    print(f"Processing paper {index+1}/{len(df)}: {title}")
    try:
        search_query = scholarly.search_pubs(title)
        paper = next(search_query, None)  # Get the first result
        if paper:
            print(f"Found paper: {paper['bib']['title']}")
            return {
                "authors": ", ".join(paper['bib']['author']) if 'author' in paper['bib'] else "N/A",
                "abstract": paper['bib'].get('abstract', 'N/A'),
                "venue": paper['bib'].get('venue', 'N/A'),
                "publish_date": paper['bib'].get('pub_year', 'N/A'),
                "url": paper.get('pub_url', 'N/A'),
                "citations": paper.get('num_citations', 0)  # Fetch the number of citations
            }
        else:
            print(f"No match found for: {title}")
            failed_rows.append(index)
            return {
                "authors": "N/A",
                "abstract": "N/A",
                "venue": "N/A",
                "publish_date": "N/A",
                "url": "N/A",
                "citations": 0,
                "error": "No match found"
            }
    except Exception as e:
        print(f"Error processing {title}: {e}")
        failed_rows.append(index)
        return {
            "authors": "N/A",
            "abstract": "N/A",
            "venue": "N/A",
            "publish_date": "N/A",
            "url": "N/A",
            "citations": 0,
            "error": str(e)
        }

# Apply the extraction function to each paper's title
print("Starting metadata extraction process using scholarly...")
try:
    df[['authors', 'abstract', 'venue', 'publish_date', 'url', 'citations', 'error']] = df.apply(
        lambda row: pd.Series(extract_scholarly_metadata(row['title'], row.name)), axis=1
    )
except Exception as e:
    print(f"An error occurred during the processing: {e}")

# Save the updated DataFrame
df.to_csv("updated_combined_papers_scholarly.csv", index=False)
print("Extraction complete. Updated DataFrame saved to 'updated_combined_papers_scholarly.csv'")

# Report any failed rows
if failed_rows:
    print(f"Metadata extraction failed for {len(failed_rows)} papers. Indices: {failed_rows}")
    failed_df = df.iloc[failed_rows]
    failed_df.to_csv("failed_papers_scholarly.csv", index=False)
    print("Failed papers saved to 'failed_papers_scholarly.csv'")
else:
    print("All papers processed successfully.")
