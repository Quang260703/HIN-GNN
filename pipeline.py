import requests
import pandas as pd
import time

TOPIC_ID = "T10181"
BASE_URL = "https://api.openalex.org/works"

params = {
    "filter": f"topics.id:{TOPIC_ID}",
    "cursor": "*",
    "per_page": 200,   # maximum allowed
}

rows = []
page = 1

while True:
    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        print(f"Error {response.status_code}")
        print(response.text)
        break

    data = response.json()
    papers = data["results"]

    if not papers:
        break

    print(f"Page {page}: {len(papers)} papers")

    for paper in papers:

        # Authors
        author_names = []
        author_ids = []

        for authorship in paper.get("authorships", []):
            author = authorship.get("author", {})
            if author.get("display_name"):
                author_names.append(author["display_name"])
            if author.get("id"):
                author_ids.append(author["id"])

        # Institutions
        institutions = []
        for authorship in paper.get("authorships", []):
            for inst in authorship.get("institutions", []):
                institutions.append(inst.get("display_name"))

        # Source
        source = None
        source_id = None
        if paper.get("primary_location") and paper["primary_location"].get("source"):
            source = paper["primary_location"]["source"].get("display_name")
            source_id = paper["primary_location"]["source"].get("id")

        rows.append({
            "paper_id": paper.get("id"),
            "doi": paper.get("doi"),
            "title": paper.get("display_name"),
            "publication_year": paper.get("publication_year"),
            "publication_date": paper.get("publication_date"),

            "citation_count": paper.get("cited_by_count"),
            "referenced_works_count": paper.get("referenced_works_count"),

            "referenced_works": ";".join(paper.get("referenced_works", [])),
            "related_works": ";".join(paper.get("related_works", [])),

            "author_ids": ";".join(author_ids),
            "authors": ";".join(author_names),
            "institutions": ";".join(sorted(set(institutions))),

            "source_id": source_id,
            "source": source,

            "language": paper.get("language"),
            "type": paper.get("type"),
            "is_open_access": paper.get("open_access", {}).get("is_oa"),
            "fwci": paper.get("fwci"),
            "updated_date": paper.get("updated_date"),
            "created_date": paper.get("created_date")
        })

    next_cursor = data["meta"]["next_cursor"]

    if next_cursor is None:
        break

    params["cursor"] = next_cursor
    page += 1

    # Optional: avoid hitting rate limits
    time.sleep(0.1)

df = pd.DataFrame(rows)

print(f"\nDownloaded {len(df):,} papers.")

df.to_csv("nlp_papers.csv", index=False)

print("Saved to nlp_papers.csv")