import requests
import csv

# =========================
# CONFIG
# =========================

OPENALEX_API_KEY = "r8TRvSMhhnq7mjdCKWmJ6d"

PER_PAGE = 200

# Set None to scrape all works matching the filter
MAX_WORKS = None
TOPIC_ID = "T10028" 

CSV_FILE = f"openalex_works_{TOPIC_ID}.csv"

# =========================
# OPENALEX API
# =========================

BASE_URL = "https://api.openalex.org/works"


def get_works(cursor="*"):
    """
    Get one batch of works from OpenAlex.
    """

    params = {
        "per-page": PER_PAGE,
        "cursor": cursor,
        "filter": f"topics.id:{TOPIC_ID}",
    }

    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=30
    )
    response.raise_for_status()

    return response.json()


# =========================
# CONVERT WORK
# =========================

def convert_work(paper):

    # -------------------------
    # CHECK PRIMARY TOPIC
    # -------------------------

    primary_topic = paper.get("primary_topic")

    if not primary_topic:
        return None

    primary_topic_id = primary_topic.get(
        "id", ""
    ).split("/")[-1]

    if primary_topic_id != TOPIC_ID:
        return None

    # -------------------------
    # AUTHOR IDS / NAMES
    # -------------------------

    author_ids = []
    author_names = []
    institutions = []

    for authorship in paper.get(
        "authorships", []
    ):

        author = authorship.get("author")

        if author:

            if author.get("id"):
                author_ids.append(
                    author["id"]
                )

            if author.get("display_name"):
                author_names.append(
                    author["display_name"]
                )

        # Institutions
        for institution in authorship.get(
            "institutions", []
        ):

            if institution.get("display_name"):
                institutions.append(
                    institution["display_name"]
                )

    # -------------------------
    # SOURCE / VENUE
    # -------------------------

    source_id = None
    source = None

    primary_location = paper.get(
        "primary_location"
    )

    if primary_location:

        source_data = primary_location.get(
            "source"
        )

        if source_data:

            source_id = source_data.get("id")

            source = source_data.get(
                "display_name"
            )

    # -------------------------
    # CSV ROW
    # -------------------------

    document = {

        "paper_id": paper.get("id"),

        "title": paper.get(
            "display_name"
        ),

        "publication_year": paper.get(
            "publication_year"
        ),

        "citation_count": paper.get(
            "cited_by_count"
        ),

        "referenced_works_count": paper.get(
            "referenced_works_count"
        ),

        "referenced_works": ";".join(
            paper.get(
                "referenced_works",
                []
            )
        ),

        "related_works": ";".join(
            paper.get(
                "related_works",
                []
            )
        ),

        "author_ids": ";".join(
            author_ids
        ),

        "authors": ";".join(
            author_names
        ),

        "institutions": ";".join(
            sorted(set(institutions))
        ),

        "source_id": source_id,

        "source": source,

        "fwci": paper.get(
            "fwci"
        )
    }

    return document


# =========================
# CSV SCRAPER
# =========================

def scrape_openalex():

    cursor = "*"

    total_processed = 0
    total_exported = 0

    fieldnames = [
        "paper_id",
        "title",
        "publication_year",
        "citation_count",
        "referenced_works_count",
        "referenced_works",
        "related_works",
        "author_ids",
        "authors",
        "institutions",
        "source_id",
        "source",
        "fwci"
    ]

    # =========================
    # CREATE / OVERWRITE CSV
    # =========================

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()

        # =========================
        # SCRAPE
        # =========================

        while True:

            print("Đang lấy dữ liệu...")

            data = get_works(cursor)

            works = data.get(
                "results",
                []
            )

            if not works:

                print(
                    "Không còn dữ liệu."
                )

                break

            for paper in works:

                # Stop at MAX_WORKS
                if (
                    MAX_WORKS is not None
                    and total_exported >= MAX_WORKS
                ):
                    break

                document = convert_work(
                    paper
                )

                total_processed += 1

                if document is not None:

                    writer.writerow(
                        document
                    )

                    total_exported += 1

            csvfile.flush()

            print(
                f"Đã export: "
                f"{total_exported} works"
            )

            # =========================
            # TEST LIMIT
            # =========================

            if (
                MAX_WORKS is not None
                and total_exported >= MAX_WORKS
            ):

                print(
                    "Đã đạt giới hạn TEST."
                )

                break

            # =========================
            # NEXT CURSOR
            # =========================

            cursor = data["meta"].get(
                "next_cursor"
            )

            if not cursor:

                print(
                    "Đã tới cuối dữ liệu."
                )

                break

    print(
        f"Hoàn thành. "
        f"Tổng số works: {total_exported}"
    )

    print(
        f"File CSV: {CSV_FILE}"
    )


# =========================
# RUN
# =========================

if __name__ == "__main__":
    scrape_openalex()