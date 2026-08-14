import requests
import csv
import os

# =========================
# CONFIG
# =========================

OPENALEX_API_KEY = None

CSV_FILE = "openalex_works.csv"

# Số paper lấy trong mỗi request
PER_PAGE = 100

# Giới hạn để TEST
# Đặt None nếu muốn chạy toàn bộ dữ liệu theo filter
MAX_WORKS = 100

TOPIC_ID = "T10028"

# =========================
# OPENALEX API
# =========================

BASE_URL = "https://api.openalex.org/works"


def get_works(cursor="*"):
    """
    Lấy một batch works từ OpenAlex.
    """

    params = {
        "per-page": PER_PAGE,
        "cursor": cursor,
        "filter": f"topics.id:{TOPIC_ID}"
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

def convert_work(work):

    primary_topic = work.get("primary_topic")

    if not primary_topic:
        return None

    primary_topic_id = primary_topic.get("id", "").split("/")[-1]

    if primary_topic_id != TOPIC_ID:
        return None

    # -------------------------
    # WORK ID
    # -------------------------

    work_id = work.get("id")

    # -------------------------
    # AUTHOR IDS
    # -------------------------

    author_ids = []

    for authorship in work.get("authorships", []):

        author = authorship.get("author")

        if author and author.get("id"):
            author_ids.append(author["id"])

    # -------------------------
    # VENUE / SOURCE
    # -------------------------

    venue_id = None

    primary_location = work.get("primary_location")

    if primary_location:

        source = primary_location.get("source")

        if source:
            venue_id = source.get("id")

    # -------------------------
    # TOPIC HIERARCHY
    # -------------------------

    topic_id = None
    topic_name = None

    subfield_id = None
    subfield_name = None

    field_id = None
    field_name = None

    domain_id = None
    domain_name = None

    topic = work.get("primary_topic")

    if topic:

        topic_id = topic.get("id")
        topic_name = topic.get("display_name")

        subfield = topic.get("subfield")

        if subfield:
            subfield_id = subfield.get("id")
            subfield_name = subfield.get("display_name")

        field = topic.get("field")

        if field:
            field_id = field.get("id")
            field_name = field.get("display_name")

        domain = topic.get("domain")

        if domain:
            domain_id = domain.get("id")
            domain_name = domain.get("display_name")

    # -------------------------
    # CITATION COUNT
    # -------------------------

    cited_by_count = work.get("cited_by_count", 0)

    # -------------------------
    # CSV ROW
    # -------------------------

    document = {
        "work_id": work_id,

        # Convert list to string for CSV
        "author_ids": ";".join(author_ids),

        "venue_id": venue_id,

        "topic_id": topic_id,
        "topic": topic_name,

        "subfield_id": subfield_id,
        "subfield": subfield_name,

        "field_id": field_id,
        "field": field_name,

        "domain_id": domain_id,
        "domain": domain_name,

        "cited_by_count": cited_by_count
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
        "work_id",
        "author_ids",
        "venue_id",
        "topic_id",
        "topic",
        "subfield_id",
        "subfield",
        "field_id",
        "field",
        "domain_id",
        "domain",
        "cited_by_count"
    ]

    # -------------------------
    # CREATE / OVERWRITE CSV
    # -------------------------

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

        # -------------------------
        # SCRAPE
        # -------------------------

        while True:

            print("Đang lấy dữ liệu...")

            data = get_works(cursor)

            works = data.get("results", [])

            if not works:
                print("Không còn dữ liệu.")
                break

            for work in works:

                # Stop if MAX_WORKS reached
                if (
                    MAX_WORKS is not None
                    and total_exported >= MAX_WORKS
                ):
                    break

                document = convert_work(work)

                total_processed += 1

                if document is not None:

                    writer.writerow(document)

                    total_exported += 1

            csvfile.flush()

            print(
                f"Đã export: {total_exported} works"
            )

            # -------------------------
            # TEST LIMIT
            # -------------------------

            if (
                MAX_WORKS is not None
                and total_exported >= MAX_WORKS
            ):
                print("Đã đạt giới hạn TEST.")
                break

            # -------------------------
            # NEXT CURSOR
            # -------------------------

            cursor = data["meta"].get("next_cursor")

            if not cursor:

                print("Đã tới cuối dữ liệu.")
                break

    print(f"Hoàn thành. Tổng số works: {total_exported}")
    print(f"File CSV: {CSV_FILE}")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    scrape_openalex()