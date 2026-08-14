import requests
from pymongo import MongoClient

# =========================
# CONFIG
# =========================

OPENALEX_API_KEY = None

MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "openalex_database"
COLLECTION_NAME = "works"

# Số paper lấy trong mỗi request
PER_PAGE = 100

# Giới hạn để TEST
# Đặt None nếu muốn chạy toàn bộ dữ liệu theo filter
MAX_WORKS = 1000

DOMAIN_ID = "3"
FIELD_ID = "17"
SUBFIELD_ID = "1702"
TOPIC_ID = "T10028"

# =========================
# MONGODB
# =========================

client = MongoClient(MONGO_URI)

db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Tránh insert trùng Work
collection.create_index("work_id", unique=True)


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

    print("PRIMARY TOPIC:", primary_topic)

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

    primary_topic = work.get("primary_topic")

    if primary_topic:

        topic_id = primary_topic.get("id")
        topic_name = primary_topic.get("display_name")

        subfield = primary_topic.get("subfield")

        if subfield:
            subfield_id = subfield.get("id")
            subfield_name = subfield.get("display_name")

        field = primary_topic.get("field")

        if field:
            field_id = field.get("id")
            field_name = field.get("display_name")

        domain = primary_topic.get("domain")

        if domain:
            domain_id = domain.get("id")
            domain_name = domain.get("display_name")


    # -------------------------
    # CITATION COUNT
    # -------------------------

    cited_by_count = work.get("cited_by_count", 0)


    # -------------------------
    # MONGODB DOCUMENT
    # -------------------------

    document = {

        "work_id": work_id,

        "author_ids": author_ids,

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
# SCRAPER
# =========================

def scrape_openalex():

    cursor = "*"

    total_inserted = 0

    while True:

        print("Đang lấy dữ liệu...")

        data = get_works(cursor)

        works = data.get("results", [])

        if not works:
            print("Không còn dữ liệu.")
            break


        documents = []

        for work in works:

            document = convert_work(work)

            if document is not None:
                documents.append(document)


        # -------------------------
        # INSERT / UPDATE
        # -------------------------

        for document in documents:

            try:

                collection.update_one(
                    {
                        "work_id": document["work_id"]
                    },
                    {
                        "$set": document
                    },
                    upsert=True
                )

                total_inserted += 1

            except Exception as e:

                print(
                    "Lỗi khi lưu:",
                    document["work_id"],
                    e
                )


        print(
            f"Đã xử lý: {total_inserted} works"
        )


        # -------------------------
        # TEST LIMIT
        # -------------------------

        if MAX_WORKS is not None:

            if total_inserted >= MAX_WORKS:

                print("Đã đạt giới hạn TEST.")
                break


        # -------------------------
        # NEXT CURSOR
        # -------------------------

        cursor = data["meta"].get("next_cursor")

        if not cursor:

            print("Đã tới cuối dữ liệu.")
            break


# =========================
# RUN
# =========================

if __name__ == "__main__":

    scrape_openalex()

    print("Hoàn thành.")