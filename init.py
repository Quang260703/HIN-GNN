from arango import ArangoClient

client = ArangoClient(hosts="http://localhost:8529")
db = client.db("sigweb_graph", username="root", password="your_pw")

# Vertex collections (node types)
db.create_collection("papers")
db.create_collection("authors")
db.create_collection("venues")
db.create_collection("pc_members")

# Edge collections (relation types) — each needs a graph definition
db.create_collection("writes", edge=True)       # author -> paper
db.create_collection("cites", edge=True)         # paper -> paper
db.create_collection("published_in", edge=True)  # paper -> venue
db.create_collection("chairs", edge=True)        # pc_member -> venue
db.create_collection("reviews", edge=True)

db.create_collection("writes", edge=True)       # author -> paper
db.create_collection("cites", edge=True)         # paper -> paper
db.create_collection("published_in", edge=True)  # paper -> venue
db.create_collection("chairs", edge=True)        # pc_member -> venue
db.create_collection("reviews", edge=True)

db.create_collection("writes", edge=True)       # author -> paper
db.create_collection("cites", edge=True)         # paper -> paper
db.create_collection("published_in", edge=True)  # paper -> venue
db.create_collection("chairs", edge=True)        # pc_member -> venue
db.create_collection("reviews", edge=True)

db.create_collection("writes", edge=True)       # author -> paper
db.create_collection("cites", edge=True)         # paper -> paper
db.create_collection("published_in", edge=True)  # paper -> venue
db.create_collection("chairs", edge=True)        # pc_member -> venue
db.create_collection("reviews", edge=True)
db.create_collection("writes", edge=True)       # author -> paper
db.create_collection("cites", edge=True)         # paper -> paper
db.create_collection("published_in", edge=True)  # paper -> venue
db.create_collection("chairs", edge=True)        # pc_member -> venue
db.create_collection("reviews", edge=True)

db.create_collection("writes", edge=True)       # author -> paper
db.create_collection("cites", edge=True)         # paper -> paper
db.create_collection("published_in", edge=True)  # paper -> venue
db.create_collection("chairs", edge=True)        # pc_member -> venue
db.create_collection("reviews", edge=True)


db.create_collection("writes", edge=True)       # author -> paper
db.create_collection("cites", edge=True)         # paper -> paper
db.create_collection("published_in", edge=True)  # paper -> venue
db.create_collection("chairs", edge=True)        # pc_member -> venue
db.create_collection("reviews", edge=True)

db.create_collection("write", edge=True)       # author -> paper
db.create_collection("cite", edge=True)         # paper -> paper
db.create_collection("published_out", edge=True)  # paper -> venue
db.create_collection("chair", edge=True)        # pc_member -> venue
db.create_collection("review", edge=True)