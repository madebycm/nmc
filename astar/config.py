"""Configuration for Astar Island solver."""

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwNjVlNzg1Zi0zNmMxLTRkZWEtOGU5YS0wZTVjMDY1OTk1MmQiLCJlbWFpbCI6ImNocmlzdGlhbi5tZWluaG9sZEBnbWFpbC5jb20iLCJpc19hZG1pbiI6ZmFsc2UsImV4cCI6MTc3NDU1ODg2MX0.GbIYHMGkrUMx0J6n3wUTjBuQrsaFmSOqs7ivVqH6H5Q"

# Polling interval in seconds
POLL_INTERVAL = 120  # 2 minutes

# Query allocation: 9 per seed base (45 total) = full 3x3 tiling, 5 for precision
QUERIES_PER_SEED = 9
