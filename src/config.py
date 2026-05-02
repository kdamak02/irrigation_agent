SEQ_LENGTH = 24

TARGET_COL = "soil_moisture"
TIME_COL = "datetime"

SELECTED_FEATURES = [
    "soil_temperature",
    "band_950",
    "band_946",
    "band_942",
    "band_938",
    "band_934",
    "band_930",
    "band_926",
    "band_922",
    "band_918",
    "band_914",
    "band_898",
]

MIN_THRESHOLD = 25.0
SAFETY_THRESHOLD = 40.0

RAIN_LIMIT = 2.0
USE_RAIN = True