"""
Pull UN Comtrade data for a single commodity.

Note: The UN Comtrade basic individual subscription allows for 100k records to be pulled per call with up to
500 calls per day at 1 call per second. This code loops over the reporter countries one at a time with a pause 
between calls to stay within rate limits.

Resources:
- https://uncomtrade.org/docs/
- https://comtradedeveloper.un.org/api-details#api=comtrade-v1&operation=get-get 
"""

import os
import time
from dotenv import load_dotenv
import pandas as pd
import requests
from tqdm import tqdm

# --- Load API key from .env file ---
load_dotenv()
api_key = os.getenv("UN_COMTRADE_API_KEY")
assert api_key, "UN_COMTRADE_API_KEY not found in .env file."


# --- Get full list of reporting countries ---

REPORTERS_URL = "https://comtradeapi.un.org/files/v1/app/reference/Reporters.json"
reporters_response = requests.get(REPORTERS_URL)
reporters_response.raise_for_status()
reporters_json = reporters_response.json()

reporters_list = reporters_json["results"]
print(f"Total reporters found: {len(reporters_list)}")
print(f"Sample entry: {reporters_list[0]}")

reporter_codes = [r["id"] for r in reporters_list]

# --- Output path ---
OUTPUT_PATH = "data/raw/comtrade_oil_imports_2024_raw.csv"

# if previous run already wrote data, we can skip those reporters to avoid duplicates
if os.path.exists(OUTPUT_PATH):
    df_existing = pd.read_csv(OUTPUT_PATH)
    already_pulled = set(df_existing["reporterCode"].unique())
    print("Some reporters already pulled in previous run. Skipping those reporters to avoid duplicates.")

    reporter_codes = [c for c in reporter_codes if c not in already_pulled]

# --- Build params ---
# HTML pattern: https://comtradeapi.un.org/data/v1/get/{typeCode}/{freqCode}/{clCode}
# Additional filters will be passed as query parameters.

BASE_URL = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
CMD_CODE = "2709" # Commodity code for crude oil
PERIOD = "2024"
FLOW_CODE = "M" # M for imports

BASE_SLEEP = 2
MAX_RETRIES = 3
RATE_LIMIT_BACKOFF = 30

# The API caps a single call at 100k records; flag any reporter that comes
# nears that threshold and check to make sure they didn't get truncated
TRUNCATION_WARNING_THRESHOLD = 90_000


# --- Loop over reporters and pull data ---
all_data = []
failed_reporters = []
empty_reporters = []
possibly_truncated_reporters = []

for code in tqdm(reporter_codes, desc="Pulling Comtrade data by reporter"):


    params = {
        "subscription-key": api_key,
        "period": PERIOD,
        "reporterCode": code,
        "cmdCode": CMD_CODE,
        "flowCode": FLOW_CODE,                 # M for imports
        "partnerCode": None,                   # None means all partners
        "breakdownMode": "plus",
        "includeDesc": "true"
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            rows = data.get("data", [])
            if rows:
                df_country = pd.DataFrame(rows)
                all_data.append(df_country)

                # Incremental save
                file_exists = os.path.exists(OUTPUT_PATH)
                df_country.to_csv(OUTPUT_PATH, mode='a', header=not file_exists, index=False)

                n_rows = len(rows)
                if n_rows >= TRUNCATION_WARNING_THRESHOLD:
                    possibly_truncated_reporters.append((code, n_rows))
                    tqdm.write(
                        f"WARNING: reporter {code} returned {n_rows:,} rows, near the "
                        "100k/call cap - this reporter's data may be truncated."
                    )
                print(f"Successfully pulled data for reporter {code}. Rows: {n_rows}")
            else:
                empty_reporters.append(code)
            
            break

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None

            if status == 429:
                tqdm.write(f"Rate limited on reporter {code}. Backing off for {RATE_LIMIT_BACKOFF} seconds.")
                time.sleep(RATE_LIMIT_BACKOFF)
                continue  # Retry the request after backoff
            else:
                error_detail = str(e)
                failed_reporters.append((code, error_detail))
                print(f"Failed to pull data for reporter {code}: {error_detail}")
                break

        except requests.exceptions.RequestException as e:
            failed_reporters.append((code, str(e)))
            print(f"Failed to pull data for reporter {code}: {str(e)}")
            break
    
    else:
        # If retries exhausted without success, log the failure
        failed_reporters.append((code, "Max retries exceeded"))
        tqdm.write(f"Max retries exceeded for reporter {code}. Moving to next reporter.")

    time.sleep(BASE_SLEEP)



# --- Summary ---
print(f"\nTotal reporters processed: {len(reporter_codes)}")
print(f"Total reporters failed: {len(failed_reporters)}")
print(f"Total reporters with no data: {len(empty_reporters)}")
print(f"Total reporters near the 100k/call cap (possibly truncated): {len(possibly_truncated_reporters)}")
if failed_reporters:
    print("Failed reporters:")
    for code, error in failed_reporters:
        print(f"Reporter {code}: {error}")

if empty_reporters:
    print("Empty reporters:")
    for code in empty_reporters:
        print(f"Reporter {code}: No data available")

if possibly_truncated_reporters:
    print("Reporters near the 100k/call cap - verify these aren't missing rows:")
    for code, n_rows in possibly_truncated_reporters:
        print(f"Reporter {code}: {n_rows:,} rows")