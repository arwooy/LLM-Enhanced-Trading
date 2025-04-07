#!/usr/bin/env python
# coding: utf-8

# This notebook processes and organizes Reddit submissions and comments from the WallStreetBets subreddit into structured datasets for analysis. It includes the following steps:
# 
# 1. Filter and Clean Raw Data:
# 
#     Filters submissions and comments by date range (2021–2023).
#     Extracts relevant fields and saves them into separate CSV files.
# 
# 2. Split Data by Year:
# 
#     Divides filtered submissions and comments into separate yearly datasets for easier processing.
# 
# 3. Merge Submissions and Comments:
# 
#     Associates comments with their corresponding parent submissions.
#     Separates unmatched comments for review and logs errors and dropped rows.
# 
# 4. Output Flattened Data:
# 
#     Generates flat, year-specific datasets that combine submissions and comments for analysis.
#     Logs errors and dropped rows to separate CSV files for transparency.

# ### Reddit Comment Filter and Export
# This script filters Reddit comments from a JSON file based on a date range and exports the results to a structured CSV file.

# In[30]:


import json
import os
import csv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # Use pytz if on Python <3.9

# Set your parameters
input_file = r"D:\reddit\subreddits23\wallstreetbets_comments\wallstreetbets_comments"
output_file = r"D:\reddit\output\filtered_comments.csv"
from_date = datetime(2021, 1, 1, tzinfo=ZoneInfo("America/New_York"))
to_date = datetime(2023, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("America/New_York"))

# Ensure the output directory exists
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_comments(input_file, output_file, from_date, to_date):
    with open(input_file, 'r', encoding='utf-8') as file_in, open(output_file, 'w', encoding='utf-8', newline='') as file_out:
        writer = csv.writer(file_out, quoting=csv.QUOTE_MINIMAL)
        
        # Writing CSV header for the comments dataset
        writer.writerow(["id", "score", "date", "time", "author", "parent_id", "link_id", "body"])

        for line in file_in:
            try:
                obj = json.loads(line.strip())  # Load each line as a JSON object
                
                # Convert created_utc to a timezone-aware datetime in UTC
                created_utc = datetime.fromtimestamp(int(obj['created_utc']), tz=timezone.utc)
                
                # Convert UTC datetime to Eastern Time (ET)
                created_et = created_utc.astimezone(ZoneInfo("America/New_York"))

                # Filter by date range in ET
                if from_date <= created_et <= to_date:
                    # Extract fields for the comments dataset
                    comment_id = obj.get('id', '')
                    score = obj.get('score', '')
                    date = created_et.strftime("%Y-%m-%d")
                    time = created_et.strftime("%H:%M:%S")  # Exact time in seconds
                    author = obj.get('author', '')
                    parent_id = obj.get('parent_id', '')
                    # Remove the 't3_' prefix from link_id for later matching
                    link_id = obj.get('link_id', '').replace('t3_', '')
                    body = obj.get('body', '')

                    # Write to CSV
                    writer.writerow([comment_id, score, date, time, author, parent_id, link_id, body])

            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON decode error: {e}")

# Run the processing function
process_comments(input_file, output_file, from_date, to_date)


# This script reads a large CSV file of filtered Reddit comments in chunks, splits the data by year (2021, 2022, 2023), and writes each year's data to a separate CSV file.

# In[31]:


import pandas as pd

# Define the file paths and create separate CSV writers for each year
input_file = 'D:/reddit/output/filtered_comments.csv'
output_files = {
    2021: 'D:/reddit/output/filtered_comments_2021.csv',
    2022: 'D:/reddit/output/filtered_comments_2022.csv',
    2023: 'D:/reddit/output/filtered_comments_2023.csv'
}

# Initialize files for each year and write headers with full columns
for year, file_path in output_files.items():
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        file.write("id,score,date,time,author,parent_id,link_id,body\n")  # Writing header

# Process the input file in chunks
chunk_size = 100000  # Adjust the chunk size based on available memory

for chunk in pd.read_csv(input_file, chunksize=chunk_size, parse_dates=["date"]):
    # Drop rows with invalid dates (if any)
    chunk = chunk.dropna(subset=['date'])

    # Ensure columns are in the correct order to avoid misalignment
    chunk = chunk[['id', 'score', 'date', 'time', 'author', 'parent_id', 'link_id', 'body']]

    # Extract the year from the 'date' column
    chunk['year'] = chunk['date'].dt.year

    # Filter each chunk by year and append to the respective output file
    for year in [2021, 2022, 2023]:
        df_year = chunk[chunk['year'] == year]
        if not df_year.empty:
            with open(output_files[year], 'a', encoding='utf-8', newline='') as file:
                df_year.to_csv(file, columns=['id', 'score', 'date', 'time', 'author', 'parent_id', 'link_id', 'body'], header=False, index=False)

print("Data successfully split by year into separate files.")


# ### Filter Reddit Submissions by Date
# This script processes a JSON file of Reddit submissions, filters entries based on a specified date range, and saves the filtered submissions to a CSV file.

# In[36]:


import json
import os
import csv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # Use pytz if on Python <3.9

# Set your parameters
input_file = r"D:\reddit\subreddits23\wallstreetbets_submissions\wallstreetbets_submissions"
output_file = r"D:\reddit\output\filtered_submissions.csv"
from_date = datetime(2021, 1, 1, tzinfo=ZoneInfo("America/New_York"))
to_date = datetime(2023, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("America/New_York"))

# Ensure the output directory exists
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_file(input_file, output_file, from_date, to_date):
    with open(input_file, 'r', encoding='utf-8') as file_in, open(output_file, 'w', encoding='utf-8', newline='') as file_out:
        writer = csv.writer(file_out, quoting=csv.QUOTE_MINIMAL)
        
        # Writing CSV header based on required fields
        writer.writerow(["id", "score", "date", "time", "title", "author", "permalink", "selftext", "url"])

        for line in file_in:
            try:
                obj = json.loads(line.strip())  # Load each line as a JSON object
                
                # Convert created_utc to a timezone-aware datetime in UTC
                created_utc = datetime.fromtimestamp(int(obj['created_utc']), tz=timezone.utc)
                
                # Convert UTC datetime to Eastern Time (ET)
                created_et = created_utc.astimezone(ZoneInfo("America/New_York"))

                # Filter by date range in ET
                if from_date <= created_et <= to_date:
                    # Extract fields based on requirements
                    submission_id = obj.get('id', '')
                    score = obj.get('score', '')  # Assuming 'score' is available; adjust if needed
                    date = created_et.strftime("%Y-%m-%d")
                    time = created_et.strftime("%H:%M:%S")  # Exact time in seconds
                    title = obj.get('title', '')
                    author = obj.get('author', '')
                    permalink = obj.get('permalink', '')
                    selftext = obj.get('selftext', '')
                    url = obj.get('url', '')

                    # Write to CSV
                    writer.writerow([submission_id, score, date, time, title, author, permalink, selftext, url])

            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON decode error: {e}")

# Run the processing function
process_file(input_file, output_file, from_date, to_date)


# This script splits a large CSV file of filtered Reddit submissions into separate files by year (2021, 2022, 2023).

# In[37]:


import pandas as pd

# Define the file paths and create separate CSV writers for each year
input_file = 'D:/reddit/output/filtered_submissions.csv'
output_files = {
    2021: 'D:/reddit/output/filtered_submissions_2021.csv',
    2022: 'D:/reddit/output/filtered_submissions_2022.csv',
    2023: 'D:/reddit/output/filtered_submissions_2023.csv'
}

# Initialize files for each year and write headers with full columns
for year, file_path in output_files.items():
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        file.write("id,score,date,time,title,author,permalink,selftext,url\n")  # Writing header

# Process the input file in chunks
chunk_size = 100000  # Adjust the chunk size based on available memory

for chunk in pd.read_csv(input_file, chunksize=chunk_size, parse_dates=["date"]):
    # Drop rows with invalid dates (if any)
    chunk = chunk.dropna(subset=['date'])

    # Ensure columns are in the correct order to avoid misalignment
    chunk = chunk[['id', 'score', 'date', 'time', 'title', 'author', 'permalink', 'selftext', 'url']]

    # Extract the year from the 'date' column
    chunk['year'] = chunk['date'].dt.year

    # Filter each chunk by year and append to the respective output file
    for year in [2021, 2022, 2023]:
        df_year = chunk[chunk['year'] == year]
        if not df_year.empty:
            with open(output_files[year], 'a', encoding='utf-8', newline='') as file:
                df_year.to_csv(file, columns=['id', 'score', 'date', 'time', 'title', 'author', 'permalink', 'selftext', 'url'], header=False, index=False)

print("Submissions data successfully split by year into separate files.")


# ### Merge Reddit Submissions and Comments by Year
# This script processes and merges Reddit submissions and comments for each year (2021, 2022, 2023). It associates comments with their parent submissions, logs unmatched comments, and outputs a flattened dataset.

# In[49]:


import pandas as pd
import csv

# Define file paths for submissions and comments by year
submissions_files = {
    2021: 'D:/reddit/output/filtered_submissions_2021.csv',
    2022: 'D:/reddit/output/filtered_submissions_2022.csv',
    2023: 'D:/reddit/output/filtered_submissions_2023.csv'
}
comments_files = {
    2021: 'D:/reddit/output/filtered_comments_2021.csv',
    2022: 'D:/reddit/output/filtered_comments_2022.csv',
    2023: 'D:/reddit/output/filtered_comments_2023.csv'
}

# Output files for merged data and unmatched comments
output_files = {
    2021: 'D:/reddit/output/flattened_submissions_with_comments_2021.csv',
    2022: 'D:/reddit/output/flattened_submissions_with_comments_2022.csv',
    2023: 'D:/reddit/output/flattened_submissions_with_comments_2023.csv'
}
unmatched_files = {
    2021: 'D:/reddit/output/unmatched_comments_2021.csv',
    2022: 'D:/reddit/output/unmatched_comments_2022.csv',
    2023: 'D:/reddit/output/unmatched_comments_2023.csv'
}
error_log_file = 'D:/reddit/output/error_log.csv'  # Error log file to keep track of problematic rows

# Initialize a list to keep track of errors and dropped rows
error_log = []
drop_log = []

# Process each year separately
for year in [2021, 2022, 2023]:
    # Load submissions with error handling
    try:
        submissions_df = pd.read_csv(submissions_files[year], on_bad_lines='skip')
    except pd.errors.ParserError as e:
        error_log.append({'file': submissions_files[year], 'year': year, 'error': str(e)})
        print(f"Error reading submissions file for {year}: {e}")
        continue

    # Rename columns in submissions to differentiate them when merged with comments
    submissions_df = submissions_df.rename(columns={
        "id": "submission_id",
        "score": "submission_score",
        "date": "submission_date",
        "time": "submission_time",
        "title": "submission_title",
        "author": "submission_author",
        "permalink": "submission_permalink",
        "selftext": "submission_selftext",
        "url": "submission_url"
    })

    # Load comments with error handling
    try:
        comments_df = pd.read_csv(comments_files[year], on_bad_lines='skip')
    except pd.errors.ParserError as e:
        error_log.append({'file': comments_files[year], 'year': year, 'error': str(e)})
        print(f"Error reading comments file for {year}: {e}")
        continue

    # Count the original number of rows in comments
    original_row_count = len(comments_df)

    # Remove rows with NaN values in `parent_id`
    comments_df = comments_df.dropna(subset=['parent_id'])

    # Count and log the number of dropped rows
    dropped_row_count = original_row_count - len(comments_df)
    drop_log.append({'year': year, 'dropped_rows': dropped_row_count})
    print(f"Dropped {dropped_row_count} rows with NaN parent_id for year {year}")

    # Filter comments to include only those with `parent_id` pointing to a submission (i.e., starts with "t3_")
    comments_df = comments_df[comments_df['parent_id'].str.startswith('t3_')].copy()

    # Clean up `parent_id` by removing the "t3_" prefix
    comments_df['parent_id'] = comments_df['parent_id'].str.replace('t3_', '')

    # Rename columns in comments for clarity
    comments_df = comments_df.rename(columns={
        "id": "comment_id",
        "score": "comment_score",
        "date": "comment_date",
        "time": "comment_time",
        "author": "comment_author",
        "body": "comment_body"
    })

    # Merge comments with their associated submission data (left join to keep all comments)
    merged_df = comments_df.merge(submissions_df, left_on='parent_id', right_on='submission_id', how='left')

    # Separate unmatched comments (those with NaN in `submission_id` column)
    unmatched_comments_df = merged_df[merged_df['submission_id'].isna()]

    # Log unmatched comments to a separate CSV file
    if not unmatched_comments_df.empty:
        unmatched_comments_df.to_csv(unmatched_files[year], index=False)
        print(f"Unmatched comments for {year} saved to {unmatched_files[year]}")
    else:
        print(f"No unmatched comments for {year}.")

    # Filter out unmatched comments from the merged data (optional)
    matched_comments_df = merged_df.dropna(subset=['submission_id'])

    # Save the matched data to a flat CSV file
    matched_comments_df.to_csv(output_files[year], index=False)
    print(f"Flattened data for {year} saved to {output_files[year]}")

# Save the error log to a CSV file if there were any errors
if error_log:
    pd.DataFrame(error_log).to_csv(error_log_file, index=False)
    print(f"Error log saved to {error_log_file}")
else:
    print("No errors encountered during processing.")

# Save the drop log to a CSV file to record the dropped rows
drop_log_file = 'D:/reddit/output/drop_log.csv'
pd.DataFrame(drop_log).to_csv(drop_log_file, index=False)
print(f"Drop log saved to {drop_log_file}")

