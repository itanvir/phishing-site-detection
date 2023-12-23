import pandas as pd
import re
import os


def data_ingestion():

    # Read the index SQL file
    with open("data/index.sql", "r") as f:
        sql_text = f.read()

    # Extract the data using regular expressions
    data_pattern = re.compile(r"\((\d+), \'(.*?)\', \'(.*?)\', (\d+), \'(.*?)\'\)")
    data_matches = data_pattern.findall(sql_text)

    # Create a list of data records
    data_records = []

    for match in data_matches:
        rec_id, url, website, result, created_date = match
        data_records.append((int(rec_id), url, website, int(result), created_date))

    # Convert the data records to a DataFrame
    columns = ["rec_id", "url", "website", "result", "created_date"]
    df = pd.DataFrame(data_records, columns=columns)

    return df


def data_preparation(df):

    # Define the path to the parent directory containing the partitions
    parent_directory = "data/dataset"

    # Define the list of partition directories
    partition_directories = [
        "dataset-part-1",
        "dataset-part-2",
        "dataset-part-3",
        "dataset-part-4",
        "dataset-part-5",
        "dataset-part-6",
        "dataset-part-7",
        "dataset-part-8",
    ]

    contents = []
    for i in range(len(df)):
        target_file = df.iloc[i]["website"]

        for partition_dir in partition_directories:
            directory_path = os.path.join(parent_directory, partition_dir)

            # Check if the partition directory exists
            if not os.path.exists(directory_path):
                continue

            # Search for the file within the partition directory
            target_file_path = os.path.join(directory_path, target_file)

            # Check if the target file exists
            if os.path.exists(target_file_path):
                # print (target_file_path)
                # Read the content of the file
                with open(target_file_path, "r") as file:
                    content = file.read()

                contents.append(content)

    df["contents"] = contents
    # df["url_contents"] = df["url"] + "||" + df["contents"]

    return df


if __name__ == "__main__":
    # Data Ingestion
    df = data_ingestion()

    # Data preparation
    df = data_preparation(df)

    # Save to a disk
    df.to_parquet("data/phishing_site_data.parquet.gzip", compression="gzip")

    print("Preprocessing Step Succeeded")
