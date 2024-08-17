import glob
import pandas as pd
import pyarrow.parquet as pq
sum_ = 0
for path in glob.glob("/data/sagawa/preprocessed_ZINC22/*.parquet"):
    file = pq.ParquetFile(path)
    num_rows = file.metadata.num_rows
    print(f"{path} has {num_rows} rows")
    sum_ += num_rows
print(f"Total number of rows: {sum_}")