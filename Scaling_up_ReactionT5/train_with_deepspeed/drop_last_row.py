import glob
import pandas as pd
import gc
# for path in set(glob.glob("/data/sagawa/preprocessed_ZINC22/*.parquet")) - set(['/data/sagawa/preprocessed_ZINC22/H17.parquet', '/data/sagawa/preprocessed_ZINC22/H04.parquet', '/data/sagawa/preprocessed_ZINC22/H05.parquet', '/data/sagawa/preprocessed_ZINC22/H06.parquet']):
#     print(path)
    # file = pd.read_parquet(path)
    # file.drop(file.tail(1).index, inplace=True)
#     file.to_parquet(path)
#     del file
#     gc.collect()



import pyarrow.parquet as pq
import pyarrow as pa

def remove_last_row_large_parquet(input_path, output_path):
    # Parquetファイルを開く
    parquet_file = pq.ParquetFile(input_path)
    
    # 書き込むためのテーブルを初期化
    writer = None
    
    # バッチでファイルを読み込む
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=1000000)):
        # データをテーブルに変換
        table = pa.Table.from_batches([batch])
        
        # 最後のバッチの場合、最終行を削除
        if i == parquet_file.num_row_groups - 1:
            table = table.slice(0, table.num_rows - 1)
        
        # データを新しいParquetファイルに書き込む
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
    
    # ファイルを閉じる
    if writer:
        writer.close()
paths = glob.glob("/media/sagawa/7182ee6c-8215-4bea-a609-999c7c2c02cf/lm/preprocessed_ZINC22/*.parquet") + glob.glob("/media/sagawa/7182ee6c-8215-4bea-a609-999c7c2c02cf/arare/preprocessed_ZINC22/*.parquet") 
paths = set(paths) - set([
'/media/sagawa/7182ee6c-8215-4bea-a609-999c7c2c02cf/lm/preprocessed_ZINC22/H28_2.parquet','/media/sagawa/7182ee6c-8215-4bea-a609-999c7c2c02cf/lm/preprocessed_ZINC22/H25.parquet','/media/sagawa/7182ee6c-8215-4bea-a609-999c7c2c02cf/lm/preprocessed_ZINC22/H26.parquet','/media/sagawa/7182ee6c-8215-4bea-a609-999c7c2c02cf/lm/preprocessed_ZINC22/H29_2.parquet','/media/sagawa/7182ee6c-8215-4bea-a609-999c7c2c02cf/lm/preprocessed_ZINC22/H28.parquet','/media/sagawa/7182ee6c-8215-4bea-a609-999c7c2c02cf/lm/preprocessed_ZINC22/H27.parquet','/media/sagawa/7182ee6c-8215-4bea-a609-999c7c2c02cf/lm/preprocessed_ZINC22/H29.parquet'])
print(paths)
for path in paths:
    print(path)
    remove_last_row_large_parquet(path, path.replace(".parquet", "_drop_last_row.parquet").replace("/lm", "").replace("/arare", ""))