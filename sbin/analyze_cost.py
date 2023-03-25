import pandas as pd
import sys
df = pd.read_csv(sys.argv[1])

print("#" * 90)
print(
    df.query('qty > 0').groupby("start_id").qty.sum()
)

print("#"*90)
print(
    df.query('qty > 0').groupby("end_id").qty.sum()
)

print("#"*90)

print(df.query("bool_is_weird == True and qty > 0").sort_values('qty'))


print("#"*90)

print(df.query("bool_is_weird == True and qty > 0").qty.sum())