import RequestData
import time
import pandas as pd
import json

start = time.perf_counter()

data = RequestData.DiabAPIreq(username="sotetestkey", collection="buckinghamdataset")

endreq = time.perf_counter()
ms1 = (endreq - start)

print("done " + str(ms1))
with open('data.json','w') as f:
    json.dump(data,f)

