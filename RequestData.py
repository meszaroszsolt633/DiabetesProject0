import requests
import json
import tempfile
import ujson

import time


def DiabAPIreq(username: str = None, collection: str = None, qtype: str = None, patid: int = None,
               myquery: dict = None, limit: int = None):
    url = "https://vmi1293676.contaboserver.net/api/getData/"

    if username is None:
        return {"Error": "No username"}
    elif collection is None:
        return {"Error": "No collection"}
    elif myquery is not None:
        payload = {"collection": collection, "user": username, "myquery": json.dumps(myquery)}
    else:
        if qtype is None:
            if patid is None:
                if limit is None:
                    payload = {"collection": collection, "user": username}
                else:
                    payload = {"collection": collection, "user": username, "limit": limit}
            else:
                if limit is None:
                    payload = {"collection": collection, "user": username, "patid": patid}
                else:
                    payload = {"collection": collection, "user": username, "limit": limit, "patid": patid}
        else:
            if patid is None:
                if limit is None:
                    payload = {"collection": collection, "user": username, "qtype": qtype}
                else:
                    payload = {"collection": collection, "user": username, "limit": limit, "qtype": qtype}
            else:
                if limit is None:
                    payload = {"collection": collection, "user": username, "patid": patid, "qtype": qtype}
                else:
                    payload = {"collection": collection, "user": username, "limit": limit, "patid": patid,
                               "qtype": qtype}

    response = requests.get(url, params=payload, stream=True)

    with tempfile.TemporaryFile() as temp_file:

        for chunk in response.iter_content(chunk_size=32 * 1024):
            if chunk:
                temp_file.write(chunk)

        temp_file.seek(0)

        binary_data = temp_file.read()

    finalData = ujson.loads(binary_data)


    return finalData