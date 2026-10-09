# web_agent.py

import requests

def web_search(query):

    url = (
        "https://duckduckgo.com/html/"
    )

    try:

        r = requests.get(
            url,
            params={
                "q": query
            },
            timeout=10
        )

        return r.text[:5000]

    except Exception as e:

        return str(e)