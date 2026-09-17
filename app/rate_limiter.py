import time
from collections import defaultdict

REQUEST_LIMIT = 10
WINDOW_SECONDS = 60

request_history = defaultdict(list)


def check_rate_limit(client_id):

    current_time = time.time()

    request_history[client_id] = [
        timestamp
        for timestamp in request_history[client_id]
        if current_time - timestamp < WINDOW_SECONDS
    ]

    if len(request_history[client_id]) >= REQUEST_LIMIT:
        return False

    request_history[client_id].append(current_time)

    return True
