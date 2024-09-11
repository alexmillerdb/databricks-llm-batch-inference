import httpx

def is_backpressure(error: httpx.HTTPStatusError):
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        return error.response.status_code in (429, 503)

def is_other_error(error: httpx.HTTPStatusError):
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        return error.response.status_code != 503 and (
            error.response.status_code >= 500 or error.response.status_code == 408)