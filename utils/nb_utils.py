from typing import Callable, Any
import pandas as pd
import time


def do_retry(
    func: Callable,
    *args,
    return_on: Callable[Any, bool],
    retry_count=3,
    sleep_seconds=3,
) -> Any:
    attempt = 1
    while attempt <= retry_count:
        print(f"## Attempt {attempt} of {retry_count}.")
        result = func(*args)
        if return_on(result):
            print(f"## Finished in {attempt} attempt.")
            return result
        else:
            time.sleep(sleep_seconds)
            attempt += 1
            continue
    print(f"## Exiting after {attempt} attempts.")


def simulate_realtime_stream(df: pd.DataFrame, chunk_size: int, delay: float):
    # Number of chunks to yield
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)

    for i in range(num_chunks):
        # Yield the chunk of DataFrame
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        yield chunk

        # Simulate real-time delay
        time.sleep(delay)
