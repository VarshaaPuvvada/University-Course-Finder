import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


QUERIES = [
    "I want to learn machine learning for finance",
    "beginner data analytics with SQL and dashboards",
    "cybersecurity fundamentals for a software engineer",
    "cloud computing and devops career path",
    "product management and business analytics",
    "deep learning with python and neural networks",
]


def post_recommend(base_url: str, query: str, timeout: int) -> tuple[float, int, str]:
    payload = {
        "query": query,
        "current_skills": ["Python Programming"],
        "student_level": "beginner",
        "career_goal": "machine learning engineer",
        "top_k": 5,
        "preferred_skills": ["Machine Learning", "Data Analysis"],
        "learner_progress": 0.35,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/recommend",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            return (time.perf_counter() - start) * 1000, response.status, str(len(body.get("recommendations", [])))
    except urllib.error.HTTPError as exc:
        return (time.perf_counter() - start) * 1000, exc.code, exc.read().decode("utf-8")[:120]
    except Exception as exc:
        return (time.perf_counter() - start) * 1000, 0, str(exc)[:120]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((pct / 100) * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test the course recommendation API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(post_recommend, args.base_url, QUERIES[index % len(QUERIES)], args.timeout)
            for index in range(args.requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.perf_counter() - started
    latencies = [latency for latency, status, _ in results if status == 200]
    failures = [result for result in results if result[1] != 200]

    print(json.dumps(
        {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "successful": len(latencies),
            "failed": len(failures),
            "throughput_rps": round(args.requests / elapsed, 3) if elapsed else 0,
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 1) if latencies else 0,
                "median": round(statistics.median(latencies), 1) if latencies else 0,
                "p95": round(percentile(latencies, 95), 1),
                "max": round(max(latencies), 1) if latencies else 0,
            },
            "failure_samples": failures[:3],
            "note": "Set SKIP_AUTO_EVALUATION=true on the API server for retrieval-only latency tests.",
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
