import argparse
import datetime as dt
import time
import pandas as pd
import tweepy
from .config import settings


# Tweepy client for Twitter/X API v2
def get_client() -> tweepy.Client:
    if not settings.BEARER:
        settings.validate()
    # Let Tweepy pace requests; we still add our own backoff below.
    return tweepy.Client(bearer_token=settings.BEARER, wait_on_rate_limit=True)


def _to_list(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def fetch_hashtag(
    hashtag: str,
    hours: int = 6,
    max_results: int = 100,  # <= 100 per API call; keep small to avoid 429s
) -> pd.DataFrame:
    """
    Fetch recent tweets for a given hashtag within the last N hours.
    Returns a pandas DataFrame with useful fields.
    """
    client = get_client()

    # timezone-aware UTC and ONLY set start_time (avoid end_time ">=10s" error)
    now_utc = dt.datetime.now(dt.timezone.utc)
    start_time = now_utc - dt.timedelta(hours=hours)

    # Simple query; while debugging keep it light
    query = f"#{hashtag} -is:reply -is:quote lang:en"

    tweets = []
    # Keep page size modest; Tweepy paginates automatically
    page_size = min(int(max_results), 100)

    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=["created_at", "public_metrics", "entities", "referenced_tweets"],
        user_fields=["username", "public_metrics", "verified"],
        expansions=[
            "author_id",
            "entities.mentions.username",
            "referenced_tweets.id",
            "in_reply_to_user_id",
        ],
        max_results=page_size,
        start_time=start_time,
    )

    count = 0

    # Iterate with explicit rate-limit handling
    for page in paginator:
        try:
            if page.data is None:
                break

            users = {u.id: u for u in _to_list(page.includes.get("users"))}
            # included tweets not used but kept for completeness
            _ = {t.id: t for t in _to_list(page.includes.get("tweets"))}

            for t in page.data:
                author = users.get(t.author_id)
                metrics = t.public_metrics or {}
                entities = t.entities or {}
                text = t.text or ""

                tag_list = [h["tag"] for h in entities.get("hashtags", [])]
                mentions = [m["username"] for m in entities.get("mentions", [])]

                is_retweet = False
                if t.referenced_tweets:
                    for ref in t.referenced_tweets:
                        if ref["type"] == "retweeted":
                            is_retweet = True
                            break

                followers_count = 0
                if author and getattr(author, "public_metrics", None):
                    followers_count = int(author.public_metrics.get("followers_count", 0))

                row = {
                    "tweet_id": str(t.id),
                    "author_id": str(t.author_id),
                    "username": getattr(author, "username", None),
                    "created_at": t.created_at,
                    "text": text,
                    "like_count": int(metrics.get("like_count", 0)),
                    "retweet_count": int(metrics.get("retweet_count", 0)),
                    "reply_count": int(metrics.get("reply_count", 0)),
                    "quote_count": int(metrics.get("quote_count", 0)),
                    "followers_count": followers_count,
                    "verified": bool(getattr(author, "verified", False)),
                    "hashtags": tag_list,
                    "mentions": mentions,
                    "is_retweet": is_retweet,
                }
                tweets.append(row)

                count += 1
                if count >= max_results:
                    break
            if count >= max_results:
                break

        except tweepy.errors.TooManyRequests as e:
            # Sleep until the rate limit resets (header is epoch seconds)
            reset_epoch = 0
            try:
                reset_epoch = int(e.response.headers.get("x-rate-limit-reset", "0"))
            except Exception:
                pass
            sleep_for = max(30, int(reset_epoch - time.time()) + 5) if reset_epoch else 60
            print(f"[Rate limited] Sleeping for {sleep_for}s…")
            time.sleep(sleep_for)
            continue  # retry the loop after sleeping

    df = pd.DataFrame(tweets)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def save_df(df: pd.DataFrame, hashtag: str, hours: int) -> str:
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d_%Hh%M")
    fname = f"data/raw/{hashtag}_{ts}_{hours}h.csv"
    df.to_csv(fname, index=False)
    return fname


def main():
    parser = argparse.ArgumentParser(description="Fetch tweets for a hashtag")
    parser.add_argument("--hashtag", required=True, help="Hashtag without # e.g. F1")
    parser.add_argument("--hours", type=int, default=3, help="Lookback window in hours")
    parser.add_argument(
        "--max-results", type=int, default=100,
        help="Max tweets to fetch (API page limit 100). Keep small on basic tiers."
    )
    args = parser.parse_args()

    df = fetch_hashtag(args.hashtag, hours=args.hours, max_results=args.max_results)
    if df.empty:
        print("No tweets found. Try increasing hours, reducing popularity, or a different tag.")
        return
    path = save_df(df, args.hashtag, args.hours)
    print(f"Saved {len(df)} tweets to {path}")


if __name__ == "__main__":
    main()
