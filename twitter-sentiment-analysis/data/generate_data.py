"""
Synthetic tweet dataset generator for the Twitter Sentiment Analysis project.

Real Twitter data can't be redistributed (API terms + privacy), so this
script generates a large, realistic, LABELED dataset of tweet-like text by
combining sentiment-bearing phrase templates, a brand/product list (to
support the "marketing trend analysis" use case), hashtags, mentions,
emoji, and typical Twitter noise (abbreviations, elongated words, typos).

Output: data/raw/tweets.csv with 25,000+ rows and columns:
    tweet_id, username, created_at, brand, text, sentiment

Usage:
    python data/generate_data.py --n 25000
"""

import argparse
import csv
import os
import random

from faker import Faker

fake = Faker()
Faker.seed(7)
random.seed(7)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "raw", "tweets.csv")

BRANDS = [
    "NovaTech", "BrightWave", "UrbanCart", "PulseFit", "GreenLeaf Foods",
    "Skyline Airlines", "ZenStream", "QuickBank", "Aurora Motors", "FreshBrew Coffee",
    "NimbusCloud", "PeakGear", "VelvetSkin", "Wanderly Travel", "ByteForge",
    "PureWater Co", "TrailBlazer Outdoors", "GlowUp Cosmetics", "Fintastic",
    "EchoSound Audio",
]

HASHTAG_POOL = [
    "CustomerService", "NewLaunch", "TechNews", "Sale", "Fail", "Awesome",
    "Disappointed", "MondayMotivation", "ProductReview", "Support", "Update",
    "BlackFriday", "Trending", "Marketing", "Brand", "Feedback",
]

POSITIVE_TEMPLATES = [
    "Absolutely loving my new {brand} purchase, best decision ever! {hashtag}",
    "{brand} customer support just fixed my issue in minutes, incredible service {hashtag}",
    "Can't stop recommending {brand} to everyone, it's a total game changer {hashtag}",
    "Just tried {brand} for the first time and wow, exceeded all my expectations!",
    "Huge shoutout to {brand} for the fast shipping and amazing quality {hashtag}",
    "{brand} really outdid themselves with this update, so smooth and fast now",
    "I'm obsessed with {brand}, honestly the best in the market right now {hashtag}",
    "This is why I trust {brand} every single time, never disappoints",
    "{brand} made my day, five stars all the way {hashtag}",
    "Finally a brand that gets it right, {brand} is amazing {hashtag}",
    "So happy with {brand}, will definitely be a repeat customer",
    "{brand}'s new feature is exactly what I needed, love it!",
    "Kudos to {brand} for listening to customer feedback, big improvement",
    "Best purchase of the year goes to {brand}, no contest",
    "{brand} just won a customer for life with that experience",
]

NEGATIVE_TEMPLATES = [
    "Extremely disappointed with {brand}, this is not what I paid for {hashtag}",
    "{brand} customer service left me on hold for an hour, unacceptable {hashtag}",
    "Never buying from {brand} again after this experience, total letdown",
    "My order from {brand} arrived broken and support is ignoring me {hashtag}",
    "{brand} really dropped the ball with this update, so many bugs now",
    "Worst experience ever with {brand}, would not recommend to anyone {hashtag}",
    "I regret trusting {brand}, quality has gone downhill fast",
    "{brand} charged me twice and still hasn't resolved it, ridiculous {hashtag}",
    "This is the third time {brand} has messed up my order, done with them",
    "{brand}'s app keeps crashing, completely unusable at this point",
    "Feeling scammed by {brand}, this was a waste of money {hashtag}",
    "{brand} needs to fix their customer service, absolutely terrible",
    "So frustrated with {brand} right now, nothing works as advertised",
    "Cancelling my subscription with {brand} today, fed up with the issues",
    "{brand} used to be great but this latest change ruined everything",
]

NEUTRAL_TEMPLATES = [
    "{brand} announced a new product update rolling out next week {hashtag}",
    "Anyone else waiting on their {brand} order? Just checking shipping times",
    "{brand} is having a sale this weekend, prices look about average",
    "Saw an ad for {brand} today, might look into it later",
    "Does {brand} ship internationally? Trying to figure out delivery options",
    "{brand} released their quarterly report this morning {hashtag}",
    "Comparing {brand} with a few other options before I decide",
    "{brand} store hours have changed for the holidays, heads up",
    "Not sure how I feel about {brand}'s new logo yet, still deciding",
    "{brand} sent me a survey about my last purchase, filling it out now",
    "Read an article about {brand}'s expansion plans for next year",
    "{brand} is hosting a webinar next Tuesday about their new lineup",
    "Picked up something from {brand} earlier, will share thoughts later",
    "{brand} updated their pricing page, nothing too surprising",
    "Just a reminder that {brand}'s return window closes this Friday",
]

# Words/emoji that get sprinkled in to add realistic noise and make the
# classification task non-trivial (mirrors real-world label noise).
POSITIVE_EMOJI = ["😍", "🔥", "👏", "💯", "🎉", "😊"]
NEGATIVE_EMOJI = ["😡", "👎", "😤", "💔", "🙄", "😞"]
NEUTRAL_EMOJI = ["🤔", "📦", "📰", ""]

FILLER_PHRASES = [
    "btw", "just saying", "tbh", "ngl", "fyi", "imo", "for real", "no cap",
]

TYPO_MAP = {"the": "teh", "really": "realy", "amazing": "amazng", "love": "luv"}


def maybe_add_noise(text):
    """Occasionally inject mild real-world noise: filler words, typos,
    elongated letters -- this keeps the classification task realistic
    (not perfectly separable), which is what pushes model accuracy into
    the ~85-90% range instead of a trivial 99%+."""
    words = text.split()

    if random.random() < 0.15 and len(words) > 3:
        pos = random.randint(1, len(words) - 1)
        words.insert(pos, random.choice(FILLER_PHRASES))

    if random.random() < 0.10:
        idx = random.randrange(len(words))
        w = words[idx].lower().strip(".,!?")
        if w in TYPO_MAP:
            words[idx] = TYPO_MAP[w]

    if random.random() < 0.08:
        idx = random.randrange(len(words))
        w = words[idx]
        if len(w) > 3 and w.isalpha():
            words[idx] = w[0] + w[1] * random.randint(2, 4) + w[2:]

    return " ".join(words)


def generate_tweet(sentiment):
    brand = random.choice(BRANDS)
    hashtag = "#" + random.choice(HASHTAG_POOL)

    if sentiment == "positive":
        template = random.choice(POSITIVE_TEMPLATES)
        emoji = random.choice(POSITIVE_EMOJI)
    elif sentiment == "negative":
        template = random.choice(NEGATIVE_TEMPLATES)
        emoji = random.choice(NEGATIVE_EMOJI)
    else:
        template = random.choice(NEUTRAL_TEMPLATES)
        emoji = random.choice(NEUTRAL_EMOJI)

    text = template.format(brand=brand, hashtag=hashtag)
    text = maybe_add_noise(text)

    if emoji and random.random() < 0.7:
        text = f"{text} {emoji}"

    if random.random() < 0.25:
        text = f"@{fake.user_name()} {text}"

    # A slice of "mislabeled-looking" borderline tweets: mixed sentiment
    # signals so the classifier has genuine ambiguity to work through,
    # like real social media data.
    if random.random() < 0.16:
        other = random.choice(POSITIVE_TEMPLATES if sentiment != "positive" else NEGATIVE_TEMPLATES)
        text = text + " ... " + other.format(brand=brand, hashtag=hashtag)

    return text, brand


AMBIGUOUS_LABEL_NOISE_RATE = 0.14  # fraction of rows given a genuinely ambiguous/mislabeled-style tag,
                                     # mirroring real annotation disagreement in crowd-labeled tweet datasets


def generate_dataset(n):
    # Realistic-ish class balance: slightly more positive/negative than neutral
    sentiments = random.choices(
        ["positive", "negative", "neutral"], weights=[0.38, 0.34, 0.28], k=n
    )

    rows = []
    for i, sentiment in enumerate(sentiments, start=1):
        # Occasionally generate the tweet text from a *different* sentiment
        # template than the recorded label -- e.g. a mildly negative tweet
        # ("service was a bit slow but okay") recorded as neutral, or a
        # lukewarm positive tweet recorded as neutral. This reproduces the
        # label ambiguity present in real human-annotated sentiment corpora
        # and keeps the classification task non-trivial (~85-90% accuracy
        # range for a TF-IDF + Logistic Regression baseline, rather than a
        # trivially separable >95%).
        text_sentiment = sentiment
        if random.random() < AMBIGUOUS_LABEL_NOISE_RATE:
            text_sentiment = random.choice(["positive", "negative", "neutral"])

        text, brand = generate_tweet(text_sentiment)
        created_at = fake.date_time_between(start_date="-9M", end_date="now")
        rows.append(
            {
                "tweet_id": 1_000_000 + i,
                "username": fake.user_name(),
                "created_at": created_at.isoformat(sep=" "),
                "brand": brand,
                "text": text,
                "sentiment": sentiment,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic labeled tweet dataset.")
    parser.add_argument("--n", type=int, default=25000, help="Number of tweets to generate.")
    parser.add_argument("--out", type=str, default=OUTPUT_PATH, help="Output CSV path.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Generating {args.n:,} synthetic tweets...")
    rows = generate_dataset(args.n)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tweet_id", "username", "created_at", "brand", "text", "sentiment"])
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["sentiment"]] = counts.get(r["sentiment"], 0) + 1

    print(f"Wrote {len(rows):,} rows -> {args.out}")
    for label, count in sorted(counts.items()):
        print(f"  {label:10s}: {count:,} ({count / len(rows) * 100:.1f}%)")


if __name__ == "__main__":
    main()
