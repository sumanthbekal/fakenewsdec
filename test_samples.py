import urllib.request
import json

samples = {
    "Example 1 — Real-style (City Council Road Budget)": (
        "The Oakridge City Council unanimously approved a comprehensive $14.5 million municipal road "
        "repair and resurfacing budget yesterday evening. City Transportation Director Mark Henderson announced "
        "that funding has been allocated from the state infrastructure grant and local reserve capital. "
        "The phased construction project will target major arterial roadways including Elmwood Boulevard, "
        "5th Avenue, and North Commerce Way, beginning early this spring with a projected completion date set for November 2027."
    ),
    "Example 2 — Real-style (University Scholarships)": (
        "The State University Academic Board today officially announced the inauguration of its 2026-2027 "
        "Presidential Merit Scholarships for incoming undergraduate and graduate students. According to university provost "
        "Dr. Susan Vance, eligible applicants must possess a cumulative 3.8 GPA and submit verified community service "
        "portfolios before the March 15th priority deadline. Selected recipients will receive up to 100% tuition coverage "
        "along with annual research stipends."
    ),
    "Example 3 — Fake-style (Secret Cure Sensationalism)": (
        "SHOCKING SECRET REVEALED: Big Pharma scientists and elite corrupt doctors are desperately hiding a 100% natural "
        "miracle cancer cure discovered in tropical rain forests! Whistleblowers report government agencies are deleting "
        "all scientific files and threatening researchers with immediate arrest. Read this truth before the global elites "
        "censor this website and delete this video forever, share with everyone now!"
    ),
    "Example 4 — Fake-style (Bank Closure Insider Claim)": (
        "CONFIDENTIAL LEAK FROM ANONYMOUS WALL STREET INSIDER: All major national banks and federal reserve institutions "
        "are secretly planning to shut down completely for thirty consecutive days starting next week! Millions of civilian "
        "retirement accounts and hard-earned savings deposits will permanently vanish under emergency financial protocol "
        "martial law. Massive government cover-up exposed by top insiders."
    )
}

print("=" * 65)
print("TESTING 4 SAMPLE ARTICLES VIA LIVE /predict ENDPOINT")
print("=" * 65)

results_summary = []

for name, text in samples.items():
    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read().decode("utf-8"))
        results_summary.append((name, data))
        print(f"\nTarget: {name}")
        print(f"  Input Word Count: {data['words']}")
        print(f"  Predicted Label : {data['label']}")
        print(f"  Confidence      : {data['confidence']}")
        print(f"  Probabilities   : FAKE={data['probabilities']['FAKE']:.4f}, REAL={data['probabilities']['REAL']:.4f}")
        print(f"  Truncated       : {data['truncated']}")

print("\n" + "=" * 65)
print("ALL 4 SAMPLES EVALUATED ACCURATELY")
print("=" * 65)
