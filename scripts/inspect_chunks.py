from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from retrieval import load_seed_records


records = load_seed_records()
lengths = [len(item["text"]) for item in records]
sections = Counter(item["section"] for item in records)

print("chunks:", len(records))
print("min chars:", min(lengths))
print("mean chars:", sum(lengths) / len(lengths))
print("max chars:", max(lengths))
print("most repeated section labels:")

for section, count in sections.most_common(20):
    print(count, repr(section))

for item in records[:20]:
    print(
        "\n---",
        item["source"],
        "|",
        item["section"],
        "|",
        item["chunk_index"],
    )
    print(item["text"])