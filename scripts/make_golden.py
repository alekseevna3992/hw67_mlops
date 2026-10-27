import json
from pathlib import Path

rows = []
topics = ["cardiology", "pulmonology", "endocrinology", "gastro", "nephrology"]
for topic in topics:
    for i in range(10):
        rows.append({
            "id": f"{topic}-{i:03d}",
            "group_id": f"golden-{topic}-{i}",
            "input": f"[REPLACE] {topic} question {i}: A - option1, B - option2, C - option3. choices: A, B, C, REFUSE.",
            "expected": "A",
            "choices": ["A", "B", "C", "REFUSE"],
            "slices": [topic, "short"],
            "source": "TEMPLATE - replace with verified NMO question",
        })

p = Path("data/eval/golden.jsonl")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
print(f"wrote {len(rows)} rows to {p}")