from pathlib import Path

BOM = b"\xef\xbb\xbf"
for f in ["data/eval/golden.jsonl",
          "data/eval/behavioral.jsonl",
          "data/eval/training-index.jsonl",
          "evaluation/params.json"]:
    p = Path(f)
    if not p.exists():
        print(f"MISSING {f}")
        continue
    b = p.read_bytes()
    print(f"{f}: size={len(b)} bom={b[:3] == BOM} first={b[:16].hex()}")
    try:
        first = p.read_text(encoding="utf-8").splitlines()[0][:120]
        print(f"  first line: {first!r}")
    except Exception as e:
        print(f"  read_text error: {e}")
