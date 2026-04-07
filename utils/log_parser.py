import json

with open("1.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

with open("pretty.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)