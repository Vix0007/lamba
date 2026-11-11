"""
Run antidoom's own exact-match repetition detector (repetition.py) against ALL of
v3's and v4's rollouts (unlike v5, we don't have a pre-filtered "loop_poisoned"
label file for these -- every rollout needs to be checked fresh).
"""
import json
import sys

sys.path.insert(0, "/home/ubuntu/antidoom/src")
from antidoom.repetition import find_inner_repetition

SOURCES = {
    "v3": {
        "rollouts": "/home/ubuntu/antidoom_data/v3/real_rollouts.jsonl",
        "out_loop_extracted": "/home/ubuntu/antidoom_data/v3/loop_extracted.jsonl",
        "out_not_detected": "/home/ubuntu/antidoom_data/v3/not_detected.jsonl",
    },
    "v4": {
        "rollouts": "/home/ubuntu/antidoom_data/v4/real_rollouts_v4_fleet.jsonl",
        "out_loop_extracted": "/home/ubuntu/antidoom_data/v4/loop_extracted.jsonl",
        "out_not_detected": "/home/ubuntu/antidoom_data/v4/not_detected.jsonl",
    },
}


def run(source_name):
    cfg = SOURCES[source_name]
    detected_f = open(cfg["out_loop_extracted"], "w")
    not_detected_f = open(cfg["out_not_detected"], "w")

    total = 0
    detected = 0
    for line in open(cfg["rollouts"]):
        d = json.loads(line)
        total += 1
        text = d.get("rollout_text", "")
        found, hit = find_inner_repetition(text)
        rollout_id = d.get("rollout_id")
        pilot_id = d.get("pilot_id")
        domain = d.get("domain")
        if found:
            detected += 1
            row = {
                "rollout_id": rollout_id,
                "pilot_id": pilot_id,
                "domain": domain,
                "rollout_text": text,
                "detected_span": hit.snippet,
                "detected_start_char": hit.repeat_start,
                "period": hit.period,
                "repeats": hit.repeats,
            }
            detected_f.write(json.dumps(row, default=str) + "\n")
        else:
            row = {
                "rollout_id": rollout_id,
                "pilot_id": pilot_id,
                "domain": domain,
                "rollout_text": text,
                "detected_span": None,
            }
            not_detected_f.write(json.dumps(row, default=str) + "\n")
        if total % 2000 == 0:
            print(f"  [{source_name}] {total} scanned, {detected} exact-match hits so far")

    detected_f.close()
    not_detected_f.close()
    print(f"[{source_name}] Done: {total} total, {detected} exact-match hits ({detected/total:.1%})")


if __name__ == "__main__":
    for src in ["v3", "v4"]:
        run(src)
