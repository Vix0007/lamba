"""setcfg.py <yaml> key=value ... : set (or add under the train block after 'resume:') simple scalar keys."""
import re, sys
f = sys.argv[1]; s = open(f).read()
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    if re.search(rf"(?m)^  {k}:", s): s = re.sub(rf"(?m)^  {k}:.*$", f"  {k}: {v}", s)
    else: s = s.replace("  resume:", f"  {k}: {v}\n  resume:", 1)
open(f, "w").write(s)
