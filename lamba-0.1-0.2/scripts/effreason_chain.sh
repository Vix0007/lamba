#!/bin/bash
until grep -q "EFFREASON_GPQA DONE" ~/effreason_gpqa.log 2>/dev/null; do sleep 20; done
bash ~/antidoom_data/effreason_malaymmlu.sh
