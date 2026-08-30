#!/bin/sh
# Is a trainer actually running? pgrep -f "train/train.py" matches the shell
# that ran the pgrep, so it answers yes when nothing is training -- which is
# how three hours went by after a container restart with the run dead and the
# check reporting ALIVE. Match python processes only, and exclude this script.
me=$$
found=""
for p in $(pgrep -x python3 2>/dev/null); do
  [ "$p" = "$me" ] && continue
  case "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)" in
    *train/train.py*) found="$found $p";;
  esac
done
if [ -n "$found" ]; then echo "TRAINING$found"; exit 0; fi
echo "NOT TRAINING"; exit 1
