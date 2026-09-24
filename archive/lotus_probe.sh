#!/bin/bash
echo "NODE=$(hostname)  CORES=$(nproc)  MEM=$(free -g | awk '/Mem:/{print $2}')GB  FREE=$(free -g | awk '/Mem:/{print $7}')GB"
[ -d ~/Solar_Workspace ] && echo "GWS=mounted" || echo "GWS=MISSING"
[ -f ~/.config/earthengine/credentials ] && echo "EE_CREDS=present" || echo "EE_CREDS=MISSING"
timeout 10 bash -c 'echo > /dev/tcp/earthengine.googleapis.com/443' 2>/dev/null && echo "GEE_TCP=reachable" || echo "GEE_TCP=BLOCKED"
~/Solar_Workspace/envs/env_solar/bin/python - <<'PY' 2>&1 | grep -iE "EE_INIT|Error|Traceback" | head -3
import ee
ee.Initialize(project="ee-abdullahr-solar")
print("EE_INIT=OK", ee.Number(21).multiply(2).getInfo())
PY
