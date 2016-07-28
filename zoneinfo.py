from __future__ import division

import csv
import re
from logstats import Zones


#def zonematch():
pp = re.compile(r'<U\+(?P<code>[0-9A-F]{4})>')
repl = r'\u\g<code>'

fp = open('data/zones.txt')
fp.next()

zones = {}
alters = set()
g = csv.reader(fp, delimiter=',', quotechar='"')
for s in g:
    if not s:
        continue
    zone, continent, area, alter, subzone, zonetype, size, lord, lvl_entry, lvl_rec_min, lvl_rec_max, lvl_npc_min, lvl_npc_max = s
    zone = re.sub(pp, repl, zone)
    zone = zone.decode('unicode-escape').encode('utf-8')
    lvl_entry = int(lvl_entry)
    lvl_rec_min = int(lvl_rec_min)
    lvl_rec_max = int(lvl_rec_max)
    lvl_npc_min = int(lvl_npc_min)
    lvl_npc_max - int(lvl_npc_max)
    

    zones[zone] = ()
