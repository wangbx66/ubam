from __future__ import division

import csv
import re
import json
from logstats import Zones
from logstats import clear

pp = re.compile(r'<U\+(?P<code>[0-9A-F]{4})>')
repl = r'\u\g<code>'

Continents = {'Northrend': 0, 'Other': 1, 'Eastern Kingdoms': 2, 'Ouland': 3, 'Kalimdor': 4, 'The Great Sea': 5, 'Outland': 6}
Areas = {"Quel'Thalas": 0, 'Central Kalimdor': 1, 'Lordaeron': 2, 'Azeroth': 3, 'Northern Kalimdor': 4, 'Khaz Modan': 5, 'Outland': 6, 'The Forbidding Sea': 7, 'Northrend': 8, 'Null': 9, 'Southern Kalimdor': 10, 'The Veiled Sea': 11}
Zonetypes = {'Arena': 0, 'City': 1, 'Zone': 2, 'Dungeon': 3, 'Transit': 4, 'Sea': 5, 'Battleground': 6, 'Event': 7}
Lords = {'PvP': 0, 'Contested': 1, 'Horde': 2, 'Alliance': 3, 'Sanctuary': 4}

def zoneattr():
    fp = open('data/zones.txt')
    fp.next()

    continents = {'Null': 0}
    areas = {'Null': 0}
    zonetypes = {'Null': 0}
    lords = {'Null': 0}

    g = csv.reader(fp, delimiter=',', quotechar='"')
    for s in g:
        if not s:
            continue
        zone, continent, area, alter, subzone, zonetype, size, lord, lvl_entry, lvl_rec_min, lvl_rec_max, lvl_npc_min, lvl_npc_max = s
        zone = re.sub(pp, repl, zone)
        zone = zone.decode('unicode-escape').encode('utf-8')
        if not continent in continents:
            continents[continent] = max(continents.values()) + 1
        if not area in areas and area:
            areas[area] = max(areas.values()) + 1
        if not zonetype in zonetypes:
            zonetypes[zonetype] = max(zonetypes.values()) + 1
        if not lord in lords:
            lords[lord] = max(lords.values()) + 1
    clear(continents, False)
    clear(areas, True)
    clear(zonetypes, False)
    clear(lords, False)
    with open('tmp.txt', 'w') as fw:
        fw.write(str(continents))
        fw.write('\n')
        fw.write(str(areas))
        fw.write('\n')
        fw.write(str(zonetypes))
        fw.write('\n')
        fw.write(str(lords))
        fw.write('\n')
    return continents, areas, zonetypes, lords

def zonematch():
    fp = open('data/zones.txt')
    fp.next()

    zones = {}
    lvls = {x+1:set() for x in range(100)}
    alters = set()
    g = csv.reader(fp, delimiter=',', quotechar='"')
    for s in g:
        if not s:
            continue
        zone, continent, area, alter, subzone, zonetype, size, lord, lvl_entry, lvl_rec_min, lvl_rec_max, lvl_npc_min, lvl_npc_max = s
        zone = re.sub(pp, repl, zone)
        zone = zone.decode('unicode-escape').encode('utf-8')
        
        if not zone in Zones:
            print('zone "{0}" not found'.format(zone))
            continue
        else:
            zone = Zones[zone]

        lvl_entry = int(lvl_entry)
        try:
            lvl_npc_min = int(lvl_npc_min)
            assert lvl_npc_min > 0
        except:
            lvl_npc_min = 1
        try:
            lvl_npc_max = int(lvl_npc_max)
            assert lvl_npc_max > 0
        except:
            lvl_npc_max = 100
        try:
            lvl_rec_min = int(lvl_rec_min)
            assert lvl_rec_min > 0
        except:
            lvl_rec_min = lvl_npc_min
        try:
            lvl_rec_max = int(lvl_rec_max)
            assert lvl_rec_max > 0
        except:
            lvl_rec_max = lvl_npc_max
        continent = Continents[continent]
        if not area:
            area = 'Null'
        area = Areas[area]
        zonetype = Zonetypes[zonetype]
        lord = Lords[lord]
        for x in range(lvl_entry, lvl_rec_max + 1):
            lvls[x].add(zone)
        zones[zone] = (continent, area, zonetype, lord, lvl_entry, lvl_rec_min, lvl_rec_max, lvl_npc_min, lvl_npc_max)
    lvls = {x:list(lvls[x]) for x in lvls}
    with open('data/zonesjson.txt', 'w') as fw:
        fw.write(json.dumps(zones))
        fw.write('\n')
        fw.write(json.dumps(lvls))
    return zones, lvls

if __name__ == '__main__':
    zonematch()
    pass
