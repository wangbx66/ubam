from __future__ import division

import csv
import re
import json
from logstats import Zones
from logstats import clear

Continents = {'Northrend': 0, 'Other': 1, 'Eastern Kingdoms': 2, 'Ouland': 3, 'Kalimdor': 4, 'The Great Sea': 5, 'Outland': 6}
Areas = {"Quel'Thalas": 0, 'Central Kalimdor': 1, 'Lordaeron': 2, 'Azeroth': 3, 'Northern Kalimdor': 4, 'Khaz Modan': 5, 'Outland': 6, 'The Forbidding Sea': 7, 'Northrend': 8, 'Null': 9, 'Southern Kalimdor': 10, 'The Veiled Sea': 11}
Zonetypes = {'Arena': 0, 'City': 1, 'Zone': 2, 'Dungeon': 3, 'Transit': 4, 'Sea': 5, 'Battleground': 6, 'Event': 7}
Lords = {'PvP': 0, 'Contested': 1, 'Horde': 2, 'Alliance': 3, 'Sanctuary': 4}

def zoneattr():
    continents = {'Null': 0}
    areas = {'Null': 0}
    zonetypes = {'Null': 0}
    lords = {'Null': 0}
    for line in csv.reader(open('data/zones.csv'), delimiter=',', quotechar='"'):
        s = record(line, style='zone')
        update(s.continent, continents)
        update(s.area, areas)
        update(s.zonetype, zonetypes)
        update(s.lord, lords)
    clear(continents, {}, 0, False)
    clear(areas, {}, 0, True)
    clear(zonetypes, {}, 0, False)
    clear(lords, {}, 0, False)
    with open('constant_zone.txt', 'w') as fw:
        fw.write('# -*- coding: utf-8 -*-\n\n')
        fw.write('Continents = {0}\n'.format(str(continents)))
        fw.write('Areas = {0}\n'.format(str(areas)))
        fw.write('Zonetypes = {0}\n'.format(zonetypes))
        fw.write('Lords = {0}\n'.format(lords))

def zonematch():
    fp = open('data/zones.csv')
    fp.__next__()

    zones = {}
    lvls = {x + 1:set() for x in range(100)}
    alters = set()
    g = csv.reader(fp, delimiter=',', quotechar='"')
    for s in g:
        if not s:
            continue
        zone, continent, area, alter, subzone, zonetype, size, lord, lvl_entry, lvl_rec_min, lvl_rec_max, lvl_npc_min, lvl_npc_max = s
        zone = re.sub(pp, repl, zone)
        zone = zone.encode('utf-8').decode('unicode-escape')

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
        #for x in range(lvl_entry, lvl_rec_max + 1):
        for x in range(lvl_entry, 100 + 1):
            lvls[x].add(zone)
        zones[zone] = (continent, area, zonetype, lord, lvl_entry, lvl_rec_min, lvl_rec_max, lvl_npc_min, lvl_npc_max)
    lvls = {x:list(lvls[x]) for x in lvls}
    with open('data/zonesjson.txt', 'w') as fw:
        fw.write(json.dumps(zones))
        fw.write('\n')
        fw.write(json.dumps(lvls))
    return zones, lvls

def zoneadj():
    locations = {}
    fp = open('data/location_coords.csv')
    fp.__next__()
    g = csv.reader(fp, delimiter=',', quotechar='"')
    for s in g:
        if not s:
            continue
        location, mapid, x, y, z = s
        if ':' in location:
            loc1 = location.split(':')[0].strip()
            loc2 = location.split(':')[1].strip()
        else:
            loc2 = location.split(':')[0].strip()
            loc2 = ''
        try:
            mapid = int(mapid)
        except ValueError:
            continue
        x = float(x)
        y = float(y)
        z = float(z)
        for zone in Zones:
            if zone in loc2:
                locations[zone] = (x, y, z)
            elif zone in loc1:
                if not zone in locations:
                    locations[zone] = (x, y, z)
        s = set(Zones.keys())
        t = set(locations.keys())


if __name__ == '__main__':
    zones, lvls = zonematch()
    pass
