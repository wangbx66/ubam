import csv
import re
import json
from constant import Zones
from logstats import record
from logstats import update
from logstats import clear

def zone_attr():
    '''
    Generates constant_zone, which infers the possible continent, area, type, and controllers associated to each zone
    '''
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
    with open('constant_zone.py', 'w') as fw:
        fw.write('# -*- coding: utf-8 -*-\n\n')
        fw.write('Continents = {0}\n'.format(str(continents)))
        fw.write('Areas = {0}\n'.format(str(areas)))
        fw.write('Zonetypes = {0}\n'.format(zonetypes))
        fw.write('Lords = {0}\n'.format(lords))

def zone_match():
    '''
    Generate zonesjson and lvlsjson. The former maps a zone ID to it's attributes, represented by categorical or numeral variables. The latter maps a level to a set of zone ID which a player on that level is able to access.
    '''
    from constant_zone import Continents
    from constant_zone import Areas
    from constant_zone import Zonetypes
    from constant_zone import Lords
    zones = {}
    lvls = {x:set() for x in range(1, 151)}
    for line in csv.reader(open('data/zones.csv'), delimiter=',', quotechar='"'):
        s = record(line, style='zone')
        if not s.zone in Zones:
            print('zone "{0}" not found'.format(s.zone))
            continue
        else:
            zone = Zones[s.zone]
        continent = Continents[s.continent]
        area = Areas[s.area]
        zonetype = Zonetypes[s.zonetype]
        lord = Lords[s.lord]
        for x in range(s.lvl_entry, 151):
            lvls[x].add(zone)
        zones[zone] = (continent, area, zonetype, lord, s.lvl_entry, s.lvl_rec_min, s.lvl_rec_max, s.lvl_npc_min, s.lvl_npc_max)
    lvls = {x:list(lvls[x]) for x in lvls}
    with open('data/zonesjson', 'w') as fw:
        fw.write(json.dumps(zones))
    with open('data/lvlsjson', 'w') as fw:
        fw.write(json.dumps(lvls))

def zone_adj():
    coords = {}
    for line in csv.reader(open('data/location_coords.csv'), delimiter=',', quotechar='"'):
        location, mapid, x, y, z = line
        if ':' in location:
            loc1 = location.split(':')[0].strip()
            loc2 = location.split(':')[1].strip()
        else:
            loc1 = location.split(':')[0].strip()
            loc2 = ''
        x = float(x)
        y = float(y)
        z = float(z)
        for zone in Zones:
            if zone in loc2:
                coords[zone] = (x, y, z)
            elif zone in loc1 and not zone in coords:
                coords[zone] = (x, y, z)
    with open('data/coordsjson', 'w') as fw:
        fw.write(json.dumps(coords))

if __name__ == '__main__':
    '''
    This script should be called once; It should instantly finish.
    '''
    zone_attr()
    zone_match()
    zone_adj()
