from __future__ import division

import os
import math
import json

#import matplotlib.pyplot as plt

Guilds = 512
Races = {'Blood Elf':0, 'Orc':1, 'Tauren':2, 'Troll':3, 'Undead':4}
Categories = {'Death Knight':0, 'Druid':1, 'Hunter':2, 'Mage':3, 'Paladin':4, 'Priest':5, 'Rogue':6, 'Shaman':7, 'Warlock':8, 'Worrier':9}
Zones = {"Quel'thalas": 0, 'Coilfang: The Slave Pens': 1, 'Uldaman': 2, 'The Steamvault': 3, 'Maraudon': 4, 'Molten Core': 5, "Ahn'kahet: The Old Kingdom": 6, 'Tempest Keep': 7, 'Auchindoun: Shadow Labyrinth': 8, 'Blackfathom Deeps': 9, "Blade's Edge Mountains": 10, 'Deeprun Tram': 11, 'Naxxramas': 12, 'The Blood Furnace': 13, 'The Frozen Sea': 14, 'Winterspring': 15, "Gruul's Lair": 16, 'Old Hillsbrad Foothills': 17, 'Swamp of Sorrows': 18, 'Badlands': 19, 'Razorfen Kraul': 20, 'Searing Gorge': 21, 'Dun Morogh': 22, "Gates of Ahn'Qiraj": 23, 'Stormwind City': 24, 'Scholomance': 25, 'Azjol-Nerub': 26, 'Ghostlands': 27, "Onyxia's Lair": 28, 'Hall of Legends': 29, 'Deadwind Pass': 30, 'Nagrand Arena': 31, 'The Culling of Stratholme': 32, 'Icecrown': 33, 'Deadmines': 34, 'Dragonblight': 35, 'The Exodar': 36, 'The Arcatraz': 37, 'Durotar': 38, 'Ragefire Chasm': 39, 'Ironforge': 40, 'Elwynn Forest': 41, 'The Underbog': 42, '\xe6\xaf\x92\xe7\x89\x99\xe6\xb2\xbc\xe6\xbe\xa4': 43, 'The Barrens': 44, 'Halls of Stone': 45, '\xe9\xbe\x8d\xe9\xaa\xa8\xe8\x8d\x92\xe9\x87\x8e': 46, 'Auchenai Crypts': 47, '\xe9\x81\x94\xe7\xb4\x8d\xe8\x98\x87\xe6\x96\xaf': 48, 'The Veiled Sea': 49, 'Dire Maul': 50, 'Blackrock Spire': 51, 'Darkshore': 52, 'The Storm Peaks': 53, '\xe9\xba\xa5\xe5\x85\x8b\xe9\x82\xa3\xe7\x88\xbe': 54, 'Utgarde Pinnacle': 55, 'Sholazar Basin': 56, 'Strand of the Ancients': 57, 'Scarlet Monastery': 58, 'Mulgore': 59, '\xe7\x9b\xa3\xe7\x8d\x84': 60, 'Desolace': 61, 'Tirisfal Glades': 62, "Zul'Gurub": 63, 'The Botanica': 64, 'Shattrath City': 65, "The Temple of Atal'Hakkar": 66, 'The Black Morass': 67, 'Blackwing Lair': 68, 'Nagrand': 69, 'Eastern Plaguelands': 70, 'Arathi Highlands': 71, '\xe6\x9c\xaa\xe7\x9f\xa5': 72, 'Tanaris': 73, 'Azshara': 74, 'Ashenvale': 75, 'Stratholme': 76, 'Burning Steppes': 77, 'Silithus': 78, '\xe5\x8c\x97\xe6\x96\xb9\xe6\xb5\xb7\xe5\xb2\xb8': 79, 'Dalaran\xe7\xab\xb6\xe6\x8a\x80\xe5\xa0\xb4': 80, "Zul'Drak": 81, 'Shadowfang Keep': 82, 'The North Sea': 83, 'Ruins of Lordaeron': 84, 'Sethekk Halls': 85, "Ruins of Ahn'Qiraj": 86, 'Shadowmoon Valley': 87, 'Eye of the Storm': 88, 'Felwood': 89, 'Hyjal': 90, 'Dalaran': 91, 'Razorfen Downs': 92, 'The Obsidian Sanctum': 93, 'The Mechanar': 94, 'Crystalsong Forest': 95, "Magisters' Terrace": 96, "Zul'Farrak": 97, 'Duskwood': 98, 'Thousand Needles': 99, 'Teldrassil': 100, "Blade's Edge Arena": 101, 'Warsong Gulch': 102, 'Borean Tundra': 103, "Un'Goro Crater": 104, 'Netherstorm': 105, 'The Shattered Halls': 106, 'Feralas': 107, 'Alterac Valley': 108, 'Sunwell Plateau': 109, 'Plaguelands: The Scarlet Enclave': 110, 'Wetlands': 111, 'Orgrimmar': 112, 'Blackrock Depths': 113, 'Utgarde Keep': 114, 'Howling Fjord': 115, "Ahn'Qiraj": 116, 'Bloodmyst Isle': 117, 'Hellfire Ramparts': 118, 'Moonglade': 119, 'Grizzly Hills': 120, '\xe6\x99\x82\xe5\x85\x89\xe6\xb4\x9e\xe7\xa9\xb4': 121, 'Azuremyst Isle': 122, 'Twisting Nether': 123, 'Wintergrasp': 124, 'The Nexus': 125, 'Gundrak': 126, 'GM Island': 127, 'Undercity': 128, 'The Eye of Eternity': 129, 'The Violet Hold': 130, "Isle of Quel'Danas": 131, 'The Ring of Valor': 132, "Magtheridon's Lair": 133, 'Stonetalon Mountains': 134, 'Hillsbrad Foothills': 135, 'Alterac Mountains': 136, 'Western Plaguelands': 137, 'Blasted Lands': 138, 'Black Temple': 139, 'Blackrock Mountain': 140, 'Zangarmarsh': 141, 'Wailing Caverns': 142, 'Hellfire Peninsula': 143, 'Eversong Woods': 144, 'Redridge Mountains': 145, 'Dustwallow Marsh': 146, 'Silvermoon City': 147, 'Stranglethorn Vale': 148, 'Terokkar Forest': 149, 'Thunder Bluff': 150, 'The Hinterlands': 151, 'Halls of Lightning': 152, 'Karazhan': 153, 'The Great Sea': 154, 'Serpentshrine Cavern': 155, "Drak'Tharon Keep": 156, 'Arathi Basin': 157, 'Silverpine Forest': 158, 'The Oculus': 159, 'Vault of Archavon': 160, 'Gnomeregan': 161, 'Mana-Tombs': 162, 'Westfall': 163, 'Loch Modan': 164}
Cnt = {"Quel'thalas": 100, 'Coilfang: The Slave Pens': 149884, 'Uldaman': 58463, 'The Steamvault': 177904, 'Maraudon': 112407, 'Molten Core': 371204, "Ahn'kahet: The Old Kingdom": 12029, 'Tempest Keep': 180191, 'Auchindoun: Shadow Labyrinth': 228552, 'Blackfathom Deeps': 32793, "Blade's Edge Mountains": 720365, '1608\xe5\xb3\xbd\xe8\xb0\xb7': 22, 'Deeprun Tram': 214, 'Naxxramas': 112279, 'The Blood Furnace': 116867, 'The Frozen Sea': 842, 'Winterspring': 435649, "Gruul's Lair": 101149, 'Old Hillsbrad Foothills': 61113, 'Swamp of Sorrows': 138871, 'Badlands': 203618, 'Razorfen Kraul': 53133, 'Searing Gorge': 144363, '1007\xe5\x9f\x8e': 1, 'Dun Morogh': 2068, "Gates of Ahn'Qiraj": 28869, 'Stormwind City': 1008, 'Scholomance': 249320, 'Azjol-Nerub': 8216, 'Ghostlands': 421959, "Onyxia's Lair": 64539, 'Hall of Legends': 42079, 'Deadwind Pass': 109838, 'Nagrand Arena': 60371, 'The Culling of Stratholme': 14701, 'Icecrown': 49254, 'Deadmines': 3708, 'Dragonblight': 115187, 'The Exodar': 118, 'The Arcatraz': 104756, 'Durotar': 779396, 'Ragefire Chasm': 45876, 'Ironforge': 448, 'Elwynn Forest': 6624, 'The Underbog': 70557, '\xe6\xaf\x92\xe7\x89\x99\xe6\xb2\xbc\xe6\xbe\xa4': 59127, 'The Barrens': 1264010, 'Halls of Stone': 6595, '\xe9\xbe\x8d\xe9\xaa\xa8\xe8\x8d\x92\xe9\x87\x8e': 198382, 'Auchenai Crypts': 45255, '\xe9\x81\x94\xe7\xb4\x8d\xe8\x98\x87\xe6\x96\xaf': 302, 'The Veiled Sea': 65, '1231\xe5\xb4\x94\xe8\x8c\xb2': 36, 'Dire Maul': 215301, 'Blackrock Spire': 363330, 'Darkshore': 6683, 'The Storm Peaks': 49939, '\xe9\xba\xa5\xe5\x85\x8b\xe9\x82\xa3\xe7\x88\xbe': 104230, 'Utgarde Pinnacle': 15318, 'Sholazar Basin': 34952, 'Strand of the Ancients': 4389, 'Scarlet Monastery': 191214, 'Mulgore': 265138, '\xe7\x9b\xa3\xe7\x8d\x84': 136, 'Desolace': 224688, 'Tirisfal Glades': 494279, "Zul'Gurub": 491957, 'The Botanica': 142129, 'Shattrath City': 1892294, "The Temple of Atal'Hakkar": 116346, 'The Black Morass': 90150, 'Blackwing Lair': 262529, 'Nagrand': 931646, 'Eastern Plaguelands': 408189, 'Arathi Highlands': 351509, 'The Forbidding Sea': 2, '\xe6\x9c\xaa\xe7\x9f\xa5': 4958, 'Tanaris': 535285, 'Azshara': 284741, 'Ashenvale': 315793, 'Stratholme': 266104, 'Burning Steppes': 103520, 'Silithus': 511335, 'Null': 0, '\xe5\x8c\x97\xe6\x96\xb9\xe6\xb5\xb7\xe5\xb2\xb8': 1694, 'Dalaran\xe7\xab\xb6\xe6\x8a\x80\xe5\xa0\xb4': 229, "Zul'Drak": 53412, 'Shadowfang Keep': 64958, 'The North Sea': 145, 'Ruins of Lordaeron': 59545, 'Sethekk Halls': 112921, "Ruins of Ahn'Qiraj": 172980, 'Shadowmoon Valley': 760166, '15641': 4, 'Eye of the Storm': 215345, 'Felwood': 412741, 'Hyjal': 109326, 'Dalaran': 89983, 'Razorfen Downs': 45408, 'The Obsidian Sanctum': 6347, 'The Mechanar': 217667, 'Crystalsong Forest': 5543, "Magisters' Terrace": 104077, "Zul'Farrak": 97702, 'Duskwood': 28641, 'Thousand Needles': 410527, 'Teldrassil': 577, "Blade's Edge Arena": 59506, 'Warsong Gulch': 471930, 'Borean Tundra': 123957, "Un'Goro Crater": 293136, 'Netherstorm': 688177, 'The Shattered Halls': 140919, 'Feralas': 389370, 'Alterac Valley': 979136, 'Sunwell Plateau': 35213, 'Plaguelands: The Scarlet Enclave': 16591, 'Wetlands': 28837, 'Orgrimmar': 3091805, 'Blackrock Depths': 129071, 'Utgarde Keep': 14178, 'Howling Fjord': 72999, "Ahn'Qiraj": 147127, 'Bloodmyst Isle': 563, 'Hellfire Ramparts': 172469, 'Moonglade': 24921, 'Grizzly Hills': 44906, '\xe6\x99\x82\xe5\x85\x89\xe6\xb4\x9e\xe7\xa9\xb4': 220, 'Azuremyst Isle': 635, 'Twisting Nether': 7435, '8585': 21, '2029': 8, '61477': 2, 'Wintergrasp': 14931, 'The Nexus': 22434, 'Gundrak': 10067, 'GM Island': 1816, 'Undercity': 782917, 'The Eye of Eternity': 1622, 'The Violet Hold': 14639, "Isle of Quel'Danas": 315415, 'The Ring of Valor': 195, "Magtheridon's Lair": 34742, 'Stonetalon Mountains': 267015, 'Hillsbrad Foothills': 527771, 'Alterac Mountains': 169478, 'Western Plaguelands': 395058, 'Blasted Lands': 74443, 'Black Temple': 111394, 'Blackrock Mountain': 117003, 'Zangarmarsh': 700613, 'Wailing Caverns': 104774, 'Hellfire Peninsula': 952413, 'Eversong Woods': 504236, 'Redridge Mountains': 10486, 'Dustwallow Marsh': 173900, 'Silvermoon City': 178245, 'Stranglethorn Vale': 1063264, 'Terokkar Forest': 983116, 'Thunder Bluff': 337417, 'The Hinterlands': 364793, 'Halls of Lightning': 14902, 'Karazhan': 897530, 'The Great Sea': 99, 'Serpentshrine Cavern': 262185, "Drak'Tharon Keep": 8334, 'Arathi Basin': 965258, 'Silverpine Forest': 260783, 'The Oculus': 7622, 'Vault of Archavon': 2496, 'Gnomeregan': 15595, 'Mana-Tombs': 84220, 'Westfall': 2646, 'Loch Modan': 11949}
Mintt = 1136069986
Lktt = 1226552400
Total = 36513648

def clear(dd, null=False):
    if 'Null' in dd and not null:
        del dd['Null']
    z = list(dd.values())
    m = {z[i]:i for i in range(len(z))}
    for x in dd:
        dd[x] = m[dd[x]]

def keymatch():
    races = {'Null':0}
    categories = {'Null':0}
    zones = {'Null':0}
    cnt = {'Null':0}
    fp = open('data/WoWAH_Node_Player_Fixed_Dynamic')
    ltpl = []
    idx = 99
    maxguild = 0
    mintt = -1
    for line in fp:
        if line.startswith('#') or line.startswith('RowID'):
            continue
        if idx % 10000 == 0:
            print(idx)
        #if idx > 100000:
        #    break
        s = line.strip().split(', ')
        idx, user, tt, guild, lvl, race, category, zone, seq = s
        idx = int(idx)
        user = int(user)
        tt = int(float(tt))
        if mintt < 0:
            mintt = tt
        else:
            mintt = min(mintt, tt)
        if guild == 'Null':
            guild = 0
        else:
            guild = int(guild)
        if guild > maxguild:
            maxguild = guild
        if not race in races:
            races[race] = max(races.values()) + 1
        if not category in categories:
            categories[category] = max(categories.values()) + 1
        if not zone in zones:
            zones[zone] = max(zones.values()) + 1
            cnt[zone] = 1
        else:
            cnt[zone] += 1
        lvl = int(lvl)
        seq = int(seq)
    for zone in cnt:
        if cnt[zone] < 50:
            del zones[zone]
    clear(zones)
    with open('data/tmp.txt', 'w') as fw:
        fw.write(str(zones))
        fw.write('\n')
        fw.write(str(cnt))
    print('maxguild is {0}'.format(maxguild))
    return zones, cnt, mintt, maxguild

def cat_user():
    from shutil import rmtree
    ulvl = {}
    idx = 99
    users = {}
    fp = open('data/WoWAH_Node_Player_Fixed_Dynamic')
    rmtree('data/users')
    os.makedirs('data/users')
    for line in fp:
        if line.startswith('#') or line.startswith('RowID'):
            continue
        if idx % 10000 == 0:
            print("{0}/{1}".format(idx, Total))
        #if idx > 3000:
        #    break
        s = line.strip().split(', ')
        idx, user, tt, guild, lvl, race, category, zone, seq = s
        idx = int(idx)
        user = int(user)
        lvl = int(lvl)
        tt = int(float(tt))
        tt = int(math.floor((tt - Mintt) / 600 + 0.5))
        if guild == 'Null':
            guild = 0
        else:
            guild = int(guild)
        if not race in Races:
            continue
        if not category in Categories:
            continue
        if not zone in Zones:
            continue
        if user in users:
            current = ulvl[user]
            if not lvl >= current:
                continue
            users[user] += 1
            if lvl > current:
                ulvl[user] = lvl
        else:
            users[user] = 1
            ulvl[user] = lvl
        s = (idx, user, tt, guild, lvl, Races[race], Categories[category], Zones[zone], seq)
        buf = ','.join([str(x) for x in s])
        with open('data/users/{0}.txt'.format(user), 'a') as fw:  
            fw.write(buf)
            fw.write('\n')
        if idx % 50000 == 0:
            try:
                with open('data/usersjson_sketch.txt', 'w') as fw:
                    fw.write(json.dumps(users))
            except KeyboardInterrupt:
                pass
    with open('data/usersjson_sketch.txt', 'w') as fw:
        fw.write(json.dumps(users))

def userstats():
    with open('data/usersjson_sketch.txt') as fp:
        user_sketch = json.loads(fp.readline())
    power = 1.8
    scores = []
    directory = 'data/users'
    usersdir = os.listdir(directory)
    totaluser = len(usersdir)
    lvls_start = []
    lvls_end = []
    lvls_elapses = []
    elapses = []
    users = {}
    for i, user in enumerate(usersdir):
        if i % 10000 == 0:
            print("{0}/{1}".format(i, totaluser))
        with open(os.path.join(directory, user)) as fp:
            for idx, line in enumerate(fp):
                if idx == 0:
                    lvl_start = int(line.strip().split(',')[4])
        lvl_end = int(line.strip().split(',')[4])
        elapse = idx + 1
        elapses.append(elapse)
        lvls_start.append(lvl_start)
        lvls_end.append(lvl_end)
        lvls_elapses.append((elapse, lvl_end))

        score = lvl_end ** power - lvl_start ** power
        if score < 0:
            print(user, lvl_start, lvl_end)
        scores.append(score)
        if score / elapse > 1: # but this will exclude those addicted
            users[user.split('.')[0]] = user_sketch[user.split('.')[0]]
    with open('data/usersjson.txt', 'w') as fw:
        fw.write(json.dumps(users))
    #plt.hist(lvls_start, bins=80)
    #plt.hist(lvls_end, bins=80)
    #plt.hist(elapses, bins=80)
    #plt.scatter(elapses, lvls_end)
    #plt.scatter([min(x, 15000) for x in elapses], scores)
    #plt.show()
    return lvls_start, lvls_end, lvls_elapses, scores, elapses

def trajectory():
    zonepair = {}
    lvlup = {}
    userlist = os.listdir('data/users')
    for iuser, user in enumerate(userlist):
        if iuser % 1000 == 0:
            print(iuser)
        with open(os.path.join('data/users', user)) as fp:
            for idx, line in enumerate(fp):
                if idx == 0:
                    previous_zone = int(line.strip().split(',')[7])
                    previous_lvl = int(line.strip().split(',')[4])
                    starting_lvl = previous_lvl
                    cnt = 0
                zone = int(line.strip().split(',')[7])
                lvl = int(line.strip().split(',')[4])
                if not zone == previous_zone:
                    if (previous_zone, zone) in zonepair:
                        zonepair[(previous_zone, zone)] += 1
                    else:
                        zonepair[(previous_zone, zone)] = 1
                    if previous_zone in zonepair:
                        zonepair[previous_zone] += 1
                    else:
                        zonepair[previous_zone] = 1
                    previous_zone = zone
                if previous_lvl > starting_lvl or previous_lvl == 1:
                    cnt += 1
                    if lvl > previous_lvl:
                        if previous_lvl in lvlup:
                            lvlup[previous_lvl][0] += 1
                            lvlup[previous_lvl][1] += cnt
                            cnt = 0
                        else:
                            lvlup[previous_lvl] = [1, cnt]
                if not lvl == previous_lvl:
                    previous_lvl = lvl
    lvlscore = {x: lvlup[x][1] / lvlup[x][0] for x in lvlup}
    with open('data/scorejson.txt', 'w') as fw:
        fw.write(json.dumps(lvlscore))

    for threshold in range(1,20):
        pairs = [x for x in zonepair if type(x) is tuple and zonepair[x]/zonepair[x[0]] > 0.01 * threshold]
        transaction = {}
        for x in pairs:
            if x[0] in transaction:
                transaction[x[0]].append(x[1])
            else:
                transaction[x[0]] = [x[1]]
        with open('data/transactionjson{0}.txt'.format(threshold), 'w') as fw:
            fw.write(json.dumps(transaction))
        avg = sum(len(transaction[x]) for x in transaction)/len(transaction)
        print(threshold, len(transaction), len(Zones), avg)
        for zone in Zones:
            if not Zones[zone] in transaction:
                try:
                    print(zonepair[zone])
                    print("should never print this lolol")
                except:
                    print('never appeared')
                print(Zones[zone], zone)
    return locals()

if __name__ == '__main__':
    #zones, cnt, mintt, maxguild = keymatch()
    #cat_user()
    #s = trajectory()
    #lvls_start, lvls_end, lvls_elapses, scores, elapses = userstats()
    pass
