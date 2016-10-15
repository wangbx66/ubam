import os
import re
import math
import json
import time
import datetime
from shutil import rmtree

p_sub = re.compile(r'<U\+(?P<code>[0-9A-F]{4})>')
p_repl = r'\u\g<code>'

class record:
    def __init__(self, line, style):
        if style == 'raw':
            idx, user, timestamp, guild, lvl, race, category, zone, seq = line.strip().split(', ') # 'category' stands for "class" in WoW
            self.user = int(user)
            self.timestamp = int(float(timestamp))
            self.guild = 0 if guild == 'Null' else int(guild)
            self.lvl = int(lvl)
            self.race = race
            self.category = category
            self.zone = zone
        elif style == 'clean':
            user, tt, guild, lvl, race, category, zone = line.strip().split(',')
            self.user = int(user)
            self.tt = int(tt)
            self.guild = int(guild)
            self.lvl = int(lvl)
            self.race = int(race)
            self.category = int(category)
            self.zone = int(zone)
        elif style == 'sat':
            idx, user, tt, guild, lvl, race, category, zone, seq, zonetype, num_zones, zone_stay, r1, r2, r3, r4, r5
        elif style == 'zone':
            zone, continent, area, alter, subzone, zonetype, size, lord, lvl_entry, lvl_rec_min, lvl_rec_max, lvl_npc_min, lvl_npc_max = line
            zone_tmp = re.sub(p_sub, p_repl, zone)
            self.zone = zone_tmp.encode('utf-8').decode('unicode-escape')
            self.continent = continent
            self.area = 'Null' if area == '' else area
            self.zonetype = zonetype
            self.lord = lord
            self.lvl_entry = int(lvl_entry)
            self.lvl_npc_min = 1 if 'NA' in lvl_npc_min or 0 == int(lvl_npc_min) else int(lvl_npc_min)
            self.lvl_npc_max = 100 if 'NA' in lvl_npc_max or 0 == int(lvl_npc_max) else int(lvl_npc_max)
            self.lvl_rec_min = self.lvl_npc_min if 'NA' in lvl_rec_min or 0 == int(lvl_rec_min) else int(lvl_rec_min)
            self.lvl_rec_max = self.lvl_npc_max if 'NA' in lvl_rec_max or 0 == int(lvl_rec_max) else int(lvl_rec_max)
        elif style == 'zone_clean':
            continent, area, zonetype, lord, lvl_entry, lvl_rec_min, lvl_rec_max, lvl_npc_min, lvl_npc_max = line
            self.continent = int(continent)
            self.area = int(area)
            self.zonetype = int(zonetype)
            self.lord = int(lord)
            self.lvl_entry = int(lvl_entry)
            self.lvl_rec_min = int(lvl_rec_min)
            self.lvl_rec_max = int(lvl_rec_max)
            self.lvl_npc_min = int(lvl_npc_min)
            self.lvl_npc_max = int(lvl_npc_max)

def clear(dct, cnt, thres, keep_null=False):
    if thres > 0:
        for item in cnt:
            if cnt[item] <= thres:
                del dct[item]
    if 'Null' in dct and not keep_null:
        del dct['Null']
    z = list(dct.values())
    keys = {z[i]:i for i in range(len(z))}
    for x in dct:
        dct[x] = keys[dct[x]]

def update(item, dct, cnt={'Null':0}):
    if not item in dct:
        dct[item] = max(dct.values()) + 1
        cnt[item] = 1
    else:
        cnt[item] += 1

def constant_generate():
    '''
    This function should be called once to generate constant.py, which includes several python variable to be imported by other python scripts.
    
    The function filters those minor zones, races, categories, but not guilds. When later importing the constants from other scripts, it should elimite those records containing any minor term.
    '''
    races = {'Null':0}
    race_cnt = {'Null':0}
    categories = {'Null':0}
    category_cnt = {'Null':0}
    zones = {'Null':0}
    zone_cnt = {'Null':0}
    idx = 99
    max_guild = 0
    min_timestamp = -1
    max_timestamp = -1
    for line_idx, line in enumerate(open('data/wowah_dynamic')):
        if line_idx % 5000000 == 0:
            print('line {0}'.format(line_idx))
        s = record(line, style='raw')
        min_timestamp = s.timestamp if min_timestamp < 0 else min(min_timestamp, s.timestamp)
        max_timestamp = s.timestamp if max_timestamp < 0 else max(max_timestamp, s.timestamp)
        max_guild = max(max_guild, s.guild)
        update(s.zone, zones, zone_cnt)
        update(s.race, races, race_cnt)
        update(s.category, categories, category_cnt)
    clear(zones, zone_cnt, 12000, keep_null=False)
    clear(races, race_cnt, 120000, keep_null=False)
    clear(categories, category_cnt, 12000, keep_null=False)
    with open('constant.py', 'w') as fw:
        fw.write('# -*- coding: utf-8 -*-\n\n')
        fw.write('Races = {0}\n'.format(str(races)))
        fw.write('Categories = {0}\n'.format(str(categories)))
        fw.write('Min_timestamp = {0}\n'.format(min_timestamp))
        fw.write('Max_timestamp = {0}\n'.format(max_timestamp))
        fw.write('Lichking_date = {0}\n'.format(1226552400))
        lk_1 = time.mktime(datetime.datetime(year=2008,month=11,day=12,hour=12).timetuple())
        lk_2 = time.mktime(datetime.datetime(year=2008,month=11,day=14,hour=12).timetuple())
        fw.write('Lichking_timestamp = {0}\n'.format((int(lk_1), int(lk_2))))
        fw.write('Lichking_tt = {0}\n'.format((int((lk_1-min_timestamp)/600),int((lk_2-min_timestamp)/600))))
        fw.write('Total_records = {0}\n'.format(line_idx + 1))
        fw.write('Max_guild = {0}\n'.format(max_guild + 1))
        fw.write('Zones = {0}\n'.format(str(zones)))
        fw.write('Zone_cnt = {0}\n'.format(str(zone_cnt)))

def cat_user():
    '''
    Generates data/user/*, where each file separates one specific user's records. In another word each file is the trajectory of one user.
    
    This takes some time especially for those w/o ssd. And this function, once executed, will erase all existing data immediately
    '''
    from constant import Total_records
    from constant import Zones
    from constant import Races
    from constant import Categories
    from constant import Min_timestamp
    rmtree('data/users')
    os.makedirs('data/users')
    user_lvl = {}
    users = {}
    for line_idx, line in enumerate(open('data/wowah_dynamic')):
        if line_idx % 1000000 == 0:
            print("line {0}/{1}".format(line_idx, Total_records))
            open('data/usersjson_sketch.txt', 'w').write(json.dumps(users))
        s = record(line, style='raw')
        tt = math.floor((s.timestamp - Min_timestamp) / 600 + 0.5)
        if not (s.zone in Zones and s.race in Races and s.category in Categories):
            continue
        if s.user in users:
            users[s.user] += 1
            current_lvl = user_lvl[s.user]
            if not s.lvl >= current_lvl:
                continue
            if s.lvl > current_lvl:
                user_lvl[s.user] = s.lvl
        else:
            users[s.user] = 1
            user_lvl[s.user] = s.lvl
        buf = (s.user, tt, s.guild, s.lvl, Races[s.race], Categories[s.category], Zones[s.zone])
        buf_string = ','.join([str(x) for x in buf])
        with open('data/users/{0}'.format(s.user), 'a') as fw:  
            fw.write(buf_string + '\n')
    open('data/usersjson_sketch.txt', 'w').write(json.dumps(users))

def sats():
    from constant import Lichking_tt
    from constant_zone import Zonetypes
    from constant_zone import Lords
    from shutil import rmtree
    rival = len(Zonetypes)
    users_sketch = json.loads(open('data/usersjson_sketch.txt').readline())
    users_sketch = {int(x):users_sketch[x] for x in users_sketch}
    users = {x:0 for x in users_sketch}
    zones = json.loads(open('data/zonesjson.txt').readline())
    zones = {int(x):records(zones[x], style=zone_clean) for x in zones}
    lvl_score_dct = json.loads(open('data/scorejson.txt').readline())
    lvl_score_dct = {int(x):lvl_score_dct[x] for x in lvl_score_dct}
    def lvlscore(lvl):
        if lvl in lvl_score_dct:
            return lvl_score_dct[lvl]
        else:
            return 0
    for dir_name in ['data/trajs_advancing', 'data/trajs_max']
        rmtree('dir_name')
        os.makedirs('dir_name')
    users_dir = os.listdir('data/users')
    total_user = len(users_dir)
    for file_idx, user_file in enumerate(users_dir):
        if file_idx % 10000 == 0:
            print("file {0}/{1}".format(file_idx, total_user))
        with open(os.path.join('data/users', userfile)) as fp:
            records = [record(x, style='clean') for x in fp.readlines()]
        user = int(user_file)
        lvl_start = records[0].lvl
        lvl_change = {s.lvl:idx for idx in range(len(records)-1) if not records[idx+1].lvl == s.lvl}
        if not lvl_change:
            continue
        lvl_range = {lvl for lvl in lvl_change if lvl - 1 in lvl_change}
        if len(lvl_range) < 5:
            continue
        lvl_gain = {lvl:lvl_score(lvl)/(lvl_change[lvl]-lvl_change[lvl-1]) for lvl in lvl_range}
        previous_zone = 'x'
        zone_session_length = 0
        recent_zones = []
        previous_guild = 0
        guild_age = 0
        previous_time = 0
        session_length = 0
        regular_length = 0
        daily_time = 0
        for s in records:
            # For the day the Lichking update happens
            if Lichking_tt[0] <= s.tt <= Lichking_tt[1]:
                continue
            lvl = s.lvl
            if not (lvl in lvl_range or lvl == 80):
                continue
            else if not lvl == 80:
                fw = open(os.path.join('data/trajs_advancing', userfile), 'a')
            else:
                fw = open(os.path.join('data/trajs_max', userfile), 'a')
            # Dictionary users records the number of replays associated to each user
            users[user] += 1
            zone = s.zone
            # Generate additional features
            zonetype = zones[zone].zonetype
            zonelord = zones[zone].lord
            if zonetype == Zonetypes['Zone'] and zonelord == Lords['Alliance']:
                zonetype = rival
            s.feature_zonetype = zonetype
            recent_zones.append(zone)
            if len(recent_zones) > 60: # i.e. recent 10 hrs
                del(recent_zones[0])
            s.feature_versatile_zones = len(set(recent_zones))
            if zone == previous_zone:
                zone_session_length += 1
            else:
                zone_session_length = 1
                previous_zone = zone
            s.feature_zone_session_length = zone_session_length
            # Generate satisfactions
            reward_advancement = lvl_gain[lvl] if not lvl == 80 else 0
            s.append(reward_advancement)
            if zonetype == Zonetype['Arena']:
                reward_competition = 0.9
            elif zonetype == Zonetype['Battleground']:
                reward_competition = 1.1
            else:
                reward_competition = 0
            s.reward_competition = reward_competition
            current_guild = s.guild
            if previous_guild == current_guild and not current_guild == 0: # guild == 0 iff no guild. check class records with style raw
                guild_age += 1
            else:
                guild_age = 0
                previous_guild = current_guild
            in_guild = not current_guild == 0
            reward_relationship = in_guild * 0.5 + math.sqrt(guild_age) / 150
            s.reward_relationship = reward_teamworkreward_relationship
            if zonetype == Zonetype['Battleground']:
                reward_teamwork = 0.5
            elif zonetype == Zonetype['Dungeon']:
                reward_teamwork = 0.7
            elif zonetype == rival: # zonetype has been modified
                reward_teamwork = 0.9
            elif zonetype == Zonetype['Arena']:
                reward_teamwork = 1.1
            else:
                reward_teamwork = 0 # we temproraly ignore raid, due to some trickiers to get raid label
            s.reward_teamwork = reward_teamwork
            current_time = s[idx].tt
            if current_time <= previous_time + 1:
                session_length += 1
            else:
                session_length = 1
            previous_time = current_time
            if current_time <= daily_time + 288 and current_time >= daily_time + 132 or daily_time == 0:
                regular_length += 1
                daily_time = current_time
            elif current_time > daily_time + 288:
                regular_length = 1
                daily_time = current_time
            reward_escapism = max(regular_length - 10, 0) / 50 + max(session_length - 24, 0) / 12
            s.reward_escapism = reward_escapism
            buf = (s.user, tt, s.guild, s.lvl, s.race, s.category, s.zone, s.feature_zonetype, s.feature_versatile_zones, s.feature_zone_session_length, s.reward_advancement, s.reward_competition, s.reward_relationship, s.reward_teamwork, s.reward_escapism)
            fw.write(','.join([str(x) for x in buf]) + '\n')
            fw.close()
    with open('data/trajsjson.txt', 'w') as fw:
        fw.write(json.dumps({x:users[x] for x in users if not users[x] == 0}))

def user_stats():
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

def traj_stats():
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
    constant_generate()
    #cat_user()
    #s = trajectory()
    #lvls_start, lvls_end, lvls_elapses, scores, elapses = userstats()
    pass
