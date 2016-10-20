import os
import sys
import re
import math
import json
import time
import datetime
from shutil import rmtree
from constant import categories

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
            if cnt[item] <= thres and not item == 'Null':
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

def mkdir(dir_name):
    if os.path.exists(dir_name):
        rmtree(dir_name)
    os.makedirs(dir_name)

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
    zones = json.loads(open('data/zonesjson').readline())
    zones = {int(x):record(zones[x], style='zone_clean') for x in zones}
    rmtree('data/users')
    os.makedirs('data/users')
    user_lvl = {}
    users = {}
    no_data_zones = set()
    for line_idx, line in enumerate(open('data/wowah_dynamic')):
        if line_idx % 1000000 == 0:
            print("line {0}/{1}".format(line_idx, Total_records))
            open('data/usersjson_sketch', 'w').write(json.dumps(users))
        s = record(line, style='raw')
        tt = math.floor((s.timestamp - Min_timestamp) / 600 + 0.5)
        if not s.zone in Zones or not Zones[s.zone] in zones or not s.race in Races or not s.category in Categories:
            if s.zone in Zones:
                no_data_zones.add(s.zone)
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
    open('data/usersjson_sketch', 'w').write(json.dumps(users))
    print('no data:')
    for x in no_data_zones:
        print(x)

def trajs():
    '''
    This generates scorejson which indicate the difficulties of leveling up at each level, and data/transaction/transactionjson* which records the popular transactions; only those popular transactions are served as candidate actions.
    '''
    from constant import Zones
    zonepair = {}
    lvlup = {}
    userlist = os.listdir('data/users')
    for user_idx, user in enumerate(userlist):
        if user_idx % 10000 == 0:
            print(user_idx)
        with open(os.path.join('data/users', user)) as fp:
            for line_idx, line in enumerate(fp):
                s = record(line, style='clean')
                if line_idx == 0:
                    previous_zone = s.zone
                    previous_lvl = s.lvl
                    starting_lvl = s.lvl
                    cnt = 0
                lvl = s.lvl

                if not s.zone == previous_zone:
                    if (previous_zone, s.zone) in zonepair:
                        zonepair[(previous_zone, s.zone)] += 1
                    else:
                        zonepair[(previous_zone, s.zone)] = 1
                    if previous_zone in zonepair:
                        zonepair[previous_zone] += 1
                    else:
                        zonepair[previous_zone] = 1
                    previous_zone = s.zone

                if previous_lvl > starting_lvl or previous_lvl == 1:
                    cnt += 1
                    if s.lvl > previous_lvl:
                        if previous_lvl in lvlup:
                            lvlup[previous_lvl][0] += 1
                            lvlup[previous_lvl][1] += cnt
                            cnt = 0
                        else:
                            lvlup[previous_lvl] = [1, cnt]
                if not s.lvl == previous_lvl:
                    previous_lvl = s.lvl
    lvlscore = {x: lvlup[x][1] / lvlup[x][0] for x in lvlup}
    with open('data/scorejson', 'w') as fw:
        fw.write(json.dumps(lvlscore))
    mkdir('data/transaction')
    for threshold in range(1,20):
        pairs = [x for x in zonepair if type(x) is tuple and zonepair[x]/zonepair[x[0]] > 0.01 * threshold]
        transaction = {}
        for x in pairs:
            if x[0] in transaction:
                transaction[x[0]].append(x[1])
            else:
                transaction[x[0]] = [x[1]]
        with open('data/transaction/transactionjson{0}'.format(threshold), 'w') as fw:
            fw.write(json.dumps(transaction))
        avg = sum(len(transaction[x]) for x in transaction)/len(transaction)
        print('threshold {0}, transactions {1}, #zones {2}/{3} total {4} avg {5}'.format(0.01 * threshold, len(transaction), len([x for x in zonepair if not type(x) is tuple]), len(Zones), len(pairs), avg))

        for zone in Zones:
            if not Zones[zone] in transaction:
                #print(zone) # for debug use only; avoid populate the console
                pass
    return transaction

def user_stats():
    with open('data/usersjson_sketch') as fp:
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
    with open('data/usersjson', 'w') as fw:
        fw.write(json.dumps(users))
    #plt.hist(lvls_start, bins=80)
    #plt.hist(lvls_end, bins=80)
    #plt.hist(elapses, bins=80)
    #plt.scatter(elapses, lvls_end)
    #plt.scatter([min(x, 15000) for x in elapses], scores)
    #plt.show()
    return lvls_start, lvls_end, lvls_elapses, scores, elapses

if __name__ == '__main__':
    if sys.argv[1] == 'constant_generate':
        constant_generate()
    elif sys.argv[1] == 'cat_user':
        cat_user()
    elif sys.argv[1] == 'trajs':
        transactions = trajs()
    elif sys.argv[1] == 'sats':
        sats()
    elif sys.argv[1] == 'user_stats':
        lvls_start, lvls_end, lvls_elapses, scores, elapses = user_stats()
    else:
        assert(0)
