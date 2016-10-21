import os
import sys
import math
import pickle
import json
import time
import logging
from itertools import islice
from shutil import rmtree

import numpy as np
import h5py

from logstats import record
from logstats import mkdir
from constant import Lichking_tt
from constant import Categories
from constant_zone import Zonetypes
from constant_zone import Lords

class filter:
    def __init__(self, name, condition):
        self.name = name
        self.condition = condition
        mkdir('data/trajs_{0}'.format(self.name))

    def init_dct(self, base):
        self.dct = {x:0 for x in base}

    def filter(self, s, lvl_range):
        self.active = self.condition(s, lvl_range)

    def write(self, fn, buf):
        if self.active:
            with open(os.path.join('data/trajs_{0}'.format(self.name), fn), 'a') as fw:
                fw.write(','.join([str(x) for x in buf]) + '\n')

    def close(self):
        with open('data/trajsjson_{0}'.format(self.name), 'w') as fw:
            fw.write(json.dumps({x:self.dct[x] for x in self.dct if not self.dct[x] == 0}))

def advancing(s, lvl_range):
    return ((s.lvl < 70 and s.tt < Lichking_tt[0]) or (s.lvl < 80 and s.tt > Lichking_tt[1])) and s.lvl in lvl_range

def max_(s, lvl_range):
    return (s.lvl == 70 and s.tt < Lichking_tt[0]) or (s.lvl == 80 and s.tt > Lichking_tt[1])

def priest(s, lvl_range):
    return advancing(s, lvl_range) and s.category == Categories['Priest']

def warrior(s, lvl_range):
    return advancing(s, lvl_range) and s.category == Categories['Warrior']

def hunter(s, lvl_range):
    return advancing(s, lvl_range) and s.category == Categories['Hunter']

def sats():
    rival = len(Zonetypes)
    users_sketch = json.loads(open('data/usersjson_sketch').readline())
    users_sketch = {int(x):users_sketch[x] for x in users_sketch}
    zones = json.loads(open('data/zonesjson').readline())
    zones = {int(x):record(zones[x], style='zone_clean') for x in zones}
    lvl_score_dct = json.loads(open('data/scorejson').readline())
    lvl_score_dct = {int(x):lvl_score_dct[x] for x in lvl_score_dct}
    def lvl_score(lvl):
        if lvl in lvl_score_dct:
            return lvl_score_dct[lvl]
        else:
            return 0
    filters = (filter('advancing', advancing),
                filter('max', max_),
                filter('priest', priest),
                filter('warrior', warrior),
                filter('hunter', hunter),
              )
    for ff in filters:
        ff.init_dct(users_sketch)
    users_dir = os.listdir('data/users')
    total_user = len(users_dir)
    for file_idx, user_file in enumerate(users_dir):
        if file_idx % 10000 == 0:
            print("file {0}/{1}".format(file_idx, total_user))
        with open(os.path.join('data/users', user_file)) as fp:
            records = [record(x, style='clean') for x in fp.readlines()]
        user = int(user_file)
        lvl_start = records[0].lvl
        lvl_change = {records[idx].lvl:idx for idx in range(len(records)-1) if not records[idx+1].lvl == records[idx].lvl}
        # If the user has experienced the lichking update, we take the update time, and the tiem when the user reaches lvl 70, whichever is chronological larger.
        lk_idx = [idx for idx in range(len(records)-1) if records[idx+1].tt > (Lichking_tt[0]) and records[idx].tt <= (Lichking_tt[0])]
        if 69 in lvl_change and 70 in lvl_change:
            lk_idx.append(lvl_change[69])
        if not lvl_change:
            continue
        lvl_range = {lvl for lvl in lvl_change if lvl - 1 in lvl_change}
        if len(lvl_range) < 5:
            continue
        # lvl -> advancement value gain ON lvl
        lvl_gain = {lvl:lvl_score(lvl)/(lvl_change[lvl]-lvl_change[lvl-1]) for lvl in lvl_range if not lvl == 70}
        if 70 in lvl_range:
            if max(lk_idx) >= lvl_change[70]:
                print(user_file, lk_idx, lvl_change)
                continue
            lvl_gain[70] = lvl_score(70)/(lvl_change[70]-max(lk_idx))
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
            # For the day the Lichking update happens (no longer needed under the filter system)
            #if Lichking_tt[0] <= s.tt <= Lichking_tt[1]:
            #    continue
            #if not lvl == 80 and lvl in lvl_range:
            #    fw = open(os.path.join('data/trajs_advancing', user_file), 'a')
            #    users_advancing[user] += 1
            #elif lvl == 80:
            #    fw = open(os.path.join('data/trajs_max', user_file), 'a')
            #    users_max[user] += 1
            #else:
            #    continue
            for ff in filters:
                ff.filter(s, lvl_range)
            if not any(ff.active for ff in filters):
                continue
            # Generate additional features
            zonetype = zones[s.zone].zonetype
            zonelord = zones[s.zone].lord
            if zonetype == Zonetypes['Zone'] and zonelord == Lords['Alliance']:
                zonetype = rival
            s.feature_zonetype = zonetype
            recent_zones.append(s.zone)
            if len(recent_zones) > 60: # i.e. recent 10 hrs
                del(recent_zones[0])
            s.feature_versatile_zones = len(set(recent_zones))
            if s.zone == previous_zone:
                zone_session_length += 1
            else:
                zone_session_length = 1
                previous_zone = s.zone
            s.feature_zone_session_length = zone_session_length
            # Generate satisfactions
            reward_advancement = lvl_gain[s.lvl] if s.lvl in lvl_gain else 0 # those s.lvl not in lvl_gain can only be 70 or 80
            s.reward_advancement = reward_advancement
            if zonetype == Zonetypes['Arena']:
                reward_competition = 0.9
            elif zonetype == Zonetypes['Battleground']:
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
            s.reward_relationship = reward_relationship
            if zonetype == Zonetypes['Battleground']:
                reward_teamwork = 0.5
            elif zonetype == Zonetypes['Dungeon']:
                reward_teamwork = 0.7
            elif zonetype == rival: # zonetype has been modified
                reward_teamwork = 0.9
            elif zonetype == Zonetypes['Arena']:
                reward_teamwork = 1.1
            else:
                reward_teamwork = 0 # we temproraly ignore raid, due to some trickiers to get raid label
            s.reward_teamwork = reward_teamwork
            current_time = s.tt
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
            # Writing the newly generated data
            buf = (s.user, s.tt, s.guild, s.lvl, s.race, s.category, s.zone, s.feature_zonetype, s.feature_versatile_zones, s.feature_zone_session_length, s.reward_advancement, s.reward_competition, s.reward_relationship, s.reward_teamwork, s.reward_escapism)
            for ff in filters:
                ff.write(user_file, buf)
    for ff in filters:
        ff.close()

def frames(num_frames=10, skip_frames=4, name='advancing', flt=None):
    num_cat_feature = 5
    num_ord_feature = 3
    with open('data/zonesjson') as fp:
        zones = json.loads(fp.readline())
    zones = {int(x):zones[x] for x in zones}
    with open('data/lvlsjson') as fp:
        lvls = json.loads(fp.readline())
    lvls = {int(x):lvls[x] for x in lvls}
    with open('data/transaction/transactionjson10.txt') as fp:
        transactions = json.loads(fp.readline())
    transactions = {int(x):transactions[x] for x in transactions}
    total_zones = max([int(x) for x in zones.keys()]) + 1
    fp = open(trajsjson)
    users = json.loads(fp.readline())
    users = {int(x):users[x] for x in users if users[x] > 2 * num_frames}
    k = np.array(list(users.keys()), dtype=np.uint64)
    p = np.array(list(users.values()), dtype=np.float32)
    norm = (p - num_frames).sum()
    p /= norm
    p /= p.sum()
    fp.close()
    frame = 0
    net_input = np.zeros((num_frames + skip_frames, num_cat_feature + num_ord_feature, 1), dtype=np.float32)
    reward = 0
    action = 0
    power = 2.3
    while True:
        valid = True
        user = rng.choice(k, p=p)
        start = rng.randint(low=0, high=users[user] - num_frames - skip_frames)
        with open(os.path.join('data/trajs', '{0}.txt'.format(user))) as fp:
            for idx, line in enumerate(fp):
                if idx >= start and idx < start + num_frames + skip_frames:
                    idx_logstats, user, tt, guild, lvl, race, category, zone, seq, zonetype, num_zones, zone_stay, r1, r2, r3, r4, r5 = line.strip().split(',')
                    idx_logstats = int(idx_logstats)
                    user = int(user)
                    tt = int(tt)
                    guild = int(guild)
                    lvl = int(lvl)
                    race = int(race)
                    category = int(category)
                    zone = int(zone)
                    seq = int(seq)
                    zonetype = int(zonetype)
                    num_zones = int(num_zones)
                    zone_stay = int(zone_stay)
                    r1 = float(r1) 
                    r2 = float(r2)
                    r3 = float(r3)
                    r4 = float(r4) 
                    r5 = float(r5)
                    if not flt is None:
                        if not flt(idx_logstats, user, tt, guild, lvl, race, category, zone, seq, zonetype, num_zones, zone_stay, r1, r2, r3, r4, r5):
                            valid = False
                            break
                    #print(idx, start, start + num_frames + skip_frames - 1)
                    # possible additional features
                    # the time elapse during current zone
                    # the time elapse during current session
                    # the player total time spending
                    lvl = int(lvl)
                    norm_lvl = lvl / 70
                    norm_idx = min(idx / 1500, 1.1)
                    norm_num_zones = min(int(num_zones) / 12, 1.1)
                    norm_zone_stay = min(int(zone_stay) / 75, 1.1)
                    net_input[idx - start, :, 0] = np.array([int(guild), int(race), int(category), int(zone), int(zonetype), norm_lvl, norm_num_zones, norm_zone_stay])
                    #print([int(guild), int(race), int(category), int(zone), int(zonetype), norm_lvl, norm_num_zones, norm_zone_stay])
                    if idx == start + num_frames - 1:
                        lvl_in = lvl
                    if idx == start + num_frames + skip_frames - 1:
                        lvl_out = lvl
        #reward = np.float32(lvl_out ** power - lvl_in ** power)
        #action = np.argmax(np.bincount(net_input[-skip_frames:, -3, 0].reshape((skip_frames, )).astype(np.uint8)))
        if not valid:
            continue
        reward = np.array([float(x) for x in [r1, r2, r3, r4, r5]])
        action = net_input[-skip_frames:, 3, 0].reshape((skip_frames, )).astype(np.uint8)
        action_set_lvl = lvls[lvl_in]
        last_zone = net_input[-(skip_frames + 1), 3, 0].astype(np.uint8)
        if last_zone in transactions:
            action_set_transaction = transactions[last_zone] + [last_zone, ]
        else:
            print("*zone too minor")
            continue
        if not action[0] in action_set_lvl:
            #print("*overlevel zone")
            #print("actual zones {0}".format(action))
            #print("zone require {0} and current level {1}".format(zones[action[0]][4], lvl_in))
            continue
        action_set = list(set(action_set_lvl) & set(action_set_transaction))
        if not action_set:
            print("*no adjacent zone")
            print(action_set_lvl, action_set_transaction)
            continue
        action_set = np.array(action_set)
        action_set_np = np.array([int(x in action_set) for x in range(total_zones)])
        frame += 1
        if frame == num_frames:
            yield (net_input, reward, action, action_set_np)
            frame = 0
            net_input = np.zeros((num_frames + skip_frames, num_cat_feature + num_ord_feature, 1), dtype=np.float32)
            reward = 0
            action = 0

def batches(trajsjson='data/trajsjson.txt', batch_size=32):
    g = frames(trajsjson=trajsjson)
    batch = []
    idx = 0
    while True:
        batch.append(g.__next__())
        idx += 1
        if idx == batch_size:
            yield [np.array(x) for x in zip(*batch)]
            idx = 0
            batch = []

def hdf_dump(trajsjson='data/trajsjson.txt', path='data/episodes.hdf', size=10000, flt=None):
    cd_size = 0
    s = frames(trajsjson=trajsjson, flt=flt).__next__()
    with open('data/zonesjson.txt') as fp:
        zones = json.loads(fp.readline())
    num_zones = max([int(x) for x in zones.keys()]) + 1
    shape = [x.shape for x in s[:-1]]
    with h5py.File(path, 'w') as fw:
        context = fw.create_dataset('context', (size, *shape[0]), 'f')
        reward = fw.create_dataset('reward', (size, *shape[1]), 'f')
        action = fw.create_dataset('action', (size, *shape[2]), 'i')
        candidates = fw.create_dataset('candidates', (size, num_zones), 'i')
        for idx, frame in enumerate(frames(trajsjson=trajsjson, flt=flt)):
            if idx == size:
                break
            if idx % 1000 == 0 and size > 30000:
                fw.flush()
                print(idx, time.time())
            context[idx] = frame[0]
            reward[idx] = frame[1]
            action[idx] = frame[2]
            candidates[idx] = frame[3]
            cd_size += frame[3].sum()
            #candidates[idx] = np.array([int(x in frame[3]) for x in range(num_zones)])
            #cd_size += len(frame[3])
    #print("averaged #candidate = {0}".format(cd_size / size))

def hdf(path='data/episodes.hdf', batch_size=32, num_batch=1000):
    fp = h5py.File(path)
    context = fp['context']
    assert(context.shape[0] >= batch_size * num_batch)
    reward = fp['reward']
    action = fp['action']
    candidates = fp['candidates']
    for idx in range(num_batch):
        i = slice(idx * batch_size, (idx+1) * batch_size) # keep intact for all other dimensions
        yield (context[i], reward[i], action[i], candidates[i], )
        

if __name__ == "__main__":
    sats()
    #hdf_dump(size=1000000)
    
