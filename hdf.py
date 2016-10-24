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
            x = int(fn)
            self.dct[x] += 1
            with open(os.path.join('data/trajs_{0}'.format(self.name), fn), 'a') as fw:
                fw.write(','.join([str(x) for x in buf]) + '\n')

    def close(self):
        with open('data/trajsjson_{0}'.format(self.name), 'w') as fw:
            fw.write(json.dumps({x:self.dct[x] for x in self.dct if not self.dct[x] == 0}))

def advancing(s, lvl_range):
    '''
    Between Lichking_tt[0] and Lichking_tt[1] is the day the update happens. The data during that period is eliminated b/c I have no clue what's the exact timing the patch is distributed and if the player can react immediately afterwards
    '''
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
    '''
    possible additional features:
    the time elapse during current zone
    the time elapse during current session
    the player total time spending
    '''
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
            satisfaction_advancement = lvl_gain[s.lvl] if s.lvl in lvl_gain else 0 # those s.lvl not in lvl_gain can only be 70 or 80
            s.satisfaction_advancement = satisfaction_advancement
            if zonetype == Zonetypes['Arena']:
                satisfaction_competition = 0.9
            elif zonetype == Zonetypes['Battleground']:
                satisfaction_competition = 1.1
            else:
                satisfaction_competition = 0
            s.satisfaction_competition = satisfaction_competition
            current_guild = s.guild
            if previous_guild == current_guild and not current_guild == 0: # guild == 0 iff no guild. check class records with style raw
                guild_age += 1
            else:
                guild_age = 0
                previous_guild = current_guild
            in_guild = not current_guild == 0
            satisfaction_relationship = in_guild * 0.5 + math.sqrt(guild_age) / 150
            s.satisfaction_relationship = satisfaction_relationship
            if zonetype == Zonetypes['Battleground']:
                satisfaction_teamwork = 0.5
            elif zonetype == Zonetypes['Dungeon']:
                satisfaction_teamwork = 0.7
            elif zonetype == rival: # zonetype has been modified
                satisfaction_teamwork = 0.9
            elif zonetype == Zonetypes['Arena']:
                satisfaction_teamwork = 1.1
            else:
                satisfaction_teamwork = 0 # we temproraly ignore raid, due to some trickiers to get raid label
            s.satisfaction_teamwork = satisfaction_teamwork
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
            satisfaction_escapism = max(regular_length - 10, 0) / 50 + max(session_length - 24, 0) / 12
            s.satisfaction_escapism = satisfaction_escapism
            # Writing the newly generated data
            buf = (s.user, s.tt, s.guild, s.lvl, s.race, s.category, s.zone, s.feature_zonetype, s.feature_versatile_zones, s.feature_zone_session_length, s.satisfaction_advancement, s.satisfaction_competition, s.satisfaction_relationship, s.satisfaction_teamwork, s.satisfaction_escapism)
            for ff in filters:
                ff.write(user_file, buf)
    for ff in filters:
        ff.close()

def frames(num_frames=10, skip_frames=4, name='advancing', flt=None):
    rng = np.random.RandomState(100)
    num_cat_feature = 5
    num_ord_feature = 3
    with open('data/zonesjson') as fp:
        zones = json.loads(fp.readline())
    zones = {int(x):zones[x] for x in zones}
    with open('data/lvlsjson') as fp:
        lvls = json.loads(fp.readline())
    lvls = {int(x):lvls[x] for x in lvls}
    with open('data/transaction/transactionjson10') as fp:
        transactions = json.loads(fp.readline())
    transactions = {int(x):transactions[x] for x in transactions}
    total_zones = max([int(x) for x in zones.keys()]) + 1
    users = json.loads(open('data/trajsjson_{0}'.format(name)).readline())
    users = {int(x):users[x] for x in users if users[x] > 2 * num_frames}
    keys = np.array(list(users.keys()), dtype=np.uint64)
    values = np.array(list(users.values()), dtype=np.float32)
    values -= num_frames
    pp = values / values.sum()
    frame = 0
    valid = True
    satisfactions = 'x'
    action = 'x'
    net_input = np.zeros((num_frames + skip_frames, num_cat_feature + num_ord_feature, 1), dtype=np.float32)
    while True:
        user = rng.choice(keys, p=pp)
        start = rng.randint(low=0, high=users[user] - num_frames - skip_frames)
        with open('data/trajs_{0}/{1}'.format(name, user)) as fp:
            for line_idx, line in enumerate(fp):
                if not (line_idx >= start and line_idx < start + num_frames + skip_frames):
                    continue
                s = record(line, style='sats')
                if not flt is None:
                    if not flt(s):
                        valid = False
                        break
                norm_lvl = s.lvl / 70
                norm_num_zones = min(s.num_zones / 12, 1.1)
                norm_zone_stay = min(s.zone_stay / 75, 1.1)
                net_input[line_idx - start, :, 0] = np.array([s.guild, s.race, s.category, s.zone, s.zonetype, norm_lvl, norm_num_zones, norm_zone_stay])
                if line_idx == start + num_frames - 1:
                    previous_lvl = s.lvl
        #action = np.argmax(np.bincount(net_input[-skip_frames:, -3, 0].reshape((skip_frames, )).astype(np.uint8)))
        if not valid:
            valid = True
            continue
        satisfactions = np.array([s.advancement, s.competition, s.relationship, s.teamwork, s.escapism])
        action = net_input[-skip_frames:, 3, 0].reshape((skip_frames, )).astype(np.uint8)
        action_set_lvl = lvls[previous_lvl]
        last_zone = net_input[-(skip_frames + 1), 3, 0].astype(np.uint8)
        if last_zone in transactions:
            action_set_transaction = transactions[last_zone] + [last_zone, ]
        else:
            print('*zone too minor {0}'.format(last_zone))
            continue
        if not action[0] in action_set_lvl:
            continue
        action_set = list(set(action_set_lvl) & set(action_set_transaction))
        if not action_set:
            print('*no adjacent zone {0}'.format(last_zone))
            print(action_set_lvl, action_set_transaction)
            continue
        action_set_np = np.array([int(x in action_set) for x in range(total_zones)])
        frame += 1
        if frame == num_frames:
            yield (net_input, satisfactions, action, action_set_np)
            frame = 0
            satisfactions = 'x'
            action = 'x'
            net_input = np.zeros((num_frames + skip_frames, num_cat_feature + num_ord_feature, 1), dtype=np.float32)

def batches(num_frames=10, skip_frames=4, name='advancing', flt=None, batch_size=32):
    g = frames(num_frames=num_frames, skip_frames=skip_frames, name=name, flt=flt)
    batch = []
    cnt = 0
    while True:
        batch.append(g.__next__())
        cnt += 1
        if cnt == batch_size:
            yield [np.array(x) for x in zip(*batch)]
            batch = []
            cnt = 0

def hdf_dump(name='advancing', size=10000, flt=None):
    cd_size = 0
    dp = frames(name=name, flt=flt).__next__()
    with open('data/zonesjson') as fp:
        zones = json.loads(fp.readline())
    num_zones = max([int(x) for x in zones.keys()]) + 1
    shape = [x.shape for x in dp[:-1]]
    with h5py.File('data/episodes_{0}-{1}.hdf'.format(name, size), 'w') as fw:
        context = fw.create_dataset('context', (size, *shape[0]), 'f')
        satisfactions = fw.create_dataset('satisfactions', (size, *shape[1]), 'f')
        action = fw.create_dataset('action', (size, *shape[2]), 'i')
        candidates = fw.create_dataset('candidates', (size, num_zones), 'i')
        for frame_idx, frame in enumerate(frames(name=name, flt=flt)):
            if frame_idx == size:
                break
            if frame_idx % 10000 == 0 and size > 30000:
                fw.flush()
                print(frame_idx, time.time())
            context[frame_idx] = frame[0]
            satisfactions[frame_idx] = frame[1]
            action[frame_idx] = frame[2]
            candidates[frame_idx] = frame[3]

def hdf(path, batch_size=32, num_batch=1000):
    fp = h5py.File(path)
    context = fp['context']
    assert(context.shape[0] >= batch_size * num_batch)
    satisfactions = fp['satisfactions']
    action = fp['action']
    candidates = fp['candidates']
    for idx in range(num_batch):
        i = slice(idx * batch_size, (idx+1) * batch_size) # index over 1st dim while keep intact for all other dimensions
        yield (context[i], satisfactions[i], action[i], candidates[i], )

if __name__ == "__main__":
    if sys.argv[1] == 'sats':
        sats()
    elif sys.argv[1] == 'hdf_dump':
        hdf_dump(name='advancing', size=10000)
    
