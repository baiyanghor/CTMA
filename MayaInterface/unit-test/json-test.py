import os
from json import load, dump
jfile = 'test.json'

if os.path.isfile(jfile):
    print "File exists"
else:
    print "Oops"

with open(jfile, 'r') as fp:
    j_data = load(fp)
    
j_data['3000'] = {"1": {"operation_name": "tag_points"}}
op = j_data['2000']
op['4'] = {"operation_name": "new_session"} 
    
with open(jfile, 'w') as wfp:
    dump(j_data, wfp)