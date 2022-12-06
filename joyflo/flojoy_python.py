from box import Box
import numpy as np
import traceback
import yaml
import json
import re
from pathlib import Path
import networkx as nx
from redis import Redis
from rq.job import Job
import os
from functools import wraps
from .utils import PlotlyJSONEncoder
import requests
from dotenv import dotenv_values


def get_port():
    try:
        p = dotenv_values()['REACT_APP_BACKEND_PORT']
    except:
        p = '8000'
    return p


port = get_port()

REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', 6379)
r = Redis(host=REDIS_HOST, port=REDIS_PORT)


class VectorXY(Box):
    '''
    A class for x-y paired numpy arrays that supports dot assignment

    Usage
    -----
    import numpy as np

    v = VectorXY()

    v.x = np.linspace(1,20,0.1)
    v.y = np.sin(v.x)
    '''

    @staticmethod
    def _ndarrayify(value):
        s = str(type(value))
        v_type = s.split("'")[1]

        match v_type:
            case 'int' | 'float':
                value = np.array([value])
            case 'list':
                value = np.array(value)
            case 'numpy.ndarray':
                pass
            case 'NoneType':
                pass
            case _:
                raise ValueError(value)
        return value

    def __init__(self, **kwargs):
        if 'x' in kwargs:
            self['x'] = self._ndarrayify(kwargs['x'])
        else:
            self['x'] = np.array([])

        if 'y' in kwargs:
            self['y'] = self._ndarrayify(kwargs['y'])
        else:
            self['y'] = np.array([])

    def __getitem__(self, key, **kwargs):
        if key not in ['x', 'y']:
            raise KeyError(key)
        elif key == '_ignore_default':
            pass
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key not in ['x', 'y']:
            raise KeyError(key)
        else:
            value = self._ndarrayify(value)
            super().__setitem__(key, value)


def get_flojoy_root_dir():
    home = str(Path.home())
    # TODO: Upate shell script to add ~/.flojoy/flojoy.yaml
    path = os.path.join(home, '.flojoy/flojoy.yaml')
    stream = open(path, 'r')
    yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)
    root_dir = ''
    if isinstance(yaml_dict, str) == True:
        root_dir = yaml_dict.split(':')[1]
    else:
        root_dir = yaml_dict['PATH']
    return root_dir


def js_to_json(s):
    '''
    Converts an ES6 JS file with a single JS Object definition to JSON
    '''
    split = s.split('=')[1]
    clean = split.replace('\n', '').replace(
        "'", '').replace(',}', '}').rstrip(';')
    single_space = ''.join(clean.split())
    dbl_quotes = re.sub(r'(\w+)', r'"\1"', single_space).replace('""', '"')
    rm_comma = dbl_quotes.replace('},}', '}}')

    return json.loads(rm_comma)


def get_parameter_manifest():
    root = get_flojoy_root_dir()
    f = open(os.path.join(root, 'src/data/manifests-latest.json'))
    param_manifest = json.load(f)
    return param_manifest['parameters']


def fetch_inputs(previous_job_ids, mock=False):
    '''
    Queries Redis for job results

    Parameters
    ----------
    previous_job_ids : list of Redis job IDs that directly precede this node

    Returns
    -------
    inputs : list of VectorXY objects
    '''
    if mock is True:
        return [VectorXY(x=np.linspace(0, 10, 100))]

    inputs = []

    try:
        for ea in previous_job_ids:
            job = Job.fetch(ea, connection=Redis(
                host=REDIS_HOST, port=REDIS_PORT))
            inputs.append(job.result)
    except Exception:
        print(traceback.format_exc())

    return inputs


def get_redis_obj(id):
    get_obj = r.get(id)
    parse_obj = json.loads(get_obj) if get_obj is not None else None
    return parse_obj


def flojoy(func):
    '''
    Decorator to turn Python functions with numerical return
    values into Flojoy nodes.

    @flojoy is intended to eliminate  boilerplate in connecting
    Python scripts as visual nodes 

    Into whatever function it wraps, @flojoy injects
    1. the last node's input as an XYVector
    2. parameters for that function (either set byt the user or default)

    Parameters
    ----------
    func : Python function object

    Returns
    -------
    VectorYX object 

    Usage Example
    -------------

    @flojoy
    def SINE(v, params):

        print('params passed to SINE', params)

        output = VectorXY(
            x=v[0].x, 
            y=np.sin(v[0].x)
        )
        return output

    pj_ids = [123, 456]

    # equivalent to: decorated_sin = flojoy(SINE)
    print(SINE(previous_job_ids = pj_ids, mock = True))    
    '''
    @wraps(func)
    # def wrapper(previous_job_ids, mock):
    def wrapper(*args, **kwargs):
        def send_to_socket(data):
            requests.post('http://localhost:'+port +
                          '/worker_response', json=data)
        try:
            previous_job_ids, mock = {}, False
            if 'previous_job_ids' in kwargs:
                previous_job_ids = kwargs['previous_job_ids']
            if 'ctrls' in kwargs:
                ctrls = kwargs['ctrls']
            node_id = kwargs['node_id']
            jobset_id = kwargs['jobset_id']
            FN = func.__name__
            # remove this node from redis ALL_NODES key
            r.lrem(jobset_id+'_ALL_NODES', 1, FN)
            sys_status = '🏃‍♀️ Running python job: ' + FN
            send_to_socket(json.dumps({
                'SYSTEM_STATUS': sys_status,
                'jobsetId': jobset_id,
                'RUNNING_NODE': FN
            }))
            # Get default command paramaters
            default_params = {}
            func_params = {}
            pm = get_parameter_manifest()
            if FN in pm:
                for param in pm[FN]:
                    default_params[param] = pm[FN][param]['default']
                # Get command parameters set by the user through the control panel
                func_params = {}
                if ctrls is not None:
                    for key, input in ctrls.items():
                        func_params[input['param']] = input['value']

                # Make sure that function parameters set is fully loaded
                # If function is missing a parameter, fill-in with default value
                for key in default_params.keys():
                    if key not in func_params.keys():
                        func_params[key] = default_params[key]

            node_inputs = fetch_inputs(previous_job_ids, mock)
            result = func(node_inputs, func_params)
            send_to_socket(json.dumps({
                'NODE_RESULTS': {'cmd': FN, 'id': node_id, 'result': result},
                'jobsetId': jobset_id
            }, cls=PlotlyJSONEncoder))
            all_nodes_length = r.llen(jobset_id + '_ALL_NODES')
            if all_nodes_length == 0:
                send_to_socket(json.dumps({
                    'SYSTEM_STATUS': '🤙 python script run successful',
                    'RUNNING_NODE': '',
                    'jobsetId': jobset_id
                }))
            return result
        except Exception:
            send_to_socket(json.dumps({
                'SYSTEM_STATUS': 'Failed to run: ' + func.__name__,
                'FAILED_NODES': func.__name__,
                'jobsetId': jobset_id
            }))
            print(traceback.format_exc())
            raise

    wrapper.original = func
    wrapper.original.__qualname__ += ".original"

    return wrapper


def reactflow_to_networkx(elems):
    DG = nx.DiGraph()
    for i in range(len(elems)):
        el = elems[i]
        if 'source' not in el:
            data = el['data']
            ctrls = data['ctrls'] if 'ctrls' in data else {}
            DG.add_node(
                i+1, pos=(el['position']['x'], el['position']['y']), id=el['id'], ctrls=ctrls)
            elems[i]['index'] = i+1
            elems[i]['label'] = el['id'].split('-')[0]
    pos = nx.get_node_attributes(DG, 'pos')

    # Add edges to networkx directed graph

    def get_tuple(edge):
        e = [-1, -1]
        src_id = edge['source']
        tgt_id = edge['target']
        # iterate through all nodes looking for matching edge
        for el in elems:
            if 'id' in el:
                if el['id'] == src_id:
                    e[0] = el['index']
                elif el['id'] == tgt_id:
                    e[1] = el['index']
        return tuple(e)

    for i in range(len(elems)):
        el = elems[i]
        if 'source' in el:
            # element is an edge
            e = get_tuple(el)
            DG.add_edge(*e)

    # Add labels (commands) to networkx nodes

    labels = {}

    for el in elems:
        # if element is not a node
        if 'source' not in el:
            labels[el['index']] = el['data']['func']

    nx.set_node_attributes(DG, labels, 'cmd')
    nx.draw(DG, pos, with_labels=True, labels=labels)

    def get_node_data_by_id():
        nodes_by_id = dict()
        for n, nd in DG.nodes().items():
            if n is not None:
                nodes_by_id[n] = nd
        return nodes_by_id
    sort = nx.topological_sort(DG)

    return {'topological_sort': sort, 'getNode': get_node_data_by_id, 'DG': DG}
