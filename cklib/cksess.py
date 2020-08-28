import numpy as np
import itertools

def flatten(__list__):
    return list(itertools.chain.from_iterable(__list__))

def get_flows(dataset):
    first_packets = np.arange(len(dataset))[dataset[:, 0] == 0]
    
    flows = [list(range(first_packets[i - 1], first_packets[i])) for i in range(1, len(first_packets))]
    flows.append(list(range(first_packets[-1], len(dataset))))
    
    for flow in flows:
        assert(dataset[flow[0]][0] == 0)
        assert(dataset[flow[-1]][0] != 0)
    
    return flows

def shuffle_flow_based_dataset(dataset, flows, random_state = None):
    state = np.random.get_state()
    
    np.random.seed(random_state)
    np.random.shuffle(flows)
    np.random.set_state(state)
    
    return dataset[flatten(flows)]

def shuffle_slot_based_dataset(dataset, session, flows, random_state = None):
    state = np.random.get_state()
    
    np.random.seed(random_state)
    np.random.shuffle(flows)
    np.random.set_state(state)
    
    return dataset[flatten(flows)], session[flatten(flows)]

def shuffle_flow(dataset, flows, random_state = None):
    state = np.random.get_state()
    
    np.random.seed(random_state)
    np.random.shuffle(flows)
    np.random.set_state(state)
    
    return dataset[flatten(flows)], flows