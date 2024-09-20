
def get_ul_key(dict_like, key):
    """Get a value from a dict-like object using a case-insensitive key."""
    key_list = list(dict_like.keys())
    key_list_lower = [k.lower() for k in key_list]
    if key.lower() not in key_list_lower:
        raise KeyError(f"could not find {key} in {dict_like}")
    ind = key_list_lower.index(key.lower())
    return dict_like[key_list[ind]]


def parse_tracer_bin(tracer_bin_key):
    """Takes a string of the form tracer_name_{int} and returns tracer_name, int."""
    key_split = tracer_bin_key.split('_')
    tracer_name = '_'.join(key_split[:-1])
    tracer_bin = int(key_split[-1])
    return tracer_name, tracer_bin


def parse_cl_key(cl_key):
    tracer_bin_keys = cl_key.split(', ')
    return list(map(parse_tracer_bin, tracer_bin_keys))
