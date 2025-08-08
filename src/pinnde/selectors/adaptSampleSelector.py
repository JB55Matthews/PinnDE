from ..data import pinndata, dondata, invpinndata

def adaptSampleSelector(model, clps_group, data):
    if isinstance(data, pinndata):
        return model(clps_group)
    elif isinstance(data, dondata):
        usensors = data.generate_sensors()
        clps_group.append(usensors)
        return model(clps_group)
    elif isinstance(data, invpinndata):
        return model(clps_group)[0]

    raise TypeError("data passed not a valid type, please check for error internally")