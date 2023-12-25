def renormalize(n, from_range, to_range):
    delta1 = from_range[1] - from_range[0]
    delta2 = to_range[1] - to_range[0]
    return (delta2 * (n - from_range[0]) / delta1) + to_range[0]


def rescale_val(val, min, max):
  return (val - min) / (max - min)