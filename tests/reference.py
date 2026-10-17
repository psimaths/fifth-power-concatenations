"""Independent Python-integer oracle for the small diagnostic domain."""


def contribution(position, value):
    return value**5 - 100000**position * value


def expected(part):
    pairs = {}
    correction = 3 * ((7 * part) % 10)
    for i in range(4):
        d = 10000 + part + 10 * i
        for j in range(40):
            e = 10000 + j
            value = contribution(1, d) + contribution(0, e)
            assert (value - correction) % 30 == 0
            key = ((value - correction) // 30) % 1009
            pairs.setdefault(key, []).append((40 * i + j, d, e, value))
    result = dict(queries=0, candidates=0, checksum=0, passed13=0,
                  solutions_count=0, solutions=[])
    for row in range(400):
        a, b = 10000 + row // 20, 10000 + row % 20
        for decade in range(4):
            c = 10000 + 10 * decade + (-a - b - part) % 10
            target = -contribution(4, a) - contribution(3, b) - contribution(2, c)
            assert (target - correction) % 30 == 0
            key = ((target - correction) // 30) % 1009
            result["queries"] += 1
            for pair_id, d, e, value in pairs.get(key, []):
                result["candidates"] += 1
                result["checksum"] = (result["checksum"] + (pair_id + 1) * (row * 4 + decade + 1)) % 2**64
                result["passed13"] += (value - target) % 13 == 0
                if value == target:
                    result["solutions_count"] += 1
                    result["solutions"].append([a, b, c, d, e])
    return result
