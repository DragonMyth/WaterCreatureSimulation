import numpy as np
import pydart2 as pydart
import itertools


def set_constraint_for_flat_model(skel):
    bodynodes_dict = construct_skel_dict(skel)
    comb = []
    for i in itertools.product(['l', 'r'], [1, 2, 3]):
        comb.append(i)
    for segIdx in range(5):
        for side, idx in comb:
            curr_key = 'wing_' + str(side) + '_' + str(segIdx) + str(idx)
            next_key = 'wing_' + str(side) + '_' + str(segIdx + 1) + str(idx)
            curr_body = bodynodes_dict[curr_key]
            next_body = bodynodes_dict[next_key]
            midPint = [(curr_body.C[0] + next_body.C[0]) / 2, (curr_body.C[1] + next_body.C[1]) / 2,
                       (curr_body.C[2] + next_body.C[2]) / 2]
            pydart.constraints.BallJointConstraint(curr_body, next_body, midPint)


def construct_skel_dict(skel):
    node_dict = {}
    bodynodes = skel.bodynodes
    for i in range(len(bodynodes)):
        node_dict[bodynodes[i].name] = bodynodes[i]

    return node_dict
