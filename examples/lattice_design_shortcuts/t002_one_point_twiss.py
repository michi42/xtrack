import xtrack as xt

collider_file = '../../test_data/hllhc15_collider/collider_00_from_mad.json'


# Load the machine and select line
collider= xt.Multiline.from_json(collider_file)

l2 = collider.lhcb2
l2.twiss(betx=1, bety=1, start=xt.START, end=xt.START, reverse=True)