def get_center_of_boxx(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]

def bbox_is_square(bbox):
    return (0.8 < abs((bbox[1] - bbox[3])/(bbox[0] - bbox[2])) < 4)

