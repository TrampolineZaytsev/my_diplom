def get_center_of_boxx(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]

def get_bbox_hight(bbox):
    return bbox[3] - bbox[1]

def bbox_is_square(bbox):
    return (0.8 < abs((bbox[1] - bbox[3])/(bbox[0] - bbox[2])) < 4)

def get_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
