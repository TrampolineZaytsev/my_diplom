def distance_color(color1, color2):
    dist = 0
    for i in range(3):
        dist += (color1[i]-color2[i])**2
    dist **= 0.5
    return dist

