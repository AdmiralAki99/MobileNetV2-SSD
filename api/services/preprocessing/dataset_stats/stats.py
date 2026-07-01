def compute_box_stats(box: tuple[float,float,float,float], image_size: tuple[int,int]):
    
    H, W = image_size
    
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    w_norm = w / W
    h_norm = h / H
    
    return float(w), float(h), float(w_norm), float(h_norm)