import numpy as np

from skimage.measure import regionprops, find_contours

def to_coco(lbl,coloring):
# if True:
    
    instances = []
    props = regionprops(lbl)
    for elt in props:
        r1,c1,r2,c2=elt.bbox
        clr = 0 if elt.label not in coloring else coloring[elt.label]-1 # 0=>bg, 1=> islands, 2 onwards=>clumps
        # bound_img = find_boundaries(lbl==elt.label,mode='inner',connectivity=2)
        poly_rc = find_contours(lbl==elt.label)[0] # one instance should have only one polygon
        poly_xy = [np.flip(np.array(poly_rc),axis=1).tolist()]
        
        if len(poly_rc)<5:
            print('skipping small polygon')
            continue
        inst = {
            'label':elt.label,
            'bbox':[c1,r1,c2,r2],
            'bbox_mode':0, 
            'category_id':clr,
            'segmentation':poly_xy,
            # 'iscrowd':0,
            }
        instances.append(inst)
        
    return {
        'file_name':None, # filled by caller
        # 'height':lbl.shape[0], 'width':lbl.shape[1],
        'image_id':None, # filled by caller
        'annotations':instances
    }

def coco_records_to_json(records, categories_list):
    coco = {
        "info":{},
        "images":[],
        "annotations":[],
        "categories":[{'id':k+1,'name':v,'supercategory':'cell'} for k,v in enumerate(categories_list)]
    }
    
    ann_id_last = 0
    for elt in records:
        
        coco['images'].append({"id":elt['image_id'],'file_name':elt['file_name'],"height":elt['height'], "width":elt['width']})
        for annot in elt['annotations']:
            annot['image_id']=elt['image_id']
            annot['segmentation']=[np.array(annot['segmentation']).ravel().tolist()]
            annot['id']=ann_id_last
            annot['category_id']=int(annot['category_id'])+1
            xmin,ymin,xmax,ymax = annot['bbox']
            annot['bbox']=(xmin,ymin,xmax-xmin,ymax-ymin)
            annot['bbox_mode']=1 # xywh
            coco['annotations'].append(annot)
            ann_id_last+=1
    
    return coco
