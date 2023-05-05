import numpy as np

from skimage.measure import regionprops, find_contours

def to_cocorecord(lbl,coloring={}):
    # for one <image , mask> pair : lbl is mask
    
    nocoloring = len(coloring)==0
          
    instances = []
    props = regionprops(lbl)
    for elt in props:
        r1,c1,r2,c2=elt.bbox
        
        if nocoloring: 
            clr = 1
        else:
            clr = -1 if elt.label not in coloring else coloring[elt.label] # 0=>bg, 1=> islands, 2 onwards=>clumps
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
        'height':lbl.shape[0], 'width':lbl.shape[1],
        'image_id':None, # filled by caller
        'annotations':instances
    }


def cocorecords_to_json(records, categories_list):
    # https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html
    coco = {
        "info":{},
        "images":[],
        "annotations":[],
        "categories":[{'id':k,'name':v,'supercategory':'cell'} for k,v in enumerate(categories_list)]
    }
    
    ann_id_last = 0
    for elt in records:
        
        coco['images'].append({"id":elt['image_id'],'file_name':elt['file_name'],"height":elt['height'], "width":elt['width']})
        for annot in elt['annotations']:
            annot['image_id']=elt['image_id']
            annot['segmentation']=[np.array(annot['segmentation']).ravel().tolist()]
            annot['id']=ann_id_last
            # annot['category_id']=int(annot['category_id'])+1
            xmin,ymin,xmax,ymax = annot['bbox']
            annot['bbox']=(xmin,ymin,xmax-xmin,ymax-ymin)
            annot['bbox_mode']=1 # xywh
            coco['annotations'].append(annot)
            ann_id_last+=1
    
    return coco


def to_onehot(lbl_cl,max_label=8):
    shp = lbl_cl.shape
    dims = shp[:2]
    if dims[0]==1:
        dims = shp[1:3]
    mx = lbl_cl.max()
    max_label=max(max_label,mx)
    oharray = np.zeros((dims[0],dims[1],max_label),dtype=np.uint8)
    for lv in range(1,mx+1):
        msk = lbl_cl==lv
        oharray[...,lv-1]=msk
    return oharray

