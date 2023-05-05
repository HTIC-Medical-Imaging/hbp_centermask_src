from tqdm import tqdm
from functions import get_coloring
from dataconverters import to_cocorecord


def get_records(datasource, offset=None,limit=None):
    
    records = []
    start=0
    end=len(datasource)
    if offset is not None:
        start = offset
    if limit is not None:
        end = min(start+limit,end)
        
    for ii in tqdm(range(start,end)):
        img,lbl=datasource.load_image(ii)
        lbl_cl, msk_2c, islands, gr, coloring = get_coloring(lbl)
        # print(coloring)
        rec = to_cocorecord(lbl,coloring)
        rec['file_name']=datasource.imgname(ii)
        rec['image_id']=ii
        
        records.append(rec)
    return records

