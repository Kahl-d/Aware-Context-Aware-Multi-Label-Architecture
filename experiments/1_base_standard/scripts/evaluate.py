import argparse,json,logging,sys
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer
from config import StandardConfig,THEMES,NUM_THEMES
from model import StandardModel
from dataset import load_split_data,create_dataloader
from metrics import flatten_masked_logits_labels,compute_metrics,compute_prauc,compute_rocauc,compute_hamming_loss,compute_exact_match_ratio,apply_thresholds,bootstrap_ci
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",handlers=[logging.StreamHandler(sys.stdout)])
logger=logging.getLogger(__name__)
def evaluate_model(model,dl,thr,dev,cal=None):
    model.eval();al,ab=[],[]
    with torch.no_grad():
        for b in dl:
            o=model(b["input_ids"].to(dev),b["attention_mask"].to(dev),b["sentence_boundaries"],b["sentence_mask"].to(dev))
            f,l=flatten_masked_logits_labels(o["logits"],b["labels"].to(dev),b["sentence_mask"].to(dev));al.append(f);ab.append(l)
    ln=torch.cat(al).cpu().numpy();lb=torch.cat(ab).cpu().numpy()
    if cal:
        a=np.array([cal.get(THEMES[i],{}).get("a",1.0) for i in range(NUM_THEMES)]);b2=np.array([cal.get(THEMES[i],{}).get("b",0.0) for i in range(NUM_THEMES)])
        pr=1.0/(1.0+np.exp(-(a[None,:]*ln+b2[None,:])))
    else: pr=1.0/(1.0+np.exp(-ln))
    pd=apply_thresholds(pr,thr);m=compute_metrics(pd,lb);pa=compute_prauc(pr,lb);m["prauc_macro"]=pa["prauc_macro"];m["prauc_per_theme"]=pa["prauc_per_theme"]
    ra=compute_rocauc(pr,lb);m["rocauc_macro"]=ra["rocauc_macro"];m["rocauc_per_theme"]=ra["rocauc_per_theme"]
    m["hamming_loss"]=compute_hamming_loss(pd,lb);m["exact_match_ratio"]=compute_exact_match_ratio(pd,lb)
    md=compute_metrics((pr>=0.5).astype(np.float32),lb);ci=bootstrap_ci(pd,lb,n_bootstrap=1000)
    return {"metrics_optimized":m,"metrics_default_05":md,"thresholds_used":thr,"bootstrap_ci":ci,"n_sentences":int(pr.shape[0])}
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--data_dir",required=True);p.add_argument("--results_dir",required=True);p.add_argument("--split",default="test");p.add_argument("--device",default=None);a=p.parse_args()
    c=StandardConfig.from_yaml(a.config);rd=Path(a.results_dir);model=StandardModel.from_config(c)
    bp=rd/"best.pt"
    if not bp.exists(): print("No best.pt");sys.exit(1)
    dev=torch.device(a.device or("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(torch.load(bp,map_location=dev,weights_only=True));model=model.to(dev)
    thr=json.load(open(rd/"thresholds.json")) if(rd/"thresholds.json").exists() else {t:0.3 for t in THEMES}
    cal=json.load(open(rd/"calibration.json")) if(rd/"calibration.json").exists() else None
    tok=AutoTokenizer.from_pretrained(c.model.encoder_name);data=load_split_data(Path(a.data_dir)/(a.split+"_data.pkl"))
    dl=create_dataloader(data,tok,c,shuffle=False);r=evaluate_model(model,dl,thr,dev,cal)
    with open(rd/("evaluation_"+a.split+".json"),"w") as f: json.dump(r,f,indent=2,default=str)
    m=r["metrics_optimized"];ci=r["bootstrap_ci"]
    print("="*70);print("EVAL",a.split.upper(),r["n_sentences"],"sentences")
    print("F1:",m["f1_macro"],"PRAUC:",m.get("prauc_macro",0),"ROCAUC:",m.get("rocauc_macro",0))
    for t in THEMES: print(t,m["f1_per_theme"].get(t,0),m.get("prauc_per_theme",{}).get(t,0))
    print("="*70)
if __name__=="__main__": main()
