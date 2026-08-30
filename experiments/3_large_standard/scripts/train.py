import argparse,json,logging,sys,pickle
from pathlib import Path
import torch
from transformers import AutoTokenizer
from config import StandardConfig,THEMES,NUM_THEMES
from model import StandardModel
from dataset import load_split_data,create_dataloader
from trainer import StandardTrainer
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",handlers=[logging.StreamHandler(sys.stdout)])
logger=logging.getLogger(__name__)
def main():
    p=argparse.ArgumentParser(description="Train Standard Base")
    p.add_argument("--config",type=str,required=True)
    p.add_argument("--data_dir",type=str,required=True)
    p.add_argument("--output_dir",type=str,required=True)
    p.add_argument("--device",type=str,default=None)
    p.add_argument("--toy",action="store_true")
    a=p.parse_args()
    config=StandardConfig.from_yaml(a.config)
    data_dir=Path(a.data_dir);output_dir=Path(a.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    with open(output_dir/"config.json","w") as f: json.dump(config.to_dict(),f,indent=2)
    tokenizer=AutoTokenizer.from_pretrained(config.model.encoder_name)
    if a.toy:
        with open(data_dir/"toy_data.pkl","rb") as f: toy=pickle.load(f)
        train_data,val_data=toy["train"],toy["val"]
    else:
        train_data=load_split_data(data_dir/"train_data.pkl")
        val_data=load_split_data(data_dir/"val_data.pkl")
    logger.info("Train: %d essays, Val: %d essays",len(train_data["essay_ids"]),len(val_data["essay_ids"]))
    tw_list=None
    sp=data_dir/"splits_stats.json"
    if sp.exists():
        with open(sp) as f: stats=json.load(f)
        tw=stats.get("theme_weights",{})
        if tw and config.loss.theme_weights is None:
            tw_list=[tw.get(t,1.0) for t in THEMES]
            logger.info("Theme weights (inverse-sqrt): %s",{t:round(w,3) for t,w in zip(THEMES,tw_list)})
    train_loader=create_dataloader(train_data,tokenizer,config,shuffle=True)
    val_loader=create_dataloader(val_data,tokenizer,config,shuffle=False)
    model=StandardModel.from_config(config)
    trainer=StandardTrainer(model=model,train_loader=train_loader,val_loader=val_loader,output_dir=str(output_dir),config=config,device=a.device,theme_weights_list=tw_list)
    history=trainer.train()
    logger.info("Done. Best PRAUC: %.4f at epoch %d",history.get("best_prauc",0),history.get("best_epoch",0))
if __name__=="__main__": main()
