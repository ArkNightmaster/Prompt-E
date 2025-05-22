import os
import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import numpy as np
import wandb
from torch.utils.data import DataLoader



def train(args):
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["batch_size"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)
    # updata wandb config
    wandb.config.update(args)
    print('================')
    print(wandb.config)
    print('================')
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    val_cnn_curve, val_nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    val_cnn_matrix, val_nme_matrix = [], []

    for task in range(data_manager.nb_tasks):
        print("All params: {}".format(count_parameters(model.network1)))
        print("Trainable params: {}".format(count_parameters(model.network1, True)))
        model.incremental_train(data_manager)
        
        # 处理eval_task返回的四个值
        result = model.eval_task()
        if len(result) == 4:  # 如果返回了四个值（包含验证集结果）
            cnn_accy, nme_accy, val_cnn_accy, val_nme_accy = result
        else:  # 兼容旧版本的模型
            cnn_accy, nme_accy = result
            val_cnn_accy, val_nme_accy = None, None
            
        model.after_task()

        # 获取验证集性能
        val_loader = DataLoader(
            data_manager.get_dataset(np.arange(0, model._total_classes), source="val", mode="val"),
            batch_size=args["batch_size"], 
            shuffle=False,
            num_workers=4
        )
        
        # 评估验证集
        val_cnn_y_pred, val_cnn_y_true = model._eval_cnn(val_loader)
        val_cnn_accy = model._evaluate(val_cnn_y_pred, val_cnn_y_true)
        
        if hasattr(model, "_class_means"):
            val_nme_y_pred, val_nme_y_true = model._eval_nme(val_loader, model._class_means)
            val_nme_accy = model._evaluate(val_nme_y_pred, val_nme_y_true)
        else:
            val_nme_accy = None

        if nme_accy is not None:
            print("CNN: {}".format(cnn_accy["grouped"]))
            print("NME: {}".format(nme_accy["grouped"]))
            print("Validation CNN: {}".format(val_cnn_accy["grouped"]))
            if val_nme_accy is not None:
                print("Validation NME: {}".format(val_nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)
            
            # 验证集矩阵
            val_cnn_keys = [key for key in val_cnn_accy["grouped"].keys() if '-' in key]
            val_cnn_values = [val_cnn_accy["grouped"][key] for key in val_cnn_keys]
            val_cnn_matrix.append(val_cnn_values)
            
            if val_nme_accy is not None:
                val_nme_keys = [key for key in val_nme_accy["grouped"].keys() if '-' in key]
                val_nme_values = [val_nme_accy["grouped"][key] for key in val_nme_keys]
                val_nme_matrix.append(val_nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])
            val_cnn_curve["top1"].append(val_cnn_accy["top1"])
            val_cnn_curve["top5"].append(val_cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])
            if val_nme_accy is not None:
                val_nme_curve["top1"].append(val_nme_accy["top1"])
                val_nme_curve["top5"].append(val_nme_accy["top5"])

            print("CNN top1 curve: {}".format(cnn_curve["top1"]))
            print("CNN top5 curve: {}".format(cnn_curve["top5"]))
            print("Validation CNN top1 curve: {}".format(val_cnn_curve["top1"]))
            print("Validation CNN top5 curve: {}".format(val_cnn_curve["top5"]))
            print("NME top1 curve: {}".format(nme_curve["top1"]))
            print("NME top5 curve: {}\n".format(nme_curve["top5"]))
            if val_nme_accy is not None:
                print("Validation NME top1 curve: {}".format(val_nme_curve["top1"]))
                print("Validation NME top5 curve: {}\n".format(val_nme_curve["top5"]))
            
            print("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
            print("Average Accuracy (NME): {}".format(sum(nme_curve["top1"]) / len(nme_curve["top1"])))
            print("Average Validation Accuracy (CNN): {}".format(sum(val_cnn_curve["top1"]) / len(val_cnn_curve["top1"])))
            if val_nme_accy is not None:
                print("Average Validation Accuracy (NME): {}".format(sum(val_nme_curve["top1"]) / len(val_nme_curve["top1"])))
            
            wandb_log_dict = {
                "Summary/CNN top1": cnn_curve["top1"][-1], 
                "Summary/CNN top5": cnn_curve["top5"][-1], 
                "Summary/NME top1": nme_curve["top1"][-1], 
                "Summary/NME top5": nme_curve["top5"][-1],
                "Summary/Average Accuracy (CNN)": sum(cnn_curve["top1"]) / len(cnn_curve["top1"]),
                "Summary/Average Accuracy (NME)": sum(nme_curve["top1"]) / len(nme_curve["top1"]),
                "Summary/Validation CNN top1": val_cnn_curve["top1"][-1],
                "Summary/Validation CNN top5": val_cnn_curve["top5"][-1],
                "Summary/Average Validation Accuracy (CNN)": sum(val_cnn_curve["top1"]) / len(val_cnn_curve["top1"]),
                "Task": task,
                "Epoch": model.global_step
            }
            
            if val_nme_accy is not None:
                wandb_log_dict.update({
                    "Summary/Validation NME top1": val_nme_curve["top1"][-1],
                    "Summary/Validation NME top5": val_nme_curve["top5"][-1],
                    "Summary/Average Validation Accuracy (NME)": sum(val_nme_curve["top1"]) / len(val_nme_curve["top1"])
                })
                
            wandb.log(wandb_log_dict, step=model.global_step)

        
        else:
            print("No NME accuracy.")
            print("CNN: {}".format(cnn_accy["grouped"]))
            print("Validation CNN: {}".format(val_cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)
            
            val_cnn_keys = [key for key in val_cnn_accy["grouped"].keys() if '-' in key]
            val_cnn_values = [val_cnn_accy["grouped"][key] for key in val_cnn_keys]
            val_cnn_matrix.append(val_cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])
            val_cnn_curve["top1"].append(val_cnn_accy["top1"])
            val_cnn_curve["top5"].append(val_cnn_accy["top5"])

            print("CNN top1 curve: {}".format(cnn_curve["top1"]))
            print("CNN top5 curve: {}".format(cnn_curve["top5"]))
            print("Validation CNN top1 curve: {}".format(val_cnn_curve["top1"]))
            print("Validation CNN top5 curve: {}".format(val_cnn_curve["top5"]))
            print("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
            print("Average Validation Accuracy (CNN): {}".format(sum(val_cnn_curve["top1"]) / len(val_cnn_curve["top1"])))
            
            wandb.log({
                "Summary/CNN top1": cnn_curve["top1"][-1], 
                "Summary/CNN top5": cnn_curve["top5"][-1],
                "Summary/Average Accuracy (CNN)": sum(cnn_curve["top1"]) / len(cnn_curve["top1"]),
                "Summary/Validation CNN top1": val_cnn_curve["top1"][-1],
                "Summary/Validation CNN top5": val_cnn_curve["top5"][-1],
                "Summary/Average Validation Accuracy (CNN)": sum(val_cnn_curve["top1"]) / len(val_cnn_curve["top1"]),
                "Task": task,
                "Epoch": model.global_step
            }, step=model.global_step)

            

    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print(f"Accuracy Matrix (CNN): {np_acctable}")
        print(f"Forgetting (CNN): {forgetting}")
        
        # Log the matrix to wandb
        table = wandb.Table(data=np_acctable.tolist(), columns=[f"Task {i}" for i in range(task + 1)])
        wandb.log(
            {
                "Summary/Accuracy Matrix (CNN)": table,
                "Summary/Forgetting (CNN)": forgetting
            }, step=model.global_step
        )
    
    if len(val_cnn_matrix) > 0:
        val_np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(val_cnn_matrix):
            idxy = len(line)
            val_np_acctable[idxx, :idxy] = np.array(line)
        val_np_acctable = val_np_acctable.T
        val_forgetting = np.mean((np.max(val_np_acctable, axis=1) - val_np_acctable[:, task])[:task])
        print(f"Validation Accuracy Matrix (CNN): {val_np_acctable}")
        print(f"Validation Forgetting (CNN): {val_forgetting}")
        
        # Log the matrix to wandb
        val_table = wandb.Table(data=val_np_acctable.tolist(), columns=[f"Task {i}" for i in range(task + 1)])
        wandb.log(
            {
                "Summary/Validation Accuracy Matrix (CNN)": val_table,
                "Summary/Validation Forgetting (CNN)": val_forgetting
            }, step=model.global_step
        )

    if len(nme_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(nme_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print(f"Accuracy Matrix (NME): {np_acctable}")
        print(f"Forgetting (NME): {forgetting}")
        
        # Log the matrix to wandb
        table = wandb.Table(data=np_acctable.tolist(), columns=[f"Task {i}" for i in range(task + 1)])
        wandb.log(
            {
                "Summary/Accuracy Matrix (NME)": table,
                "Summary/Forgetting (NME)": forgetting
            }, step=model.global_step
        )
        
    if len(val_nme_matrix) > 0:
        val_np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(val_nme_matrix):
            idxy = len(line)
            val_np_acctable[idxx, :idxy] = np.array(line)
        val_np_acctable = val_np_acctable.T
        val_forgetting = np.mean((np.max(val_np_acctable, axis=1) - val_np_acctable[:, task])[:task])
        print(f"Validation Accuracy Matrix (NME): {val_np_acctable}")
        print(f"Validation Forgetting (NME): {val_forgetting}")
        
        # Log the matrix to wandb
        val_table = wandb.Table(data=val_np_acctable.tolist(), columns=[f"Task {i}" for i in range(task + 1)])
        wandb.log(
            {
                "Summary/Validation Accuracy Matrix (NME)": val_table,
                "Summary/Validation Forgetting (NME)": val_forgetting
            }, step=model.global_step
        )
    wandb.finish()




def _set_random(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))