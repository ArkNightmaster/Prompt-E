def get_model(model_name, args):
    name = model_name.lower()
    if name == "clip_pe":
        from models.clip_visual import Learner
    elif name == "l2p":
        from models.l2p import Learner
    elif name == "dualprompt":
        from models.dualprompt import Learner
    elif name == "lgcl":
        from models.LGCL import Learner
    elif name == "coda_prompt":
        from models.coda_prompt import Learner
    elif name == "clip4coda":
        from models.clip_prompt_coda import Learner
    elif name == "clip4l2p":
        from models.clip_prompt_l2p import Learner
    elif name == 'clip4dual':
        from models.clip_prompt_dual import Learner
    elif name == "finetune":
        from models.finetune import Learner
    elif name == "icarl":
        from models.icarl import Learner
    elif name == "der":
        from models.der import Learner
    elif name == "memo":
        from models.memo import Learner
    elif name == 'lwf':
        from models.lwf import Learner
    else:
        assert 0
    
    return Learner(args)
