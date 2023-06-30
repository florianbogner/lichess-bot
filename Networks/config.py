import torch
import yaml
import os

class Configuration():

    def __init__(self) -> None:
        try:
            with open("./config.yml", "r") as ymlfile:
                cfg = yaml.safe_load(ymlfile)
        except:
            with open("./../config.yml", "r") as ymlfile:
                cfg = yaml.safe_load(ymlfile)

        self.yaml = cfg

        self.wandb_identifier = cfg["wandb_identifier"]


        # data
        
        self.train_path = cfg["data"]["train_path"]
        self.val_path = cfg["data"]["val_path"]
        
        if isinstance(cfg["data"]["shuffle"], bool):
            self.shuffle = cfg["data"]["shuffle"]
        else:
            raise ValueError(f"Error in config: shuffle needs to be boolean.")
        
        if isinstance(cfg["data"]["precompute"], bool):
            self.precompute = cfg["data"]["precompute"]
        else:
            raise ValueError(f"Error in config: precompute needs to be boolean.")
        
        self.data_type = cfg['data']['type']


        # architecture

        legal_modes = {'board+move_to_rating', 'board_to_best_move', 'board+rating_to_move', 'board_to_move_with_rating_loss', 'move_history_to_best_move'}
        if cfg["architecture"]["mode"] in legal_modes:
            self.mode = cfg["architecture"]["mode"]
        else:
            raise ValueError(f"Error in config: Legal values for train_mode: {legal_modes}")
        
        if self.mode == 'board+move_to_rating':
            self.in_channels = 19
            self.out_channels = 1
        elif self.mode == 'board_to_best_move':
            self.in_channels = 17
            self.out_channels = 1858
        elif self.mode == 'board+rating_to_move':
            self.in_channels = 18
            self.out_channels = 1858
        elif self.mode == 'board_to_move_with_rating_loss':
            self.in_channels = 17
            self.out_channels = 1858
        elif self.mode == 'move_history_to_best_move':
            self.in_channels = 17 + 2 * cfg['architecture']['move_history']
            self.out_channels = 1858
        
        self.move_history = cfg['architecture']['move_history']

        
        self.consistency_loss = cfg["architecture"]["consistency_loss"]["enabled"]
        if self.consistency_loss:
            self.consis_loss_weight = cfg["architecture"]["consistency_loss"]["weight"]
            self.consis_loss_type = cfg["architecture"]["consistency_loss"]["type"]
            self.consis_loss_cdf = cfg["architecture"]["consistency_loss"]["use_cdf"]

        legal_models = {'rcnn', 'cnn', 'transformer', 'maia-move', 'multi-move', 'maia-move-u', 'multi-task-move'}
        if cfg["architecture"]["model"] in legal_models:
            self.model = cfg["architecture"]["model"]
        else:
            raise ValueError(f"Error in config: Legal values for model: {legal_models}")
        
        if self.model == 'rcnn':
            self.res_blocks = cfg['architecture']['rcnn']['residual_blocks']
        
        if self.model in {'maia-move', 'maia-move-u'}:
            self.res_blocks = cfg['architecture']['maia-move']['residual_blocks']
            self.hidden_channels = cfg['architecture']['maia-move']['hidden_channels']
        
        if self.model == 'multi-move':
            self.res_blocks = cfg['architecture']['multi-move']['residual_blocks']
            self.hidden_channels = cfg['architecture']['multi-move']['hidden_channels']
            self.output_planes = cfg['architecture']['multi-move']['output_planes']
            self.bins = cfg['architecture']['multi-move']['bins']


        if self.model == 'multi-task-move':
            self.res_blocks = cfg['architecture']['multi-task-move']['residual_blocks']
            self.hidden_channels = cfg['architecture']['multi-task-move']['hidden_channels']
            self.output_planes = cfg['architecture']['multi-task-move']['output_planes']
            self.outcomeLossFn = cfg['architecture']['multi-task-move']['outcomeLossFn']
            self.outcomeLossWeight = cfg['architecture']['multi-task-move']['outcomeLossWeight']
            self.evalLossFn = cfg['architecture']['multi-task-move']['evalLossFn']
            self.evalLossWeight = cfg['architecture']['multi-task-move']['evalLossWeight']


        # training 

        legal_train_modes = {"default", "overfit", "random", "grid", "finetune"}
        if cfg["training"]["train_mode"] in legal_train_modes:
            self.train_mode = cfg["training"]["train_mode"]
        else:
            raise ValueError(f"Error in config: Legal values for train_mode: {legal_train_modes}")
        
        self.pretrained = cfg['training']['pretrained']

        
        self.lr = cfg['training'][self.train_mode]['lr']
        self.weight_decay = cfg['training'][self.train_mode]['weight_decay']
        self.batch_size = cfg["training"][self.train_mode]["batch_size"]
        self.max_epochs = cfg["training"][self.train_mode]["max_epochs"]
        self.patience = cfg["training"][self.train_mode]["early_stopping"]["patience"]
        self.epsilon = cfg["training"][self.train_mode]["early_stopping"]["epsilon"]

        try:
            self.endgame_loss = cfg["training"][self.train_mode]["endgame_loss_factor"]
        except:
            self.endgame_loss = None

        try:
            self.kl_weight = cfg['training'][self.train_mode]['kl_weight']
        except:
            self.kl_weight = None

        if cfg["training"]["scheduler"]["active"]:
            self.scheduler_active = True
            self.scheduler_ms = cfg["training"]["scheduler"]["milestones"]
            self.scheduler_gamma = cfg["training"]["scheduler"]["gamma"]
        else:
            self.scheduler_active = False

        if self.train_mode == 'finetune':
            self.finetune_model = cfg["training"][self.train_mode]["model_path"]
        
        if isinstance(cfg["training"]["silent"], bool):
            self.silent = cfg["training"]["silent"]
        else:
            raise ValueError(f"Error in config: silent needs to be boolean.")

        self.stockfish = cfg["stockfish"]

        # plausibility checks
        if self.data_type == 'np':
            if self.batch_size > 1:
                print("#"*4 + " Batch size is > 1, but data type is numpy")
                print("Batch size will be set to 1")
                self.batch_size = 1
            if self.shuffle:
                print("#"*4 + " Shuffle is set to true, but data type is numpy")
                print("Shuffling will be deactivated, if dataset should be shuffled, generate a shuffled dataset")
                self.shuffle = False


        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"


        self.num_workers = cfg["training"]["num_workers"]

