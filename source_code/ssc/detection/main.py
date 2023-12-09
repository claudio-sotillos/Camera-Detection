import os
from pytorch_lightning import Trainer
from util import init_exp_folder, Args
from util import constants as C
from lightning import (get_task, load_task, get_trained_task, get_ckpt_callback, get_early_stop_callback,
                       get_logger)
import torch
import time


def train(save_dir=C.SANDBOX_PATH,
          tb_path=C.TB_PATH,
          exp_name="DemoExperimentTrain_2pac2", # NAME #################
          model="FasterRCNN",
          task='detection',
          gpus=1,
          pretrained=True,  ## this does not have any effect, since the logic inside the code is commented
          batch_size=8,
          accelerator="gpu",  ## changing from ddp to dp
          gradient_clip_val=0.5,
          max_epochs=45,
          learning_rate=1e-5,
          patience=20,  # 30
          limit_train_batches=1.0,
          limit_val_batches=1.0,
          limit_test_batches=1.0,
          weights_summary="top", # None
          ):
    """
    Run the training experiment.

    Args:
        save_dir: Path to save the checkpoints and logs
        exp_name: Name of the experiment
        model: Model name
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
        pretrained: Whether or not to use the pretrained model
        num_classes: Number of classes
        accelerator: Distributed computing mode
        gradient_clip_val:  Clip value of gradient norm
        limit_train_batches: Proportion of training data to use
        max_epochs: Max number of epochs
                patience: number of epochs with no improvement after
                                  which training will be stopped.
        tb_path: Path to global tb folder
        loss_fn: Loss function to use
        weights_summary: Prints a summary of the weights when training begins.

    Returns: None

    """
    num_classes = 2
    dataset_name = "camera-detection-new"

    args = Args(locals())
    init_exp_folder(args)

    # task = get_task(args) ## uncomment if you want the original version of the code

    # this modified version, allows to import the already trained checkpoint for training again with all the
    # arguments passed
    task = get_trained_task(args, ckpt_path="/home/hinux/Desktop/project04/checkpoint/pac.ckpt")
    # task = get_trained_task(args, ckpt_path="/home/hinux/Desktop/project04/ssc/detection/sandbox/DemoExperimentTrain_Need4Speed_2/my_ck.ckpt")
    # print(task.model)

    # for i in task.model.parameters():  # this is important since this example shows the "requires_grad=True"
    #     print(i)

    for name, param in task.model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # for name, param in task.model.named_parameters():
    #     print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    trainer = Trainer(gpus=gpus,
                      accelerator=accelerator,
                      logger=get_logger(save_dir, exp_name),
                      callbacks=[get_early_stop_callback(patience),
                                 get_ckpt_callback(save_dir, exp_name, monitor="mAP", mode="max")],
                      weights_save_path=os.path.join(save_dir, exp_name),
                      gradient_clip_val=gradient_clip_val,
                      limit_train_batches=limit_train_batches,
                      limit_val_batches=limit_val_batches,
                      limit_test_batches=limit_test_batches,
                      weights_summary=weights_summary,
                      max_epochs=max_epochs)
    trainer.fit(task)
    return save_dir, exp_name


def test(ckpt_path,
         visualize=True,
         deploy=True,
         limit_test_batches=1.0,
         gpus=1,
         deploy_meta_path= "/home/hinux/Desktop/project04/checking_folder/TEST_200.csv", #"/home/hinux/Desktop/google_images/found_images_9.csv",
         # This path should be changed when you chose which csv file you want to test

         # deploy_meta_path="/home/hinux/Desktop/project04/ssc/data/Vedran_images/VEDRAN_IMAGES.csv"
         test_batch_size=1,
         **kwargs):
    """
    Run the testing experiment.

    Args:
        ckpt_path: Path for the experiment to load
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
    Returns: None

    """
    task = load_task(ckpt_path,
                     visualize=visualize,
                     deploy=deploy,
                     deploy_meta_path=deploy_meta_path,
                     test_batch_size=test_batch_size,
                     **kwargs)
    trainer = Trainer(gpus=gpus,
                      limit_test_batches=limit_test_batches)

    trainer.test(task)


# def nni():
#     run_nni(train, test)


if __name__ == "__main__":
    # Record the start time
    start_time = time.time()

    # Your program code goes here

    # train()  # change ssc/detection/data/__init__.py

    # change ('deploy_meta_path' in this file and ssc/detection/data/__init__.py)
    # must have same csv.file so it can test

    # test(ckpt_path="/home/hinux/Desktop/project04/checkpoint/best.ckpt") # the best checkpoint, the original version
    test(ckpt_path="/home/hinux/Desktop/project04/ssc/detection/sandbox/DemoExperimentTrain_2pac2/ckpts.ckpt")

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = (end_time - start_time)/60.0

    print(f"Program ran for {elapsed_time:.3f} minutes")


