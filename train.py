# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data import MyDataset, load_embeddings
from utils import train, validate
from model import SiaGRU


def main(train_file, dev_file,
         embeddings_file, vocab_file,
         target_dir,max_length=30,
         epochs=50,batch_size=128,
         lr=0.0005,patience=5,
         max_grad_norm=5.0,checkpoint=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # -------------------- 加载数据 ------------------- #
    print("\t* Loading training data...")
    train_data = MyDataset(train_file, vocab_file, max_length)#加载训练数据
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)#构造迭代器
    print("\t* Loading validation data...")
    dev_data = MyDataset(dev_file, vocab_file, max_length)#加载验证集数据
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)#构造迭代器
    # -------------------- 定义模型 ------------------- #
    print("\t* Building model...")
    embeddings = load_embeddings(embeddings_file)#加载预训练字向量
    model = SiaGRU(embeddings, max_len=max_length,device=device).to(device)#实例化SiaGRU模型
    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    # 过滤出需要梯度更新的参数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)#构造优化器
    #构造学习率衰减器  动态学习率技术
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",factor=0.85, patience=0)

    # -------------------- 加载模型 ------------------- #
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count,train_losses,valid_losses = [],[],[]
    if checkpoint:#如果模型文件存在，就加载模型继续训练
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, auc = validate(model, dev_loader, criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}".format(valid_loss,
                                                                                               (valid_accuracy * 100),
                                                                                               auc))





    # -------------------- 训练epochs ------------------- #
    print("\n", 20 * "=", "Training SiaGRU model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        #执行一个批次训练
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer,criterion, max_grad_norm)
        train_losses.append(epoch_loss)#保存损失
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%".format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        print("* Validation for epoch {}:".format(epoch))
        #完成一次验证
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc = validate(model, dev_loader, criterion)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n".format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc))
        scheduler.step(epoch_accuracy)#学习率衰减
        if epoch_accuracy < best_score:#判断当前轮次成绩是否比最有成绩差，如果差，计数器+1
            patience_counter += 1
        else:
            best_score = epoch_accuracy#本轮成绩最优
            patience_counter = 0
            #保存最优成绩的模型
            torch.save({"epoch": epoch,"model": model.state_dict(),
                        "best_score": best_score,"epochs_count": epochs_count,
                        "train_losses": train_losses,"valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))
        #保存当前轮次模型
        torch.save({"epoch": epoch,"model": model.state_dict(),
                    "best_score": best_score,"optimizer": optimizer.state_dict(),
                    "epochs_count": epochs_count,"train_losses": train_losses,
                    "valid_losses": valid_losses},
                   os.path.join(target_dir, "SiaGRU_{}.pth.tar".format(epoch)))
        #如果计数器>早停阈值，停止训练
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break



if __name__ == "__main__":
    main("data/train.csv", "data/dev.csv",
         "data/token_vec_300.bin", "data/vocab.txt", "models",checkpoint='models/SiaGRU_31.pth.tar')
