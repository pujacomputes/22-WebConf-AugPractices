def train_aug(model,optimizer, scheduler, dataloader, dataloader_eval, args, model_id):
    args.skip_counter = 0
    for epoch in range(1, epochs + 1):
        loss_all = 0
        model.train()

        running_loss = 0.0
        running_backbone_norm = 0.0
        running_encoder_norm = 0.0
        running_predictor_norm = 0.0
        running_backbone_std = 0.0
        running_encoder_std = 0.0
        running_predictor_std = 0.0 

        for data in dataloader:

            data, data_aug = data
            optimizer.zero_grad()

            node_num, _ = data.x.size()
            data = data.to(device)

            if (
                args.aug == "dnodes"
                or args.aug == "subgraph"
                or args.aug == "random2"
                or args.aug == "random3"
                or args.aug == "random4"
            ):
                # node_num_aug, _ = data_aug.x.size()
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [
                    n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])
                ]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
                edge_idx = [
                    [idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]]
                    for n in range(edge_num)
                    if not edge_idx[0, n] == edge_idx[1, n]
                ]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device)

            try:
                z1, z2, p1, p2 = model(data, data_aug)
                loss = model.loss(z1, z2, p1, p2)
                #print(loss.item())

                loss_all += loss.item()
                loss.backward()
                optimizer.step()

                #iteration-level bookkeeping
                model.eval()
                with torch.no_grad():
                    running_loss += loss.item()
                    
                    b1= model.encoder[0](data.x,
                                            data.edge_index,
                                            data.batch)[0]
                    
                    b2 = model.encoder[0](data_aug.x,
                                            data_aug.edge_index, 
                                            data_aug.batch)[0]
                    
                    p1 = model.encoder[1](b1)
                    
                    p2 = model.encoder[1](b2)
                    
                    z1 = model.predictor(p1)
                    z2 = model.predictor(p2)
                    
                    #similarities
                    b_sim = torch.nn.functional.cosine_similarity(b1,b2,dim=1).mean()
                    p_sim = torch.nn.functional.cosine_similarity(p1,p2,dim=1).mean()
                    z_sim = torch.nn.functional.cosine_similarity(z1,z2,dim=1).mean()
                    
                    #norms & std
                    running_backbone_norm += b1.norm(dim=1).mean()
                    running_backbone_std += (b1 / (b1.norm(dim=1,keepdim=True) + 1e-10)).std(dim=0).mean()
                    
                    running_encoder_norm += p1.norm(dim=1).mean()
                    running_encoder_std += (p1 / (p1.norm(dim=1,keepdim=True) + 1e-10)).std(dim=0).mean()
                    
                    running_predictor_norm += z1.norm(dim=1).mean()
                    running_predictor_std += (z1 / (z1.norm(dim=1,keepdim=True)+ 1e-10)).std(dim=0).mean()

                    losses['backbone_sim'].append(b_sim.item())
                    losses['encoder_sim'].append(p_sim.item())
                    losses['predictor_sim'].append(z_sim.item())
                model.train()
            except:
                args.skip_counter += 1
                print("*"*50)
                print(" SKIPPING SAMPLE: {}".format(skip_count))
                print("*"*50)
        
        print()
        print("Epoch {}, Loss {}".format(epoch, loss_all / len(dataloader)))
        print()

        model.eval()     
        if epoch % log_interval == 0:
            emb, y = model.backbone.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies["val"].append(acc_val)
            accuracies["test"].append(acc)
            # print(accuracies['val'][-1], accuracies['test'][-1])
 
        running_loss /= dataloader.__len__()
        running_backbone_norm /=  dataloader.__len__()
        running_backbone_std /=  dataloader.__len__()
        running_encoder_norm /=  dataloader.__len__()
        running_encoder_std /=  dataloader.__len__()
        running_predictor_norm /=  dataloader.__len__()
        running_predictor_std /=  dataloader.__len__()

        losses['epoch_loss'].append(running_loss)
        losses["val_acc"].append(acc_val)
        losses["acc"].append(acc)
        losses['backbone_norm'].append(running_backbone_norm)
        losses['backbone_std'].append(running_backbone_std.item())
        losses['encoder_norm'].append(running_encoder_norm)
        losses['encoder_std'].append(running_encoder_std.item())
        losses['predictor_norm'].append(running_predictor_norm)
        losses['predictor_std'].append(running_predictor_std.item())

        if acc_val > best_epoch:
            best_ckpt["unsupervised_best_epoch"] = epoch
            best_ckpt["val_acc"] = acc_val
            best_ckpt['model'] = model.state_dict()
            best_ckpt["args"] = vars(args)
            best_ckpt['stats'] = losses
            best_epoch = acc_val
            torch.save(best_ckpt, "CKPTS/best_{}.pth".format(model_id))
            print("Epoch {0} -- Best Acc: {1:.4f} ".format(epoch, best_epoch))
    ## make ckpt and save! 
    final_ckpt = {}
    final_ckpt["unsupervised_best_epoch"] = epoch
    final_ckpt["val_acc"] = acc_val
    final_ckpt['model'] = model.state_dict()
    final_ckpt["args"] = vars(args)
    final_ckpt['stats'] = losses
    torch.save(final_ckpt, "CKPTS/final_{}.pth".format(model_id))

    print("TOTAL SKIPS: ",args.skip_counter)

    tpe = ("local" if args.local else "") + ("prior" if args.prior else "")
    with open("logs/log_" + args.DS + "_" + args.aug, "a+") as f:
        s = json.dumps(accuracies)
        f.write(
            "{},{},{},{},{},{},{}\n".format(
                args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s
            )
        )
        f.write("\n")